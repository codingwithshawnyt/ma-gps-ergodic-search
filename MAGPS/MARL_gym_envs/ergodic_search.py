"""
Multi-Agent Ergodic Search Environment (v3)

Key insight: The SMC controller's control law is u* ∝ Σ_k λ_k(c_k - φ_k)∇f_k(x).
At episode start (c_k≈0), this simplifies to u* ∝ -Σ_k λ_k φ_k ∇f_k(x),
which is the gradient of the λ-weighted Fourier reconstruction of the target PDF.

This is a BROAD, SMOOTH attractor field that covers the ENTIRE space (unlike
Gaussians which die off exponentially). MA-GPS's behavior cloning loss then
pushes the learned policy toward this SMC-like behavior.

Changes from v2:
- LQ cost gradient = SMC Fourier reconstruction gradient (precomputed, non-JIT)
- Reward: PDF value + exploration bonus only (no repulsion/velocity exploit)
- Minimal inter-agent repulsion in LQ cost (β=0.5 not 5.0)
"""

from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
from scipy.stats import multivariate_normal as mvn


class ErgodicSearchEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    PEAKS = [
        {"mean": np.array([0.35, 0.38]), "cov": np.array([[0.01, 0.004], [0.004, 0.01]]), "weight": 0.5},
        {"mean": np.array([0.68, 0.25]), "cov": np.array([[0.005, -0.003], [-0.003, 0.005]]), "weight": 0.2},
        {"mean": np.array([0.56, 0.64]), "cov": np.array([[0.008, 0.0], [0.0, 0.004]]), "weight": 0.3},
    ]

    def __init__(
        self,
        num_agents: int = 2,
        num_k_per_dim: int = 10,
        dt: float = 0.05,
        max_episode_steps: int = 1000,
        action_limit: float = 2.0,
        velocity_limit: float = 0.5,
        damping: float = 0.5,
        reward_pdf_weight: float = 10.0,
        reward_control_weight: float = 0.1,
        init_pos_low: float = 0.2,
        init_pos_high: float = 0.8,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.num_agents = num_agents
        self.num_players = num_agents
        self.num_k_per_dim = num_k_per_dim
        self.dt = dt
        self.max_episode_steps = max_episode_steps
        self.action_limit = action_limit
        self.velocity_limit = velocity_limit
        self.damping = damping
        self.init_pos_low = init_pos_low
        self.init_pos_high = init_pos_high
        self.render_mode = render_mode
        self.w_pdf = reward_pdf_weight
        self.w_ctrl = reward_control_weight

        # Per-agent state: [x, y, vx, vy]
        self.nx = 4
        self.nu = 2
        self.total_state_dim = self.nx * self.num_players
        self.total_action_dim = self.nu * self.num_players

        self.players_u_index_list = torch.tensor(
            [[i * 2, i * 2 + 1] for i in range(self.num_players)]
        )

        state_low = np.tile([0.0, 0.0, -velocity_limit, -velocity_limit], num_agents)
        state_high = np.tile([1.0, 1.0, velocity_limit, velocity_limit], num_agents)
        action_low = -action_limit * np.ones(self.total_action_dim)
        action_high = action_limit * np.ones(self.total_action_dim)

        self.observation_space = spaces.Box(low=state_low, high=state_high, dtype=np.float64)
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float64)

        self.L_list = np.array([1.0, 1.0])

        # ---- Precompute Fourier basis and SMC coefficients ----
        self._initialize_fourier_basis()
        self.phik_list = self._compute_target_coefficients()

        # The SMC-inspired weight: w_k = λ_k * φ_k / h_k
        # Used in: ∂Cost/∂x = Σ_k w_k * kπ * sin(kπx) * cos(kπy)  (etc.)
        self.smc_weights = self.lamk_list * self.phik_list / self.hk_list

        # Store as torch tensors for use in costs_jacobian_and_hessian
        self.ks_torch = torch.tensor(self.ks, dtype=torch.float64)
        self.smc_weights_torch = torch.tensor(self.smc_weights, dtype=torch.float64)

        # Visit grid for exploration bonus
        self.grid_res = 20
        self.visit_grid = None

        # Running state
        self.state = None
        self.ck_list_update = None
        self.current_time = None
        self.step_count = None

        self.is_nonlinear_game = True

    def _target_pdf_single(self, pos):
        val = 0.0
        for peak in self.PEAKS:
            val += peak["weight"] * mvn.pdf(pos, peak["mean"], peak["cov"])
        return val

    def _target_pdf(self, positions):
        result = np.zeros(positions.shape[0])
        for peak in self.PEAKS:
            result += peak["weight"] * mvn.pdf(positions, peak["mean"], peak["cov"])
        return result

    def _initialize_fourier_basis(self):
        ks_dim1, ks_dim2 = np.meshgrid(
            np.arange(self.num_k_per_dim), np.arange(self.num_k_per_dim)
        )
        self.ks = np.array([ks_dim1.ravel(), ks_dim2.ravel()]).T

        grids_x, grids_y = np.meshgrid(
            np.linspace(0, self.L_list[0], 100),
            np.linspace(0, self.L_list[1], 100),
        )
        grids = np.array([grids_x.ravel(), grids_y.ravel()]).T
        dx, dy = self.L_list[0] / 99, self.L_list[1] / 99

        self.hk_list = np.zeros(self.ks.shape[0])
        for i, k_vec in enumerate(self.ks):
            fk = np.prod(np.cos(np.pi * k_vec * grids), axis=1)
            self.hk_list[i] = np.sqrt(np.sum(fk ** 2) * dx * dy)

        self.lamk_list = np.power(1.0 + np.linalg.norm(self.ks, axis=1), -1.5)

    def _compute_target_coefficients(self):
        grids_x, grids_y = np.meshgrid(
            np.linspace(0, self.L_list[0], 100),
            np.linspace(0, self.L_list[1], 100),
        )
        grids = np.array([grids_x.ravel(), grids_y.ravel()]).T
        dx, dy = self.L_list[0] / 99, self.L_list[1] / 99
        pdf_vals = self._target_pdf(grids)

        phik = np.zeros(self.ks.shape[0])
        for i, k_vec in enumerate(self.ks):
            fk = np.prod(np.cos(np.pi * k_vec * grids), axis=1) / self.hk_list[i]
            phik[i] = np.sum(fk * pdf_vals) * dx * dy
        return phik

    def _evaluate_fourier_basis(self, positions):
        fk = np.zeros(self.ks.shape[0])
        for pos in positions:
            fk += np.prod(np.cos(np.pi * self.ks * pos), axis=1) / self.hk_list
        return fk / self.num_agents

    def _compute_ergodic_metric(self):
        if self.current_time > 0:
            ck = self.ck_list_update / self.current_time
        else:
            ck = np.zeros_like(self.ck_list_update)
        return np.sum(self.lamk_list * np.square(ck - self.phik_list))

    def step(self, action):
        action = np.clip(action, -self.action_limit, self.action_limit)
        accel = action.reshape(self.num_agents, 2)

        agents = self.state.reshape(self.num_agents, 4)
        pos = agents[:, :2].copy()
        vel = agents[:, 2:].copy()

        # Double integrator with damping
        new_vel = (1.0 - self.damping * self.dt) * vel + self.dt * accel
        new_vel = np.clip(new_vel, -self.velocity_limit, self.velocity_limit)
        new_pos = pos + self.dt * new_vel

        # Boundary: clip + reflect velocity
        for i in range(self.num_agents):
            for d in range(2):
                if new_pos[i, d] < 0.0:
                    new_pos[i, d] = 0.0
                    new_vel[i, d] = abs(new_vel[i, d]) * 0.5
                elif new_pos[i, d] > 1.0:
                    new_pos[i, d] = 1.0
                    new_vel[i, d] = -abs(new_vel[i, d]) * 0.5

        # === Reward ===
        reward = 0.0

        # 1. PDF value at each agent position (primary driver)
        for i in range(self.num_agents):
            reward += self.w_pdf * self._target_pdf_single(new_pos[i])

        # 2. Control penalty
        reward -= self.w_ctrl * np.sum(accel ** 2)

        # 3. Exploration bonus: reward visiting new grid cells
        for i in range(self.num_agents):
            gx = min(int(new_pos[i, 0] * self.grid_res), self.grid_res - 1)
            gy = min(int(new_pos[i, 1] * self.grid_res), self.grid_res - 1)
            visit_count = self.visit_grid[gx, gy]
            if visit_count < 10:
                # Weight by PDF at that cell: high-value cells give more exploration bonus
                cell_center = np.array([(gx + 0.5) / self.grid_res, (gy + 0.5) / self.grid_res])
                cell_pdf = self._target_pdf_single(cell_center)
                reward += 0.5 * (1.0 - visit_count / 10.0) * (1.0 + cell_pdf * 0.1)
            self.visit_grid[gx, gy] += 1

        # Update Fourier coefficients for metric logging
        fk = self._evaluate_fourier_basis(new_pos)
        self.ck_list_update += fk * self.dt
        self.current_time += self.dt

        new_agents = np.column_stack([new_pos, new_vel])
        self.state = new_agents.ravel()
        self.step_count += 1

        # Per-player costs for MA-GPS info
        costs = np.zeros(self.num_players)
        for i in range(self.num_players):
            pdf_val = max(self._target_pdf_single(new_pos[i]), 1e-10)
            costs[i] = -np.log(pdf_val) + 0.1 * np.sum(accel[i] ** 2)

        terminated = False
        truncated = self.step_count >= self.max_episode_steps

        erg_metric = self._compute_ergodic_metric()
        coverage = np.sum(self.visit_grid > 0) / (self.grid_res ** 2)

        info = {
            "ergodic_metric": float(erg_metric),
            "coverage_fraction": float(coverage),
            "time": float(self.current_time),
            "step": self.step_count,
            "individual_cost": costs,
        }

        return self.state, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if options is not None and "initial_state" in options:
            self.state = options["initial_state"]
        else:
            agents = np.zeros((self.num_agents, 4))
            agents[:, :2] = self.np_random.uniform(
                low=self.init_pos_low, high=self.init_pos_high, size=(self.num_agents, 2)
            )
            agents[:, 2:] = self.np_random.uniform(low=-0.05, high=0.05, size=(self.num_agents, 2))
            self.state = agents.ravel()

        self.ck_list_update = np.zeros(self.ks.shape[0])
        self.current_time = 0.0
        self.step_count = 0
        self.visit_grid = np.zeros((self.grid_res, self.grid_res))
        return self.state, {}

    def render(self):
        pass

    # =========================================================================
    # MA-GPS interface: costs_jacobian_and_hessian (NOT JIT — uses precomputed data)
    # =========================================================================

    def costs_jacobian_and_hessian(self, z):
        """
        SMC-inspired LQ guidance cost.

        Cost = -P(x) + α||u||² + β*inter_agent_repulsion
        where P(x) = Σ_k (λ_k φ_k / h_k) cos(k1πx1) cos(k2πx2)
        is the λ-weighted Fourier reconstruction of the target PDF.

        The gradient of -P(x) naturally drives agents toward ALL peaks of the
        target distribution with broad spatial reach (unlike Gaussians which
        decay exponentially). This matches the SMC control direction at t=0.

        z: (batch_size, state_dim + action_dim)
        Returns: jacobians (num_players, batch, dim), hessians (num_players, batch, dim, dim)
        """
        batch_size, input_dim = z.shape
        num_players = self.num_players
        state_dim = self.total_state_dim
        alpha = 0.2  # control cost
        beta = 0.5   # inter-agent repulsion (low — just enough to prevent overlap)
        r = 20.0     # repulsion length scale

        ks = self.ks_torch.to(z.device)       # (num_k, 2)
        wk = self.smc_weights_torch.to(z.device)  # (num_k,)
        pi = np.pi

        jacobians = torch.zeros(num_players, batch_size, input_dim, device=z.device, dtype=z.dtype)
        hessians = torch.zeros(num_players, batch_size, input_dim, input_dim, device=z.device, dtype=z.dtype)

        for i in range(num_players):
            ix = i * 4       # x position index
            iy = i * 4 + 1   # y position index
            iax = state_dim + i * 2   # ax control index
            iay = state_dim + i * 2 + 1  # ay control index

            px = z[:, ix]  # (batch,)
            py = z[:, iy]  # (batch,)

            # ---- SMC Fourier gradient: ∂(-P)/∂x ----
            # ∂(-P)/∂x1 = Σ_k w_k * k1*π * sin(k1*π*x1) * cos(k2*π*x2)
            # ∂(-P)/∂x2 = Σ_k w_k * k2*π * cos(k1*π*x1) * sin(k2*π*x2)

            k1 = ks[:, 0]  # (num_k,)
            k2 = ks[:, 1]  # (num_k,)

            # (batch, num_k) — broadcast: px is (batch,), k1 is (num_k,)
            sin_k1x = torch.sin(pi * k1.unsqueeze(0) * px.unsqueeze(1))
            cos_k1x = torch.cos(pi * k1.unsqueeze(0) * px.unsqueeze(1))
            sin_k2y = torch.sin(pi * k2.unsqueeze(0) * py.unsqueeze(1))
            cos_k2y = torch.cos(pi * k2.unsqueeze(0) * py.unsqueeze(1))

            # Jacobian of -P(x) w.r.t. position
            # Scale factor for gradient strength
            grad_scale = 50.0

            # ∂(-P)/∂x = Σ_k w_k * k1*π * sin(k1πx) * cos(k2πy)
            jac_x = grad_scale * torch.sum(wk * k1 * pi * sin_k1x * cos_k2y, dim=1)
            jac_y = grad_scale * torch.sum(wk * k2 * pi * cos_k1x * sin_k2y, dim=1)

            jacobians[i, :, ix] = jac_x
            jacobians[i, :, iy] = jac_y

            # Hessian of -P(x) w.r.t. position
            hess_xx = grad_scale * torch.sum(wk * (k1 * pi) ** 2 * cos_k1x * cos_k2y, dim=1)
            hess_yy = grad_scale * torch.sum(wk * (k2 * pi) ** 2 * cos_k1x * cos_k2y, dim=1)
            hess_xy = grad_scale * torch.sum(-wk * k1 * k2 * pi**2 * sin_k1x * sin_k2y, dim=1)

            hessians[i, :, ix, ix] = hess_xx
            hessians[i, :, iy, iy] = hess_yy
            hessians[i, :, ix, iy] = hess_xy
            hessians[i, :, iy, ix] = hess_xy

            # ---- Inter-agent repulsion (weak) ----
            for j in range(num_players):
                if j != i:
                    jx = j * 4
                    jy = j * 4 + 1
                    dx = px - z[:, jx]
                    dy = py - z[:, jy]
                    dist_sq = dx * dx + dy * dy
                    repul = beta * torch.exp(-r * dist_sq)

                    jacobians[i, :, ix] += -2.0 * r * dx * repul
                    jacobians[i, :, iy] += -2.0 * r * dy * repul

                    hessians[i, :, ix, ix] += -2.0 * r * repul * (1.0 - 2.0 * r * dx * dx)
                    hessians[i, :, iy, iy] += -2.0 * r * repul * (1.0 - 2.0 * r * dy * dy)
                    hessians[i, :, ix, iy] += 4.0 * r * r * dx * dy * repul
                    hessians[i, :, iy, ix] += 4.0 * r * r * dx * dy * repul

            # ---- Control penalty ----
            jacobians[i, :, iax] = 2.0 * alpha * z[:, iax]
            jacobians[i, :, iay] = 2.0 * alpha * z[:, iay]
            hessians[i, :, iax, iax] = 2.0 * alpha
            hessians[i, :, iay, iay] = 2.0 * alpha

        return jacobians, hessians

    # =========================================================================
    # MA-GPS interface: dynamics (JIT — no precomputed data needed)
    # =========================================================================

    @torch.jit.script
    def dynamics_jacobian(states, controls):
        """Double integrator with damping."""
        batch_size = states.shape[0]
        n = states.shape[1]
        m = controls.shape[1]
        num_players = n // 4
        dt = 0.05
        damp = 0.5

        jacobian = torch.zeros(batch_size, n, n + m, device=states.device)
        for i in range(num_players):
            sx, sy, svx, svy = i*4, i*4+1, i*4+2, i*4+3
            cax, cay = n + i*2, n + i*2 + 1

            jacobian[:, sx, sx] = 1.0
            jacobian[:, sx, svx] = dt
            jacobian[:, sy, sy] = 1.0
            jacobian[:, sy, svy] = dt
            jacobian[:, svx, svx] = 1.0 - damp * dt
            jacobian[:, svx, cax] = dt
            jacobian[:, svy, svy] = 1.0 - damp * dt
            jacobian[:, svy, cay] = dt
        return jacobian

    @torch.jit.script
    def dynamics(states, controls):
        """Double integrator with damping."""
        batch_size = states.shape[0]
        n = states.shape[1]
        num_players = n // 4
        dt = 0.05
        damp = 0.5

        next_states = torch.zeros_like(states)
        for i in range(num_players):
            x = states[:, i*4]
            y = states[:, i*4+1]
            vx = states[:, i*4+2]
            vy = states[:, i*4+3]
            ax = controls[:, i*2]
            ay = controls[:, i*2+1]

            new_vx = (1.0 - damp * dt) * vx + dt * ax
            new_vy = (1.0 - damp * dt) * vy + dt * ay
            next_states[:, i*4] = x + dt * new_vx
            next_states[:, i*4+1] = y + dt * new_vy
            next_states[:, i*4+2] = new_vx
            next_states[:, i*4+3] = new_vy
        return next_states
