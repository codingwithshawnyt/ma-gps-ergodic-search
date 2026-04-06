"""
Multi-Agent Ergodic Search Environment (v2)

Redesigned based on:
- Kernel ergodic metric decomposition (Sun et al. 2024): E = -(2/T)∫p(s(t))dt + (1/T²)∫∫g(s,s')dtdτ + const
  → Reward decomposes into: maximize time at high-PDF locations + minimize trajectory self-overlap
- SMC dynamics (Mathew & Mezic 2011): second-order dynamics, T~50s for multi-agent coverage
- Miller & Murphey 2013: ergodic trajectory optimization descent = LQR solution (maps to MA-GPS)
- Iannelli et al.: multi-agent objective = ergodic metric + control energy + inter-agent repulsion

Key changes from v1:
1. Double-integrator dynamics (state includes velocity) — gives LQ guidance real structure
2. 1000-step episodes (dt=0.05, 50s total) — enough time for ergodic coverage
3. Reward based on kernel metric decomposition: PDF value + exploration + control penalty
4. Multi-modal LQ guidance cost using Gaussian mixture gradients (not single-point attraction)
5. Inter-agent repulsion in both reward and LQ cost
6. Position clipping + velocity reflection (no termination at boundaries)
7. Ergodic metric tracked for evaluation only, not used as reward
"""

from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
from scipy.stats import multivariate_normal as mvn


class ErgodicSearchEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    # Gaussian mixture parameters (class-level constants)
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
        reward_repulsion_weight: float = 1.0,
        reward_velocity_weight: float = 0.5,
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

        # Reward weights
        self.w_pdf = reward_pdf_weight
        self.w_ctrl = reward_control_weight
        self.w_repel = reward_repulsion_weight
        self.w_vel = reward_velocity_weight

        # Per-agent state: [x, y, vx, vy] — double integrator
        self.nx = 4
        self.nu = 2  # acceleration [ax, ay]
        self.total_state_dim = self.nx * self.num_players
        self.total_action_dim = self.nu * self.num_players

        # MA-GPS index list
        self.players_u_index_list = torch.tensor(
            [[i * 2, i * 2 + 1] for i in range(self.num_players)]
        )

        # State bounds: positions in [0,1], velocities in [-vel_limit, vel_limit]
        state_low = np.tile([0.0, 0.0, -velocity_limit, -velocity_limit], num_agents)
        state_high = np.tile([1.0, 1.0, velocity_limit, velocity_limit], num_agents)

        action_low = -action_limit * np.ones(self.total_action_dim)
        action_high = action_limit * np.ones(self.total_action_dim)

        self.observation_space = spaces.Box(low=state_low, high=state_high, dtype=np.float64)
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float64)

        # Search space
        self.L_list = np.array([1.0, 1.0])

        # Fourier basis setup
        self._initialize_fourier_basis()
        self.phik_list = self._compute_target_coefficients()

        # Pre-compute inverse covariances for Gaussian mixture (used in cost Jacobian)
        self.peak_inv_covs = []
        for peak in self.PEAKS:
            self.peak_inv_covs.append(np.linalg.inv(peak["cov"]))

        # Visit grid for exploration bonus (20x20)
        self.grid_res = 20
        self.visit_grid = None

        # Running state
        self.state = None
        self.ck_list_update = None
        self.current_time = None
        self.step_count = None

        self.is_nonlinear_game = True

    def _target_pdf(self, positions):
        """Evaluate Gaussian mixture at positions (n_points, 2)."""
        result = np.zeros(positions.shape[0])
        for peak in self.PEAKS:
            result += peak["weight"] * mvn.pdf(positions, peak["mean"], peak["cov"])
        return result

    def _target_pdf_single(self, pos):
        """Evaluate Gaussian mixture at single position (2,)."""
        val = 0.0
        for peak in self.PEAKS:
            val += peak["weight"] * mvn.pdf(pos, peak["mean"], peak["cov"])
        return val

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
        dx = self.L_list[0] / 99
        dy = self.L_list[1] / 99

        self.hk_list = np.zeros(self.ks.shape[0])
        for i, k_vec in enumerate(self.ks):
            fk_vals = np.prod(np.cos(np.pi * k_vec * grids), axis=1)
            self.hk_list[i] = np.sqrt(np.sum(np.square(fk_vals)) * dx * dy)

        self.lamk_list = np.power(1.0 + np.linalg.norm(self.ks, axis=1), -3.0 / 2.0)

    def _compute_target_coefficients(self):
        grids_x, grids_y = np.meshgrid(
            np.linspace(0, self.L_list[0], 100),
            np.linspace(0, self.L_list[1], 100),
        )
        grids = np.array([grids_x.ravel(), grids_y.ravel()]).T
        dx = self.L_list[0] / 99
        dy = self.L_list[1] / 99
        pdf_vals = self._target_pdf(grids)

        phik_list = np.zeros(self.ks.shape[0])
        for i, k_vec in enumerate(self.ks):
            fk_vals = np.prod(np.cos(np.pi * k_vec * grids), axis=1) / self.hk_list[i]
            phik_list[i] = np.sum(fk_vals * pdf_vals) * dx * dy
        return phik_list

    def _evaluate_fourier_basis(self, positions):
        """Evaluate Fourier basis at agent positions (num_agents, 2), return averaged."""
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

        # Extract current state
        agents = self.state.reshape(self.num_agents, 4)
        pos = agents[:, :2].copy()
        vel = agents[:, 2:].copy()

        # Double integrator with damping
        new_vel = (1.0 - self.damping * self.dt) * vel + self.dt * accel
        new_vel = np.clip(new_vel, -self.velocity_limit, self.velocity_limit)
        new_pos = pos + self.dt * new_vel

        # Boundary: clip position and reflect velocity
        for i in range(self.num_agents):
            for d in range(2):
                if new_pos[i, d] < 0.0:
                    new_pos[i, d] = 0.0
                    new_vel[i, d] = abs(new_vel[i, d]) * 0.5
                elif new_pos[i, d] > 1.0:
                    new_pos[i, d] = 1.0
                    new_vel[i, d] = -abs(new_vel[i, d]) * 0.5

        # === Reward computation (kernel ergodic metric decomposition) ===
        reward = 0.0

        # 1. PDF value at each agent's position (from -(2/T)∫p(s(t))dt term)
        for i in range(self.num_agents):
            reward += self.w_pdf * self._target_pdf_single(new_pos[i])

        # 2. Control penalty
        reward -= self.w_ctrl * np.sum(accel ** 2)

        # 3. Inter-agent repulsion (encourage spreading — multiple agents shouldn't cluster)
        if self.num_agents > 1:
            for i in range(self.num_agents):
                for j in range(i + 1, self.num_agents):
                    dist = np.linalg.norm(new_pos[i] - new_pos[j])
                    reward += self.w_repel * min(dist, 0.5)  # cap at 0.5 to avoid dominating

        # 4. Velocity bonus (encourage movement — stationary agents can't be ergodic)
        for i in range(self.num_agents):
            speed = np.linalg.norm(new_vel[i])
            reward += self.w_vel * min(speed, 0.3)

        # 5. Exploration bonus: reward visiting new grid cells
        for i in range(self.num_agents):
            gx = min(int(new_pos[i, 0] * self.grid_res), self.grid_res - 1)
            gy = min(int(new_pos[i, 1] * self.grid_res), self.grid_res - 1)
            visit_count = self.visit_grid[gx, gy]
            if visit_count < 10:
                reward += 0.5 * (1.0 - visit_count / 10.0)  # diminishing returns
            self.visit_grid[gx, gy] += 1

        # Update Fourier coefficients (for ergodic metric logging)
        fk = self._evaluate_fourier_basis(new_pos)
        self.ck_list_update += fk * self.dt
        self.current_time += self.dt

        # Assemble new state
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
            # Small random initial velocities
            agents[:, 2:] = self.np_random.uniform(
                low=-0.05, high=0.05, size=(self.num_agents, 2)
            )
            self.state = agents.ravel()

        self.ck_list_update = np.zeros(self.ks.shape[0])
        self.current_time = 0.0
        self.step_count = 0
        self.visit_grid = np.zeros((self.grid_res, self.grid_res))

        return self.state, {}

    def render(self):
        pass

    # =========================================================================
    # MA-GPS JIT functions
    # =========================================================================

    @torch.jit.script
    def costs_jacobian_and_hessian(z):
        """
        LQ guidance cost: negative Gaussian mixture + control penalty + inter-agent repulsion.

        Cost_i = -Σ_j w_j * exp(-s_j * ||pos_i - μ_j||²) + α*||u_i||² + β*Σ_{k≠i} exp(-r*||pos_i - pos_k||²)

        The negative Gaussian mixture naturally creates multi-modal attraction — agents get pulled
        toward ALL peaks of the target distribution, not just the weighted center.

        z: (batch, 4*num_players + 2*num_players) = [x,y,vx,vy,..., ax,ay,...]
        """
        batch_size, input_dim = z.shape

        # Infer dimensions: input_dim = nx*N + nu*N = 4*N + 2*N = 6*N
        num_players = input_dim // 6
        state_dim = 4 * num_players
        alpha = 0.2   # control cost weight
        beta = 5.0    # inter-agent repulsion weight
        r = 50.0      # repulsion length scale (1/σ²)
        vel_cost = 0.0  # no explicit velocity cost in LQ guidance

        # Gaussian mixture peaks (unnormalized for numerical stability in exp)
        # Using widths (1/2σ²) derived from the actual covariance diagonal
        # Peak 1: cov diag ~0.01 → s = 1/(2*0.01) = 50
        # Peak 2: cov diag ~0.005 → s = 1/(2*0.005) = 100
        # Peak 3: cov diag ~0.006 → s = 1/(2*0.006) ≈ 83
        peak_mx = torch.tensor([0.35, 0.68, 0.56], device=z.device)
        peak_my = torch.tensor([0.38, 0.25, 0.64], device=z.device)
        peak_w = torch.tensor([0.5, 0.2, 0.3], device=z.device)
        peak_sx = torch.tensor([50.0, 100.0, 62.5], device=z.device)
        peak_sy = torch.tensor([50.0, 100.0, 125.0], device=z.device)

        jacobians = torch.zeros(num_players, batch_size, input_dim, device=z.device)
        hessians = torch.zeros(num_players, batch_size, input_dim, input_dim, device=z.device)

        for i in range(num_players):
            # State indices: x, y, vx, vy
            ix = i * 4
            iy = i * 4 + 1
            ivx = i * 4 + 2
            ivy = i * 4 + 3
            # Control indices
            iax = state_dim + i * 2
            iay = state_dim + i * 2 + 1

            px = z[:, ix]
            py = z[:, iy]
            ax_val = z[:, iax]
            ay_val = z[:, iay]

            # --- Gaussian mixture attraction (negative mixture → gradient pushes toward peaks) ---
            for j in range(3):
                dx = px - peak_mx[j]
                dy = py - peak_my[j]
                exponent = -(peak_sx[j] * dx * dx + peak_sy[j] * dy * dy)
                gauss = peak_w[j] * torch.exp(exponent)

                # Jacobian of -gauss w.r.t. position
                jacobians[i, :, ix] += 2.0 * peak_sx[j] * dx * gauss
                jacobians[i, :, iy] += 2.0 * peak_sy[j] * dy * gauss

                # Hessian of -gauss w.r.t. position
                hessians[i, :, ix, ix] += 2.0 * peak_sx[j] * gauss * (1.0 - 2.0 * peak_sx[j] * dx * dx)
                hessians[i, :, iy, iy] += 2.0 * peak_sy[j] * gauss * (1.0 - 2.0 * peak_sy[j] * dy * dy)
                hessians[i, :, ix, iy] += -4.0 * peak_sx[j] * peak_sy[j] * dx * dy * gauss
                hessians[i, :, iy, ix] += -4.0 * peak_sx[j] * peak_sy[j] * dx * dy * gauss

            # --- Inter-agent repulsion ---
            for k in range(num_players):
                if k != i:
                    kx = k * 4
                    ky = k * 4 + 1
                    dx_ik = px - z[:, kx]
                    dy_ik = py - z[:, ky]
                    dist_sq = dx_ik * dx_ik + dy_ik * dy_ik
                    repul = beta * torch.exp(-r * dist_sq)

                    # Jacobian: ∂(β*exp(-r*d²))/∂x_i = -2*r*dx*repul
                    jacobians[i, :, ix] += -2.0 * r * dx_ik * repul
                    jacobians[i, :, iy] += -2.0 * r * dy_ik * repul

                    # Hessian
                    hessians[i, :, ix, ix] += -2.0 * r * repul * (1.0 - 2.0 * r * dx_ik * dx_ik)
                    hessians[i, :, iy, iy] += -2.0 * r * repul * (1.0 - 2.0 * r * dy_ik * dy_ik)
                    hessians[i, :, ix, iy] += 4.0 * r * r * dx_ik * dy_ik * repul
                    hessians[i, :, iy, ix] += 4.0 * r * r * dx_ik * dy_ik * repul

            # --- Control penalty: α*(ax² + ay²) ---
            jacobians[i, :, iax] = 2.0 * alpha * ax_val
            jacobians[i, :, iay] = 2.0 * alpha * ay_val
            hessians[i, :, iax, iax] = 2.0 * alpha
            hessians[i, :, iay, iay] = 2.0 * alpha

        return jacobians, hessians

    @torch.jit.script
    def dynamics_jacobian(states, controls):
        """
        Double integrator with damping:
        x' = x + dt*vx
        y' = y + dt*vy
        vx' = (1-d*dt)*vx + dt*ax
        vy' = (1-d*dt)*vy + dt*ay

        State per agent: [x, y, vx, vy]
        Control per agent: [ax, ay]
        """
        batch_size = states.shape[0]
        n = states.shape[1]   # 4 * num_players
        m = controls.shape[1]  # 2 * num_players
        num_players = n // 4
        dt = 0.05
        damp = 0.5

        jacobian = torch.zeros(batch_size, n, n + m, device=states.device)

        for i in range(num_players):
            sx = i * 4      # x index in state
            sy = i * 4 + 1  # y
            svx = i * 4 + 2 # vx
            svy = i * 4 + 3 # vy
            cax = n + i * 2     # ax index in [state, control]
            cay = n + i * 2 + 1 # ay

            # ∂x'/∂x = 1
            jacobian[:, sx, sx] = 1.0
            # ∂x'/∂vx = dt
            jacobian[:, sx, svx] = dt
            # ∂y'/∂y = 1
            jacobian[:, sy, sy] = 1.0
            # ∂y'/∂vy = dt
            jacobian[:, sy, svy] = dt
            # ∂vx'/∂vx = 1 - d*dt
            jacobian[:, svx, svx] = 1.0 - damp * dt
            # ∂vx'/∂ax = dt
            jacobian[:, svx, cax] = dt
            # ∂vy'/∂vy = 1 - d*dt
            jacobian[:, svy, svy] = 1.0 - damp * dt
            # ∂vy'/∂ay = dt
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
            x = states[:, i * 4]
            y = states[:, i * 4 + 1]
            vx = states[:, i * 4 + 2]
            vy = states[:, i * 4 + 3]
            ax = controls[:, i * 2]
            ay = controls[:, i * 2 + 1]

            new_vx = (1.0 - damp * dt) * vx + dt * ax
            new_vy = (1.0 - damp * dt) * vy + dt * ay
            next_states[:, i * 4] = x + dt * new_vx
            next_states[:, i * 4 + 1] = y + dt * new_vy
            next_states[:, i * 4 + 2] = new_vx
            next_states[:, i * 4 + 3] = new_vy

        return next_states
