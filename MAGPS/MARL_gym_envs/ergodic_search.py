"""
Multi-Agent Ergodic Search Environment (v3 — final)

All fixes applied:
1. Fixed diagonal Hessian (pos=15, vel=0.5, ctrl=0.4) — no singular matrix possible
2. Corrected dynamics_jacobian: ∂x/∂vx = dt*(1-damp*dt), ∂x/∂ax = dt²
3. Jacobian-only inter-agent repulsion (safe since Hessian is fixed)
4. SMC Fourier gradient in Jacobian (k_max=5, grad_scale=1.0)
5. All outputs clamped, NaN fallback
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
        {
            "mean": np.array([0.35, 0.38]),
            "cov": np.array([[0.01, 0.004], [0.004, 0.01]]),
            "weight": 0.5,
        },
        {
            "mean": np.array([0.68, 0.25]),
            "cov": np.array([[0.005, -0.003], [-0.003, 0.005]]),
            "weight": 0.2,
        },
        {
            "mean": np.array([0.56, 0.64]),
            "cov": np.array([[0.008, 0.0], [0.0, 0.004]]),
            "weight": 0.3,
        },
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

        self.observation_space = spaces.Box(
            low=state_low, high=state_high, dtype=np.float64
        )
        self.action_space = spaces.Box(
            low=action_low, high=action_high, dtype=np.float64
        )

        self.L_list = np.array([1.0, 1.0])

        self._initialize_fourier_basis()
        self.phik_list = self._compute_target_coefficients()

        self.lq_k_max = 5
        self._precompute_lq_weights()

        self.grid_res = 20
        self.visit_grid = None
        self.state = None
        self.ck_list_update = None
        self.current_time = None
        self.step_count = None
        self.is_nonlinear_game = True

    def _precompute_lq_weights(self):
        ks_d1, ks_d2 = np.meshgrid(np.arange(self.lq_k_max), np.arange(self.lq_k_max))
        self.lq_ks = np.array([ks_d1.ravel(), ks_d2.ravel()]).T

        grids_x, grids_y = np.meshgrid(
            np.linspace(0, self.L_list[0], 100), np.linspace(0, self.L_list[1], 100)
        )
        grids = np.array([grids_x.ravel(), grids_y.ravel()]).T
        dx, dy = self.L_list[0] / 99, self.L_list[1] / 99
        pdf_vals = self._target_pdf(grids)

        lq_hk = np.zeros(len(self.lq_ks))
        lq_phik = np.zeros(len(self.lq_ks))
        lq_lamk = np.power(1.0 + np.linalg.norm(self.lq_ks, axis=1), -1.5)

        for i, k_vec in enumerate(self.lq_ks):
            fk = np.prod(np.cos(np.pi * k_vec * grids), axis=1)
            lq_hk[i] = max(np.sqrt(np.sum(fk**2) * dx * dy), 1e-10)
            lq_phik[i] = np.sum((fk / lq_hk[i]) * pdf_vals) * dx * dy

        self.lq_wk = lq_lamk * lq_phik / lq_hk
        self.lq_ks_torch = torch.tensor(self.lq_ks, dtype=torch.float32)
        self.lq_wk_torch = torch.tensor(self.lq_wk, dtype=torch.float32)

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
            np.linspace(0, self.L_list[0], 100), np.linspace(0, self.L_list[1], 100)
        )
        grids = np.array([grids_x.ravel(), grids_y.ravel()]).T
        dx, dy = self.L_list[0] / 99, self.L_list[1] / 99

        self.hk_list = np.zeros(self.ks.shape[0])
        for i, k_vec in enumerate(self.ks):
            fk = np.prod(np.cos(np.pi * k_vec * grids), axis=1)
            self.hk_list[i] = max(np.sqrt(np.sum(fk**2) * dx * dy), 1e-10)
        self.lamk_list = np.power(1.0 + np.linalg.norm(self.ks, axis=1), -1.5)

    def _compute_target_coefficients(self):
        grids_x, grids_y = np.meshgrid(
            np.linspace(0, self.L_list[0], 100), np.linspace(0, self.L_list[1], 100)
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

        new_vel = (1.0 - self.damping * self.dt) * vel + self.dt * accel
        new_vel = np.clip(new_vel, -self.velocity_limit, self.velocity_limit)
        new_pos = pos + self.dt * new_vel

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

        # 1. MOVEMENT reward: reward speed (agents MUST keep moving)
        for i in range(self.num_agents):
            speed = np.linalg.norm(new_vel[i])
            reward += 20.0 * speed  # dominant signal: keep moving

        # 2. Exploration: reward visiting NEW high-PDF cells (diminishing returns)
        for i in range(self.num_agents):
            gx = min(int(new_pos[i, 0] * self.grid_res), self.grid_res - 1)
            gy = min(int(new_pos[i, 1] * self.grid_res), self.grid_res - 1)
            visit_count = self.visit_grid[gx, gy]
            if visit_count < 5:
                cell_center = np.array(
                    [(gx + 0.5) / self.grid_res, (gy + 0.5) / self.grid_res]
                )
                cell_pdf = self._target_pdf_single(cell_center)
                reward += 10.0 * (1.0 + cell_pdf) * (1.0 - visit_count / 5.0)
            self.visit_grid[gx, gy] += 1

        # 3. Light control penalty
        reward -= 0.05 * np.sum(accel ** 2)

        # 4. Stillness penalty: if agent barely moved, penalize
        for i in range(self.num_agents):
            displacement = np.linalg.norm(new_pos[i] - pos[i])
            if displacement < 0.001:
                reward -= 5.0

        fk = self._evaluate_fourier_basis(new_pos)
        self.ck_list_update += fk * self.dt
        self.current_time += self.dt

        new_agents = np.column_stack([new_pos, new_vel])
        self.state = new_agents.ravel()
        self.step_count += 1

        costs = np.zeros(self.num_players)
        for i in range(self.num_players):
            pdf_val = max(self._target_pdf_single(new_pos[i]), 1e-10)
            costs[i] = -np.log(pdf_val) + 0.1 * np.sum(accel[i] ** 2)

        terminated = False
        truncated = self.step_count >= self.max_episode_steps

        info = {
            "ergodic_metric": float(self._compute_ergodic_metric()),
            "coverage_fraction": float(
                np.sum(self.visit_grid > 0) / (self.grid_res**2)
            ),
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
                low=self.init_pos_low,
                high=self.init_pos_high,
                size=(self.num_agents, 2),
            )
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
    # MA-GPS: costs_jacobian_and_hessian
    # =========================================================================

    def costs_jacobian_and_hessian(self, z):
        batch_size, input_dim = z.shape
        num_players = self.num_players
        state_dim = self.total_state_dim
        alpha = 0.2
        grad_scale = 1.0
        pos_hess = 5.0  # position curvature (max |δx| = 3.25/5 = 0.65)
        vel_hess = 0.5  # velocity curvature (prevents ill-conditioning)

        ks = self.lq_ks_torch.to(device=z.device, dtype=z.dtype)
        # Replace static wk with dynamic S_k
        if self.current_time > 0:
            ck_current = self.ck_list_update / self.current_time
        else:
            ck_current = np.zeros(len(self.lq_ks))
        # Only use the lq_k_max subset
        sk_dynamic = ck_current[:len(self.lq_ks)] - self.lq_phik
        dynamic_wk = self.lq_lamk * sk_dynamic / self.lq_hk
        # Convert to torch
        wk = torch.tensor(dynamic_wk, device=z.device, dtype=z.dtype)
        pi = 3.141592653589793

        jacobians = torch.zeros(
            num_players, batch_size, input_dim, device=z.device, dtype=z.dtype
        )
        hessians = torch.zeros(
            num_players,
            batch_size,
            input_dim,
            input_dim,
            device=z.device,
            dtype=z.dtype,
        )

        for i in range(num_players):
            ix = i * 4
            iy = i * 4 + 1
            ivx = i * 4 + 2
            ivy = i * 4 + 3
            iax = state_dim + i * 2
            iay = state_dim + i * 2 + 1

            px = z[:, ix]
            py = z[:, iy]

            k1 = ks[:, 0]
            k2 = ks[:, 1]

            sin_k1x = torch.sin(pi * k1.unsqueeze(0) * px.unsqueeze(1))
            cos_k1x = torch.cos(pi * k1.unsqueeze(0) * px.unsqueeze(1))
            sin_k2y = torch.sin(pi * k2.unsqueeze(0) * py.unsqueeze(1))
            cos_k2y = torch.cos(pi * k2.unsqueeze(0) * py.unsqueeze(1))

            # SMC Fourier gradient
            jac_x = grad_scale * torch.sum(wk * k1 * pi * sin_k1x * cos_k2y, dim=1)
            jac_y = grad_scale * torch.sum(wk * k2 * pi * cos_k1x * sin_k2y, dim=1)

            # Boundary repulsion: quadratic wall potential
            margin = 0.02  # minimum distance from boundary
            w_wall = 0.5  # wall strength
            px_safe = torch.clamp(px, margin, 1.0 - margin)
            py_safe = torch.clamp(py, margin, 1.0 - margin)
            wall_jac_x = (
                -w_wall / (px_safe - 0.0 + margin) ** 2
                + w_wall / (1.0 - px_safe + margin) ** 2
            )
            wall_jac_y = (
                -w_wall / (py_safe - 0.0 + margin) ** 2
                + w_wall / (1.0 - py_safe + margin) ** 2
            )

            jacobians[i, :, ix] = torch.clamp(jac_x + wall_jac_x, -10.0, 10.0)
            jacobians[i, :, iy] = torch.clamp(jac_y + wall_jac_y, -10.0, 10.0)

            # Inter-agent repulsion (Jacobian only — safe since Hessian is fixed)
            for j in range(num_players):
                if j != i:
                    dx = px - z[:, j * 4]
                    dy = py - z[:, j * 4 + 1]
                    dist_sq = dx * dx + dy * dy
                    repul = 0.3 * torch.exp(-5.0 * dist_sq)
                    jacobians[i, :, ix] += torch.clamp(-10.0 * dx * repul, -5.0, 5.0)
                    jacobians[i, :, iy] += torch.clamp(-10.0 * dy * repul, -5.0, 5.0)

            # Control Jacobian
            jacobians[i, :, iax] = 2.0 * alpha * z[:, iax]
            jacobians[i, :, iay] = 2.0 * alpha * z[:, iay]

            # Velocity damping: penalize zero velocity (agents should move)
            jacobians[i, :, ivx] = 0.1 * z[:, ivx]
            jacobians[i, :, ivy] = 0.1 * z[:, ivy]

            # FIXED positive-definite diagonal Hessian
            hessians[i, :, ix, ix] = pos_hess
            hessians[i, :, iy, iy] = pos_hess
            hessians[i, :, ivx, ivx] = vel_hess
            hessians[i, :, ivy, ivy] = vel_hess
            hessians[i, :, iax, iax] = 2.0 * alpha
            hessians[i, :, iay, iay] = 2.0 * alpha

        # NaN/Inf safety
        if torch.isnan(jacobians).any() or torch.isinf(jacobians).any():
            jacobians = torch.where(
                torch.isfinite(jacobians), jacobians, torch.zeros_like(jacobians)
            )

        return jacobians, hessians

    # =========================================================================
    # MA-GPS: dynamics (corrected Jacobian)
    # =========================================================================

    @torch.jit.script
    def dynamics_jacobian(states, controls):
        """
        Corrected analytical Jacobian for double integrator with damping.

        next_x  = x + dt * new_vx = x + dt*((1-d*dt)*vx + dt*ax)
        next_vx = (1-d*dt)*vx + dt*ax

        So: ∂next_x/∂x   = 1
            ∂next_x/∂vx  = dt*(1-d*dt)     (NOT just dt)
            ∂next_x/∂ax  = dt*dt = dt²      (was MISSING before)
            ∂next_vx/∂vx = 1-d*dt
            ∂next_vx/∂ax = dt
        """
        batch_size = states.shape[0]
        n = states.shape[1]
        m = controls.shape[1]
        num_players = n // 4
        dt = 0.05
        damp = 0.5
        dt_damped = dt * (1.0 - damp * dt)  # = 0.04875
        dt_sq = dt * dt  # = 0.0025

        jacobian = torch.zeros(batch_size, n, n + m, device=states.device)
        for i in range(num_players):
            sx, sy, svx, svy = i * 4, i * 4 + 1, i * 4 + 2, i * 4 + 3
            cax, cay = n + i * 2, n + i * 2 + 1

            # Position x
            jacobian[:, sx, sx] = 1.0
            jacobian[:, sx, svx] = dt_damped
            jacobian[:, sx, cax] = dt_sq

            # Position y
            jacobian[:, sy, sy] = 1.0
            jacobian[:, sy, svy] = dt_damped
            jacobian[:, sy, cay] = dt_sq

            # Velocity x
            jacobian[:, svx, svx] = 1.0 - damp * dt
            jacobian[:, svx, cax] = dt

            # Velocity y
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
