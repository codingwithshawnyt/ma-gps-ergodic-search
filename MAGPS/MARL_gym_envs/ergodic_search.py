"""
Multi-Agent Ergodic Search Environment (v6 — coherent MAPPO baseline)

This version makes the environment consistent with how the repo actually
trains policies:

1. The scalar env reward and info["individual_cost"] now encode the same
   cooperative objective.
2. The training signal is based on ergodic metric improvement and control
   effort, not on static peak-seeking terms like -log(pdf).
3. The observation exposes the full set of coverage-error coefficients used
   by the ergodic reward, so the policy can observe the same trajectory-
   history signal that drives the objective.

The MA-GPS local Riccati teacher is still only a static directional hint.
For ergodic_search-v0, this env is intended to be validated first with the
guidance-free MAPPO baseline.
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
        reward_metric_weight: float = 500.0,
        reward_control_weight: float = 0.05,
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
        self.w_metric = reward_metric_weight
        self.w_ctrl = reward_control_weight
        self.obs_clip = 2.0
        # Expose the full Fourier coefficient set used by the reward so the
        # policy is not judged on hidden coverage state.
        self.obs_k_per_dim = self.num_k_per_dim

        self.nx = 4
        self.nu = 2
        self.total_state_dim = self.nx * self.num_players
        self.total_action_dim = self.nu * self.num_players

        self.players_u_index_list = torch.tensor(
            [[i * 2, i * 2 + 1] for i in range(self.num_players)]
        )

        state_low = np.tile([0.0, 0.0, -velocity_limit, -velocity_limit], num_agents)
        state_high = np.tile([1.0, 1.0, velocity_limit, velocity_limit], num_agents)
        obs_coeff_dim = self.obs_k_per_dim * self.obs_k_per_dim
        obs_low = np.concatenate(
            [state_low, -self.obs_clip * np.ones(obs_coeff_dim, dtype=np.float64)]
        )
        obs_high = np.concatenate(
            [state_high, self.obs_clip * np.ones(obs_coeff_dim, dtype=np.float64)]
        )
        action_low = -action_limit * np.ones(self.total_action_dim)
        action_high = action_limit * np.ones(self.total_action_dim)

        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float64
        )
        self.action_space = spaces.Box(
            low=action_low, high=action_high, dtype=np.float64
        )

        self.L_list = np.array([1.0, 1.0])

        self._initialize_fourier_basis()
        self.phik_list = self._compute_target_coefficients()
        obs_mask = (
            (self.ks[:, 0] < self.obs_k_per_dim)
            & (self.ks[:, 1] < self.obs_k_per_dim)
        )
        self.obs_coeff_indices = np.flatnonzero(obs_mask)
        self.obs_coeff_dim = len(self.obs_coeff_indices)

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

    def _compute_coverage_error_obs(self):
        if self.current_time > 0:
            ck = self.ck_list_update / self.current_time
        else:
            ck = np.zeros_like(self.ck_list_update)
        s_obs = ck[self.obs_coeff_indices] - self.phik_list[self.obs_coeff_indices]
        return np.clip(s_obs, -self.obs_clip, self.obs_clip)

    def _get_obs(self):
        return np.concatenate([self.state, self._compute_coverage_error_obs()])

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

        # ---- Ergodic metric BEFORE update ----
        old_metric = self._compute_ergodic_metric()

        # ---- Update Fourier coefficients (trajectory statistics) ----
        fk = self._evaluate_fourier_basis(new_pos)
        self.ck_list_update += fk * self.dt
        self.current_time += self.dt

        # ---- Ergodic metric AFTER update ----
        new_metric = self._compute_ergodic_metric()
        delta_metric = new_metric - old_metric

        # ---- Update visit grid (for coverage tracking only) ----
        for i in range(self.num_agents):
            gx = min(int(new_pos[i, 0] * self.grid_res), self.grid_res - 1)
            gy = min(int(new_pos[i, 1] * self.grid_res), self.grid_res - 1)
            self.visit_grid[gx, gy] += 1

        global_metric_term = self.w_metric * (-delta_metric)

        new_agents = np.column_stack([new_pos, new_vel])
        self.state = new_agents.ravel()
        self.step_count += 1

        costs = np.zeros(self.num_players)
        for i in range(self.num_players):
            control_penalty = self.w_ctrl * np.sum(accel[i] ** 2)
            costs[i] = global_metric_term / self.num_players - control_penalty

        reward = float(np.sum(costs))

        terminated = False
        truncated = self.step_count >= self.max_episode_steps

        info = {
            "ergodic_metric": float(new_metric),
            "delta_metric": float(delta_metric),
            "coverage_fraction": float(
                np.sum(self.visit_grid > 0) / (self.grid_res**2)
            ),
            "mean_speed": float(np.mean(np.linalg.norm(new_vel, axis=1))),
            "time": float(self.current_time),
            "step": self.step_count,
            "individual_cost": costs,
        }
        return self._get_obs(), reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if options is not None and "initial_state" in options:
            initial_state = np.asarray(options["initial_state"], dtype=np.float64)
            if initial_state.shape[0] == self.observation_space.shape[0]:
                initial_state = initial_state[: self.total_state_dim]
            self.state = initial_state
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
        return self._get_obs(), {}

    def render(self):
        pass

    # =========================================================================
    # MA-GPS: costs_jacobian_and_hessian
    # =========================================================================
    # NOTE: This provides a STATIC directional hint toward high-PDF regions.
    # It CANNOT encode the trajectory-level ergodic objective (Mathew & Mezić
    # 2011, §3). The actual coverage behavior comes from the RL reward above.
    # Use with --behavior-loss-weight 0.1 so RL dominates.
    # =========================================================================

    def costs_jacobian_and_hessian(self, z):
        batch_size, input_dim = z.shape
        num_players = self.num_players
        state_dim = self.total_state_dim
        alpha = 0.2
        grad_scale = 1.0
        pos_hess = 5.0
        vel_hess = 0.5

        ks = self.lq_ks_torch.to(device=z.device, dtype=z.dtype)
        wk = self.lq_wk_torch.to(device=z.device, dtype=z.dtype)
        pi = 3.141592653589793

        jacobians = torch.zeros(
            num_players, batch_size, input_dim, device=z.device, dtype=z.dtype
        )
        hessians = torch.zeros(
            num_players, batch_size, input_dim, input_dim,
            device=z.device, dtype=z.dtype,
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

            # SMC Fourier gradient (static — directional hint only)
            jac_x = grad_scale * torch.sum(wk * k1 * pi * sin_k1x * cos_k2y, dim=1)
            jac_y = grad_scale * torch.sum(wk * k2 * pi * cos_k1x * sin_k2y, dim=1)

            # Boundary repulsion
            margin = 0.02
            w_wall = 0.5
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

            # Inter-agent repulsion
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

            # Velocity: penalize stillness (negative cost gradient for speed)
            jacobians[i, :, ivx] = -0.2 * z[:, ivx]
            jacobians[i, :, ivy] = -0.2 * z[:, ivy]

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
        batch_size = states.shape[0]
        n = states.shape[1]
        m = controls.shape[1]
        num_players = n // 4
        dt = 0.05
        damp = 0.5
        dt_damped = dt * (1.0 - damp * dt)
        dt_sq = dt * dt

        jacobian = torch.zeros(batch_size, n, n + m, device=states.device)
        for i in range(num_players):
            sx, sy, svx, svy = i * 4, i * 4 + 1, i * 4 + 2, i * 4 + 3
            cax, cay = n + i * 2, n + i * 2 + 1

            jacobian[:, sx, sx] = 1.0
            jacobian[:, sx, svx] = dt_damped
            jacobian[:, sx, cax] = dt_sq

            jacobian[:, sy, sy] = 1.0
            jacobian[:, sy, svy] = dt_damped
            jacobian[:, sy, cay] = dt_sq

            jacobian[:, svx, svx] = 1.0 - damp * dt
            jacobian[:, svx, cax] = dt

            jacobian[:, svy, svy] = 1.0 - damp * dt
            jacobian[:, svy, cay] = dt

        return jacobian

    @torch.jit.script
    def dynamics(states, controls):
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
