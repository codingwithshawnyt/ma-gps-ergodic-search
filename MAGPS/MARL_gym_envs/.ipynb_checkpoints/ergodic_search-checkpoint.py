from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
from scipy.stats import multivariate_normal as mvn


class ErgodicSearchEnv(gym.Env):
    """
    Multi-agent ergodic search environment.

    Agents with first-order integrator dynamics search a 2D unit square
    to match a target spatial distribution (Gaussian mixture).

    State: [x1, y1, x2, y2, ..., xn, yn] (positions of all agents)
    Action: [vx1, vy1, vx2, vy2, ..., vxn, vyn] (velocities of all agents)
    Reward: -scale * (ergodic_metric_t - ergodic_metric_{t-1})

    The ergodic metric measures how well the agents' trajectory matches
    the target distribution in the Fourier domain.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(
        self,
        num_agents: int = 2,
        num_k_per_dim: int = 10,
        dt: float = 0.1,
        max_episode_steps: int = 200,
        action_limit: float = 1.0,
        reward_scale: float = 1000.0,
        init_pos_low: float = 0.2,
        init_pos_high: float = 0.8,
        boundary_termination: bool = True,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        # Store configuration
        self.num_agents = num_agents
        self.num_k_per_dim = num_k_per_dim
        self.dt = dt
        self.max_episode_steps = max_episode_steps
        self.action_limit = action_limit
        self.reward_scale = reward_scale
        self.init_pos_low = init_pos_low
        self.init_pos_high = init_pos_high
        self.boundary_termination = boundary_termination
        self.render_mode = render_mode

        # Search space dimensions (unit square)
        self.L_list = np.array([1.0, 1.0])

        # Define state and action spaces
        self.state_dim = 2 * self.num_agents  # x,y for each agent
        self.action_dim = 2 * self.num_agents  # vx,vy for each agent

        # State space: positions in [0,1] for each agent
        self.state_low = np.zeros(self.state_dim)
        self.state_high = np.ones(self.state_dim)

        # Action space: velocities in [-action_limit, action_limit]
        self.action_low = -self.action_limit * np.ones(self.action_dim)
        self.action_high = self.action_limit * np.ones(self.action_dim)

        self.observation_space = spaces.Box(
            low=self.state_low, high=self.state_high, dtype=np.float64
        )

        self.action_space = spaces.Box(
            low=self.action_low, high=self.action_high, dtype=np.float64
        )

        # Initialize Fourier basis setup
        self._initialize_fourier_basis()

        # Pre-compute target distribution coefficients
        self.phik_list = self._compute_target_coefficients()

        # Running state (initialized in reset)
        self.state = None
        self.ck_list_update = None
        self.current_time = None
        self.previous_metric = None
        self.step_count = None

        # For MA-GPS compatibility
        self.is_nonlinear_game = True

    def _initialize_fourier_basis(self):
        """Initialize Fourier basis index vectors and normalization terms."""
        # Generate all k-index vectors
        ks_dim1, ks_dim2 = np.meshgrid(
            np.arange(self.num_k_per_dim), np.arange(self.num_k_per_dim)
        )
        self.ks = np.array([ks_dim1.ravel(), ks_dim2.ravel()]).T

        # Pre-compute normalization terms hk
        # Create grid for normalization
        grids_x, grids_y = np.meshgrid(
            np.linspace(0, self.L_list[0], 100), np.linspace(0, self.L_list[1], 100)
        )
        grids = np.array([grids_x.ravel(), grids_y.ravel()]).T
        dx = self.L_list[0] / 99
        dy = self.L_list[1] / 99

        self.hk_list = np.zeros(self.ks.shape[0])
        for i, k_vec in enumerate(self.ks):
            fk_vals = np.prod(np.cos(np.pi * k_vec * grids), axis=1)
            hk = np.sqrt(np.sum(np.square(fk_vals)) * dx * dy)
            self.hk_list[i] = hk

        # Pre-compute lambda weights
        self.lamk_list = np.power(1.0 + np.linalg.norm(self.ks, axis=1), -3 / 2.0)

    def _target_pdf(self, positions):
        """
        Evaluate Gaussian mixture PDF at given positions.

        Args:
            positions: array of shape (n_points, 2)

        Returns:
            pdf_vals: array of shape (n_points,)
        """
        # Gaussian mixture from tutorial
        mean1 = np.array([0.35, 0.38])
        cov1 = np.array([[0.01, 0.004], [0.004, 0.01]])
        w1 = 0.5

        mean2 = np.array([0.68, 0.25])
        cov2 = np.array([[0.005, -0.003], [-0.003, 0.005]])
        w2 = 0.2

        mean3 = np.array([0.56, 0.64])
        cov3 = np.array([[0.008, 0.0], [0.0, 0.004]])
        w3 = 0.3

        pdf_vals = (
            w1 * mvn.pdf(positions, mean1, cov1)
            + w2 * mvn.pdf(positions, mean2, cov2)
            + w3 * mvn.pdf(positions, mean3, cov3)
        )

        return pdf_vals

    def _compute_target_coefficients(self):
        """Compute Fourier coefficients of target Gaussian mixture distribution."""
        # Discretize search space
        grids_x, grids_y = np.meshgrid(
            np.linspace(0, self.L_list[0], 100), np.linspace(0, self.L_list[1], 100)
        )
        grids = np.array([grids_x.ravel(), grids_y.ravel()]).T
        dx = self.L_list[0] / 99
        dy = self.L_list[1] / 99

        # Evaluate target PDF at all grid points
        pdf_vals = self._target_pdf(grids)

        # Compute coefficients for each k vector
        phik_list = np.zeros(self.ks.shape[0])
        for i, k_vec in enumerate(self.ks):
            fk_vals = np.prod(np.cos(np.pi * k_vec * grids), axis=1)
            hk = self.hk_list[i]
            fk_vals /= hk
            phik = np.sum(fk_vals * pdf_vals) * dx * dy
            phik_list[i] = phik

        return phik_list

    def _evaluate_fourier_basis(self, positions):
        """
        Evaluate all Fourier basis functions at given positions.

        For multi-agent case, positions is (num_agents, 2).
        Returns average of Fourier basis across all agents.

        Args:
            positions: array of shape (num_agents, 2)

        Returns:
            fk_vals: array of shape (num_k,)
        """
        fk_vals_all_agents = np.zeros(self.ks.shape[0])

        for agent_pos in positions:
            # Evaluate fk for this agent
            fk_vals = np.prod(np.cos(np.pi * self.ks * agent_pos), axis=1)
            fk_vals /= self.hk_list
            fk_vals_all_agents += fk_vals

        # Average over agents
        return fk_vals_all_agents / self.num_agents

    def _compute_ergodic_metric(self):
        """
        Compute current ergodic metric.

        Returns:
            metric: scalar, sum of lambda_k * (ck - phik)^2
        """
        # Normalize running coefficients by time
        if self.current_time > 0:
            ck_current = self.ck_list_update / self.current_time
        else:
            ck_current = np.zeros_like(self.ck_list_update)

        # Compute weighted squared error
        metric = np.sum(self.lamk_list * np.square(ck_current - self.phik_list))

        return metric

    def step(self, action):
        """
        Execute one timestep.

        Args:
            action: array of shape (2*num_agents,), velocities [vx1, vy1, vx2, vy2, ...]

        Returns:
            state: new positions
            reward: -scale * (metric_t - metric_{t-1})
            terminated: bool
            truncated: bool
            info: dict with metrics
        """
        # Reshape for easier manipulation
        positions = self.state.reshape(self.num_agents, 2)
        velocities = action.reshape(self.num_agents, 2)

        # 1. Update positions (first-order integrator)
        new_positions = positions + self.dt * velocities
        new_state = new_positions.ravel()

        # 2. Compute previous metric (before update)
        metric_before = self._compute_ergodic_metric()

        # 3. Update running Fourier coefficients
        fk_vals = self._evaluate_fourier_basis(new_positions)
        self.ck_list_update += fk_vals * self.dt
        self.current_time += self.dt

        # 4. Compute new metric
        metric_after = self._compute_ergodic_metric()

        # 5. Compute reward (negative change in metric, scaled)
        reward = -self.reward_scale * (metric_after - metric_before)

        # 6. Update state
        self.state = new_state
        self.step_count += 1

        # 7. Check termination conditions
        terminated = False
        truncated = False

        # Boundary check
        if self.boundary_termination:
            if np.any(new_positions < 0.0) or np.any(new_positions > 1.0):
                terminated = True

        # Time limit check
        if self.step_count >= self.max_episode_steps:
            truncated = True

        # 8. Store metric for next iteration
        self.previous_metric = metric_after

        # 9. Info dict
        info = {
            "ergodic_metric": float(metric_after),
            "metric_improvement": float(metric_before - metric_after),
            "time": float(self.current_time),
            "step": self.step_count,
        }

        return self.state, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment to initial state."""
        super().reset(seed=seed)

        # Initialize positions randomly
        if options is not None and "initial_state" in options:
            self.state = options["initial_state"]
        else:
            positions = self.np_random.uniform(
                low=self.init_pos_low,
                high=self.init_pos_high,
                size=(self.num_agents, 2),
            )
            self.state = positions.ravel()

        # Reset running state
        self.ck_list_update = np.zeros(self.ks.shape[0])
        self.current_time = 0.0
        self.previous_metric = self._compute_ergodic_metric()  # initial metric
        self.step_count = 0

        return self.state, {}

    def render(self):
        """Render the environment (placeholder)."""
        pass

    @torch.jit.script
    def costs_jacobian_and_hessian(z):
        """
        Compute Jacobian and Hessian of ergodic metric cost.

        NOTE: This is a placeholder. Proper implementation requires
        tracking trajectory history and is complex for JIT compilation.

        Args:
            z: (batch_size, 4*num_agents) = [state, control]

        Returns:
            jacobians: (1, batch_size, 4*num_agents) with zeros
            hessians: (1, batch_size, 4*num_agents, 4*num_agents) with zeros
        """
        batch_size, input_dim = z.shape

        jacobians = torch.zeros(1, batch_size, input_dim, device=z.device)
        hessians = torch.zeros(1, batch_size, input_dim, input_dim, device=z.device)

        return jacobians, hessians

    @torch.jit.script
    def dynamics_jacobian(states, controls):
        """
        Jacobian of dynamics w.r.t. [state, control].

        For first-order integrator: x' = x + dt*u
        ∂x'/∂x = I (identity)
        ∂x'/∂u = dt*I

        Args:
            states: (batch_size, 2*num_agents)
            controls: (batch_size, 2*num_agents)

        Returns:
            jacobian: (batch_size, 2*num_agents, 4*num_agents)
                      [∂x'/∂x, ∂x'/∂u]
        """
        batch_size = states.shape[0]
        n = states.shape[1]  # 2*num_agents
        dt = 0.1

        jacobian = torch.zeros(batch_size, n, 2 * n, device=states.device)

        # ∂x'/∂x = I
        jacobian[:, :, :n] = torch.eye(n, device=states.device)

        # ∂x'/∂u = dt*I
        jacobian[:, :, n:] = dt * torch.eye(n, device=states.device)

        return jacobian

    @torch.jit.script
    def dynamics(states, controls):
        """
        First-order integrator dynamics.

        Args:
            states: (batch_size, 2*num_agents), positions [x1, y1, x2, y2, ...]
            controls: (batch_size, 2*num_agents), velocities [vx1, vy1, vx2, vy2, ...]

        Returns:
            next_states: (batch_size, 2*num_agents)
        """
        dt = 0.1
        next_states = states + dt * controls
        return next_states
