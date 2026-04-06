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
        self.num_players = num_agents  # MA-GPS compatibility
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

        # MA-GPS compatibility attributes
        self.nx = 2  # state dims per player (x, y)
        self.nu = 2  # control dims per player (vx, vy)
        self.total_state_dim = self.nx * self.num_players
        self.total_action_dim = self.nu * self.num_players

        # Create index list for player actions
        self.players_u_index_list = torch.tensor(
            [[i * 2, i * 2 + 1] for i in range(self.num_players)]
        )

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

        # Basis matrices for cost function definition (following unicycle pattern)
        self.basis = np.eye(self.total_state_dim)
        self.basis_u = np.eye(self.total_action_dim)

        # Define per-player cost functions
        # Proxy cost: -log(p(x_i, y_i)) + α * (u_xi² + u_yi²)
        # Encourages agents to spend time in high-density regions
        self.cost_functions = []
        alpha = 0.1  # Control penalty weight

        for i in range(self.num_players):
            # Cost for player i: depends on their position and control
            def make_player_cost(player_idx):
                def player_cost(x, u):
                    # Extract this player's position
                    pos_x = x[player_idx * 2]
                    pos_y = x[player_idx * 2 + 1]

                    # Extract this player's control
                    control_x = u[player_idx * 2]
                    control_y = u[player_idx * 2 + 1]

                    # Target PDF value at position (add epsilon to avoid log(0))
                    pdf_val = self._target_pdf_single(np.array([pos_x, pos_y]))
                    pdf_val = max(pdf_val, 1e-10)

                    # Cost = -log(pdf) + alpha * control penalty
                    cost_val = -np.log(pdf_val) + alpha * (control_x**2 + control_y**2)
                    return cost_val

                return player_cost

            self.cost_functions.append(make_player_cost(i))

        # Store alpha for cost functions
        self.cost_alpha = alpha

    def _target_pdf_single(self, position):
        """
        Evaluate Gaussian mixture PDF at a single position.

        Args:
            position: array of shape (2,)

        Returns:
            pdf_val: scalar
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

        pdf_val = (
            w1 * mvn.pdf(position, mean1, cov1)
            + w2 * mvn.pdf(position, mean2, cov2)
            + w3 * mvn.pdf(position, mean3, cov3)
        )

        return pdf_val

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
            reward: -scale * (metric_t - metric_{t-1})  (ergodic metric improvement)
            terminated: bool
            truncated: bool
            info: dict with metrics and individual costs
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

        # 6. Compute individual costs for MA-GPS (proxy costs)
        self.costs = np.array(
            [cost(self.state, action) for cost in self.cost_functions]
        )

        # 7. Update state
        self.state = new_state
        self.step_count += 1

        # 8. Check termination conditions
        terminated = False
        truncated = False

        # Boundary check
        if self.boundary_termination:
            if np.any(new_positions < 0.0) or np.any(new_positions > 1.0):
                terminated = True

        # Time limit check
        if self.step_count >= self.max_episode_steps:
            truncated = True

        # 9. Store metric for next iteration
        self.previous_metric = metric_after

        # 10. Info dict (following unicycle pattern)
        info = {
            "ergodic_metric": float(metric_after),
            "metric_improvement": float(metric_before - metric_after),
            "time": float(self.current_time),
            "step": self.step_count,
            "individual_cost": self.costs,  # MA-GPS compatibility
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
        Compute Jacobian and Hessian of proxy cost functions.

        Proxy cost for player i: cost_i = -log(p(x_i, y_i)) + α * (u_xi² + u_yi²)

        Gradient of -log(p(x)) = -(∇p(x))/p(x)

        Args:
            z: (batch_size, input_dim) where input_dim = 2*num_players + 2*num_players
                     [x1, y1, x2, y2, ..., vx1, vy1, vx2, vy2, ...]

        Returns:
            jacobians: (num_players, batch_size, input_dim)
            hessians: (num_players, batch_size, input_dim, input_dim)
        """
        batch_size, input_dim = z.shape
        num_players = input_dim // 4  # input_dim = 4*num_players
        state_dim = 2 * num_players
        alpha = 0.1  # Control penalty weight

        # Weighted center of Gaussian mixture:
        # 0.5*(0.35,0.38) + 0.2*(0.68,0.25) + 0.3*(0.56,0.64)
        target_x = 0.479
        target_y = 0.412
        w_pos = 10.0  # Position cost weight

        jacobians = torch.zeros(num_players, batch_size, input_dim, device=z.device)
        hessians = torch.zeros(
            num_players, batch_size, input_dim, input_dim, device=z.device
        )

        for i in range(num_players):
            # State indices
            sx = i * 2
            sy = i * 2 + 1
            # Control indices
            cx = state_dim + i * 2
            cy = state_dim + i * 2 + 1

            # Jacobians: ∂cost/∂state = 2*w_pos*(pos - target), ∂cost/∂control = 2*alpha*control
            jacobians[i, :, sx] = 2 * w_pos * (z[:, sx] - target_x)
            jacobians[i, :, sy] = 2 * w_pos * (z[:, sy] - target_y)
            jacobians[i, :, cx] = 2 * alpha * z[:, cx]
            jacobians[i, :, cy] = 2 * alpha * z[:, cy]

            # Hessians: ∂²cost/∂state² = 2*w_pos*I, ∂²cost/∂control² = 2*alpha*I
            hessians[i, :, sx, sx] = 2 * w_pos
            hessians[i, :, sy, sy] = 2 * w_pos
            hessians[i, :, cx, cx] = 2 * alpha
            hessians[i, :, cy, cy] = 2 * alpha

        return jacobians, hessians

    @torch.jit.script
    def dynamics_jacobian(states, controls):
        """
        Jacobian of dynamics w.r.t. [state, control].

        For first-order integrator: x' = x + dt*u
        ∂x'/∂x = I (identity)
        ∂x'/∂u = dt*I

        Args:
            states: (batch_size, 2*num_players)
            controls: (batch_size, 2*num_players)

        Returns:
            jacobian: (batch_size, 2*num_players, 4*num_players)
                      [∂x'/∂x, ∂x'/∂u]
        """
        batch_size = states.shape[0]
        n = states.shape[1]  # 2*num_players
        dt = 0.1

        jacobian = torch.zeros(batch_size, n, 2 * n, device=states.device)

        # ∂x'/∂x = I, ∂x'/∂u = dt*I (vectorized)
        eye_n = torch.eye(n, device=states.device)
        jacobian[:, :, :n] = eye_n.unsqueeze(0).expand(batch_size, -1, -1)
        jacobian[:, :, n:] = dt * eye_n.unsqueeze(0).expand(batch_size, -1, -1)

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
