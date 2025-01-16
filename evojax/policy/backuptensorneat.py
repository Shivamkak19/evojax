import sys
import os

# Get the absolute path to the directory containing evojax and tensorneat
current_dir = os.path.dirname(os.path.abspath(__file__))
evojax_dir = os.path.dirname(os.path.dirname(current_dir))
project_root = os.path.dirname(evojax_dir)
sys.path.append(project_root)

import jax
import jax.numpy as jnp
from typing import Tuple, Optional
from evojax.policy.base import PolicyNetwork, PolicyState
from evojax.task.base import TaskState
from evojax.util import create_logger
from tensorneat.src.tensorneat.genome import DefaultGenome, BiasNode
from tensorneat.src.tensorneat.common import ACT, AGG, State


class NEATPolicy(PolicyNetwork):
    """Custom PolicyNetwork that works with TensorNEAT's networks."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: int = 20,
        max_nodes: int = 50,
        max_conns: int = 200,
        logger=None,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_nodes = max_nodes
        self.max_conns = max_conns
        self._logger = logger or create_logger("NEATPolicy")

        # Create genome template
        self.genome = DefaultGenome(
            num_inputs=input_dim,
            num_outputs=output_dim,
            max_nodes=max_nodes,
            max_conns=max_conns,
            node_gene=BiasNode(
                activation_options=ACT.tanh,
                aggregation_options=AGG.sum,
            ),
            output_transform=ACT.tanh,
        )

        # Initialize state with randkey
        initial_key = jax.random.PRNGKey(0)
        self.state = State(randkey=initial_key)
        self.state = self.genome.setup(self.state)

        # Initialize nodes and connections
        init_key, new_key = jax.random.split(initial_key)
        self.nodes, self.conns = self.genome.initialize(self.state, init_key)
        self.state = State(randkey=new_key)  # Update state with new key

        # Store default params
        self.params = (self.nodes, self.conns, self.state)

    def reset(self, task_state: Optional[TaskState] = None) -> PolicyState:
        """Reset policy state."""
        batch_size = task_state.obs.shape[0] if task_state is not None else 1
        return jnp.zeros((batch_size, 1))  # Simple policy state

    @property
    def num_params(self) -> int:
        """Return total number of parameters."""
        return self.max_nodes * 3 + self.max_conns * 4  # Approximate parameter count

    def init_params(self, init_scale: float = 0.1) -> jnp.ndarray:
        """Initialize parameters."""
        # Return indices that will be used to look up actual parameters
        return jnp.zeros(1)

    def set_params(self, params):
        """Set current network parameters."""
        if isinstance(params, tuple) and len(params) == 3:
            self.nodes, self.conns, self.state = params
            self.params = params
        else:
            raise ValueError("Expected params to be tuple of (nodes, conns, state)")

    def get_actions(
        self, t_states: TaskState, params: jnp.ndarray, p_states: PolicyState
    ) -> Tuple[jnp.ndarray, PolicyState]:
        """Get actions from observations using current network."""
        obs = t_states.obs
        # Ensure obs has correct batch shape
        if len(obs.shape) == 1:
            obs = obs[None, :]  # Add batch dimension if not present
        elif len(obs.shape) > 2:
            # Reshape if needed for batch processing
            batch_size = obs.shape[0]
            obs = obs.reshape(batch_size, -1)

        # Use current parameters
        nodes, conns, neat_state = self.params

        # Transform network
        transformed = self.genome.transform(neat_state, nodes, conns)

        # Forward pass
        actions = self.genome.forward(neat_state, transformed, obs)

        return actions, p_states
