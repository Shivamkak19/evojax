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
from tensorneat.src.tensorneat.genome.gene import DefaultConn
from tensorneat.src.tensorneat.genome import DefaultGenome, BiasNode
from tensorneat.src.tensorneat.common import ACT, AGG, State


class NEATPolicy(PolicyNetwork):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: int = 20,
        max_nodes: int = 50,
        max_conns: int = 200,
        logger=None,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._logger = logger or create_logger("NEATPolicy")
        
        # Create initial hidden layer structure
        init_hidden_layers = (hidden_size, hidden_size//2, hidden_size//4)  # Create multiple hidden layers
        
        self.genome = DefaultGenome(
            num_inputs=input_dim,
            num_outputs=output_dim,
            max_nodes=max_nodes,
            max_conns=max_conns,
            node_gene=BiasNode(
                # Use more activation functions
                activation_options=[
                    ACT.tanh,
                    ACT.sigmoid,
                    ACT.relu,
                    ACT.lelu,
                    ACT.sin,
                    ACT.scaled_tanh,
                    ACT.scaled_sigmoid
                ],
                aggregation_options=[
                    AGG.sum,
                    AGG.product,
                    AGG.mean,
                    AGG.max,
                    AGG.min
                ],
                # Increase mutation rates for more exploration
                bias_init_std=1.5,
                bias_mutate_rate=0.4,
                bias_replace_rate=0.2,
                activation_replace_rate=0.3  # Increase likelihood of trying different activation functions
            ),
            conn_gene=DefaultConn(
                weight_init_std=1.5,
                weight_mutate_rate=0.4,
                weight_replace_rate=0.2,
            ),
            output_transform=ACT.tanh,
            init_hidden_layers=init_hidden_layers  # Add initial hidden layer structure
        )

        self.state = State(randkey=jax.random.PRNGKey(0))
        self.state = self.genome.setup(self.state)
        self.current_nodes = None
        self.current_conns = None
        self.genome = self.genome

    def set_params(self, nodes, conns):
        """Set current population parameters."""
        self.current_nodes = nodes
        self.current_conns = conns

    def get_actions(
        self,
        t_states: TaskState,
        params: jnp.ndarray,
        p_states: PolicyState
    ) -> Tuple[jnp.ndarray, PolicyState]:
        """Get actions from observations using current parameters."""
        # Reshape observation if needed
        obs = t_states.obs
        if len(obs.shape) == 1:
            obs = obs.reshape(1, -1)

        # Extract index and get network parameters
        idx = jnp.asarray(params[0] if len(params.shape) > 0 else params)
        
        # Instead of dynamic slice, use regular indexing
        nodes = jnp.array(self.current_nodes[idx])
        conns = jnp.array(self.current_conns[idx])

        # Ensure shapes are correct before transformation
        if len(nodes.shape) == 3:  # If we got a batch dimension
            nodes = nodes[0]  # Take first element
        if len(conns.shape) == 3:  # If we got a batch dimension
            conns = conns[0]  # Take first element

        # Transform and forward pass
        transformed = self.genome.transform(self.state, nodes, conns)
        actions = self.genome.forward(self.state, transformed, obs)

        # Ensure actions have correct shape
        if len(actions.shape) == 1:
            actions = actions.reshape(1, -1)

        return actions, p_states