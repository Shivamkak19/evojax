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
    def __init__(self, input_dim, output_dim, hidden_size=20, max_nodes=50, max_conns=2000, logger=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._logger = logger or create_logger("NEATPolicy")
        
        init_hidden_layers = (hidden_size, hidden_size//2)
        
        self.genome = DefaultGenome(
            num_inputs=input_dim,
            num_outputs=output_dim,
            max_nodes=max_nodes,
            max_conns=max_conns,
            node_gene=BiasNode(
                activation_options=[ACT.tanh, ACT.sigmoid, ACT.relu],
                aggregation_options=[AGG.sum],
                bias_init_std=1.0,
                bias_mutate_rate=0.3,
            ),
            conn_gene=DefaultConn(
                weight_init_std=1.0,
                weight_mutate_rate=0.3,
            ),
            output_transform=ACT.tanh,  # Important for slimevolley control
            init_hidden_layers=init_hidden_layers
        )
        
        self.state = State(randkey=jax.random.PRNGKey(0))
        self.state = self.genome.setup(self.state)
        self.current_nodes = None
        self.current_conns = None

    def get_actions(self, t_states, params, p_states):
        if self.current_nodes is None or self.current_conns is None:
            raise ValueError("Policy parameters not set.")

        # Handle observation shape
        obs = t_states.obs
        batch_size = obs.shape[0] if len(obs.shape) > 1 else 1
        obs = obs.reshape(batch_size, -1)

        # Convert params to indices
        indices = jnp.asarray(params, dtype=jnp.int32)
        if len(indices.shape) == 0:
            indices = indices.reshape(1)

        # Get corresponding networks
        nodes = jnp.take(self.current_nodes, indices, axis=0)
        conns = jnp.take(self.current_conns, indices, axis=0)

        def process_single(node, conn, ob):
            # Remove extra batch dimension from nodes and conns
            node = node.reshape(node.shape[-2:])  # (max_nodes, node_attrs)
            conn = conn.reshape(conn.shape[-2:])  # (max_conns, conn_attrs)
            
            # Transform and get action
            transformed = self.genome.transform(self.state, node, conn)
            action = self.genome.forward(self.state, transformed, ob)
            
            # Ensure consistent output shape
            return jnp.reshape(action, (self.output_dim,))

        # Process batch
        actions = jax.vmap(process_single)(nodes, conns, obs) 
        return actions, p_states

    def reset(self, t_states):
        """Reset policy state."""
        return None  # No state needed for feedforward networks

    def set_params(self, nodes, conns): 
        """Set current population parameters."""
        self.current_nodes = jax.device_put(nodes)
        self.current_conns = jax.device_put(conns)