from typing import Union
import jax
import jax.numpy as jnp
import numpy as np
from evojax.algo.base import NEAlgorithm
from tensorneat.src.tensorneat.algorithm.neat import NEAT as TensorNEAT
from tensorneat.src.tensorneat.common import State
from tensorneat.src.tensorneat.genome import DefaultGenome, BiasNode
from tensorneat.src.tensorneat.genome.gene import DefaultConn
from tensorneat.src.tensorneat.common import ACT, AGG


class NEATWrapper(NEAlgorithm):
    def __init__(self, input_dim, output_dim, pop_size=128, hidden_size=20, 
                 species_size=10, max_nodes=50, max_conns=200, seed=0):
        super().__init__()
        self.pop_size = pop_size

        # Configure NEAT specifically for slimevolley
        self.neat = TensorNEAT(
            genome=DefaultGenome(
                num_inputs=input_dim, 
                num_outputs=output_dim,  # Should be 3 for slimevolley
                max_nodes=max_nodes,
                max_conns=max_conns, 
                init_hidden_layers=(hidden_size, hidden_size//2),
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
                output_transform=ACT.tanh,  # Important for control tasks
            ),
            pop_size=pop_size,
            species_size=species_size,
            survival_threshold=0.2,
            compatibility_threshold=3.0,
        )

        self.state = State(randkey=jax.random.PRNGKey(seed)) 
        self.state = self.neat.setup(self.state)
        
        self._best_idx = None
        self.best_fitness = float("-inf")
        self.current_nodes = None
        self.current_conns = None
        self.policy = None

    def ask(self) -> jnp.ndarray:
        """Get current population."""
        nodes, conns = self.neat.ask(self.state)
        
        # Store current population
        self.current_nodes = nodes
        self.current_conns = conns
        
        # Update policy if available
        if self.policy is not None:
            self.policy.set_params(nodes, conns)
            
        # Return indices into population
        return jnp.arange(self.pop_size)
        
    def tell(self, fitness: Union[np.ndarray, jnp.ndarray]) -> None:
        # Update NEAT state with fitness scores
        if isinstance(fitness, np.ndarray):
            fitness = jnp.array(fitness)
            
        self.state = self.neat.tell(self.state, fitness)
        
        # Track best performing network
        best_idx = jnp.argmax(fitness)
        if fitness[best_idx] > self.best_fitness:
            self.best_fitness = fitness[best_idx]
            self._best_idx = int(best_idx)

    @property
    def best_params(self) -> jnp.ndarray:
        if self._best_idx is None:
            raise ValueError("No best parameters available yet")
        return jnp.array([self._best_idx], dtype=jnp.int32)

    @best_params.setter
    def best_params(self, params: Union[np.ndarray, jnp.ndarray]) -> None:
        """Set best parameters - not implemented as we use indices."""
        raise NotImplementedError("Setting best_params directly not supported in NEAT")

    def set_policy(self, policy):
        """Set reference to policy for parameter updates."""
        self.policy = policy