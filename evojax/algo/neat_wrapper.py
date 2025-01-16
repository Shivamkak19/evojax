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
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: int = 20,
        pop_size: int = 128,
        species_size: int = 10,
        max_nodes: int = 50,
        max_conns: int = 200,
        seed: int = 0,
    ):
        super().__init__()
        self.pop_size = pop_size

        # Create NEAT algorithm with more varied initialization
        self.neat = TensorNEAT(
            genome=DefaultGenome(
                num_inputs=input_dim,
                num_outputs=output_dim,
                max_nodes=max_nodes,
                max_conns=max_conns,
                node_gene=BiasNode(
                    activation_options=ACT.tanh,
                    aggregation_options=AGG.sum,
                    bias_init_std=1.0,  # Increased for more variation
                    bias_mutate_rate=0.3,  # Increased mutation rate
                    bias_replace_rate=0.1,
                ),
                conn_gene=DefaultConn(
                    weight_init_std=1.0,  # Increased for more variation
                    weight_mutate_rate=0.3,  # Increased mutation rate
                    weight_replace_rate=0.1,
                ),
                output_transform=ACT.tanh,
            ),
            pop_size=pop_size,
            species_size=species_size,
            survival_threshold=0.2,  # Adjusted for more selection pressure
            compatibility_threshold=3.0,  # Increased for more speciation
        )

        # Initialize state
        self.state = State(randkey=jax.random.PRNGKey(seed))
        self.state = self.neat.setup(self.state)

        self._best_idx = None
        self.best_fitness = float("-inf")

        # Store current population
        self.current_nodes = None
        self.current_conns = None

        # Reference to policy for parameter updates
        self.policy = None

    def ask(self) -> jnp.ndarray:
        """Get current population parameters."""
        # Get population from NEAT
        nodes, conns = self.neat.ask(self.state)

        # Ensure proper shapes
        nodes = jnp.array(nodes)
        conns = jnp.array(conns)

        # Store current population with proper shapes
        self.current_nodes = nodes
        self.current_conns = conns

        # Update policy's parameters
        if self.policy is not None:
            self.policy.set_params(self.current_nodes, self.current_conns)

        return jnp.arange(self.pop_size)

    def tell(self, fitness: Union[np.ndarray, jnp.ndarray]) -> None:
        """Update algorithm with fitness results."""
        if isinstance(fitness, np.ndarray):
            fitness = jnp.array(fitness)

        self.state = self.neat.tell(self.state, fitness)

        # Update best params
        best_idx = jnp.argmax(fitness)
        if fitness[best_idx] > self.best_fitness:
            self.best_fitness = fitness[best_idx]
            self._best_idx = best_idx.astype(jnp.int32)

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
