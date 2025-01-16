from typing import Union
import jax
import jax.numpy as jnp
import numpy as np
from evojax.algo.base import NEAlgorithm
from tensorneat.src.tensorneat.algorithm.neat import NEAT as TensorNEAT
from tensorneat.src.tensorneat.common import State
from tensorneat.src.tensorneat.genome import DefaultGenome, BiasNode
from tensorneat.src.tensorneat.common import ACT, AGG


class NEATWrapper(NEAlgorithm):
    """Wrapper to make TensorNEAT compatible with EvoJAX."""

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
        self._best_params = None
        self.best_fitness = float("-inf")

        # Create genome
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

        # Initialize NEAT algorithm
        self.neat = TensorNEAT(
            genome=self.genome,
            pop_size=pop_size,
            species_size=species_size,
            survival_threshold=0.1,
            compatibility_threshold=1.0,
            species_elitism=2,
            max_stagnation=15,
            spawn_number_change_rate=0.5,
        )

        # Initialize state with randkey
        initial_key = jax.random.PRNGKey(seed)
        self.state = State(randkey=initial_key)
        self.state = self.neat.setup(self.state)

        # Store current population
        self.current_pop_nodes = None
        self.current_pop_conns = None

    def ask(self) -> jnp.ndarray:
        """Get current population parameters."""
        self.current_pop_nodes, self.current_pop_conns = self.neat.ask(self.state)
        return jnp.arange(self.pop_size)

    def tell(self, fitness: Union[np.ndarray, jnp.ndarray]) -> None:
        """Update algorithm with fitness results."""
        if isinstance(fitness, np.ndarray):
            fitness = jnp.array(fitness)

        self.state = self.neat.tell(self.state, fitness)

        # Update best params
        best_idx = jnp.argmax(fitness)
        if self._best_params is None or fitness[best_idx] > self.best_fitness:
            self.best_fitness = fitness[best_idx]
            self._best_params = (
                self.current_pop_nodes[best_idx],
                self.current_pop_conns[best_idx],
                self.state,
            )

    @property
    def best_params(self) -> jnp.ndarray:
        """Get best performing individual's parameters."""
        if self._best_params is None:
            raise ValueError("No best parameters available yet")
        return self._best_params

    @best_params.setter
    def best_params(self, params: Union[np.ndarray, jnp.ndarray]) -> None:
        """Set best parameters."""
        raise NotImplementedError("Setting best_params directly not supported in NEAT")

    def get_params(self, idx: int) -> tuple:
        """Get parameters for specific index."""
        return (
            self.current_pop_nodes[idx],
            self.current_pop_conns[idx],
            self.state,
        )
