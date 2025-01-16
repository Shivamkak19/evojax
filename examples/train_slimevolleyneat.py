import argparse
import os
import shutil
import jax
import jax.numpy as jnp

from evojax.task.slimevolley import SlimeVolley
from evojax.policy.tensorneat import NEATPolicy
from evojax.algo.neat_wrapper import NEATWrapper
from evojax import Trainer
from evojax import util
from tensorneat.common.sympy_tools import to_latex_code, to_python_code


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pop-size", type=int, default=128, help="NEAT population size."
    )
    parser.add_argument(
        "--hidden-size", type=int, default=20, help="Initial hidden layer size."
    )
    parser.add_argument(
        "--max-nodes", type=int, default=50, help="Maximum nodes in NEAT network."
    )
    parser.add_argument(
        "--max-conns",
        type=int,
        default=200,
        help="Maximum connections in NEAT network.",
    )
    parser.add_argument(
        "--species-size", type=int, default=10, help="Number of species in NEAT."
    )
    parser.add_argument(
        "--num-tests", type=int, default=100, help="Number of test rollouts."
    )
    parser.add_argument(
        "--n-repeats", type=int, default=16, help="Training repetitions."
    )
    parser.add_argument(
        "--max-iter", type=int, default=500, help="Max training iterations."
    )
    parser.add_argument("--test-interval", type=int, default=50, help="Test interval.")
    parser.add_argument(
        "--log-interval", type=int, default=10, help="Logging interval."
    )
    parser.add_argument(
        "--seed", type=int, default=123, help="Random seed for training."
    )
    parser.add_argument("--gpu-id", type=str, help="GPU(s) to use.")
    parser.add_argument("--debug", action="store_true", help="Debug mode.")
    config, _ = parser.parse_known_args()
    return config


def main(config):
    log_dir = "./log/slimevolley_neat"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    logger = util.create_logger(name="SlimeVolley", log_dir=log_dir, debug=config.debug)
    logger.info("EvoJAX SlimeVolley with NEAT")
    logger.info("=" * 30)

    max_steps = 3000
    train_task = SlimeVolley(test=False, max_steps=max_steps)
    test_task = SlimeVolley(test=True, max_steps=max_steps)

    # Create NEAT policy
    policy = NEATPolicy(
        input_dim=train_task.obs_shape[0],
        output_dim=train_task.act_shape[0],
        hidden_size=config.hidden_size,
        max_nodes=config.max_nodes,
        max_conns=config.max_conns,
        logger=logger,
    )

    # Create NEAT solver
    solver = NEATWrapper(
        input_dim=train_task.obs_shape[0],
        output_dim=train_task.act_shape[0],
        pop_size=config.pop_size,
        hidden_size=config.hidden_size,
        species_size=config.species_size,
        max_nodes=config.max_nodes,
        max_conns=config.max_conns,
        seed=config.seed,
    )

    # Update trainer initialization
    trainer = Trainer(
        policy=policy,
        solver=solver,
        train_task=train_task,
        test_task=test_task,
        max_iter=config.max_iter,
        log_interval=config.log_interval,
        test_interval=config.test_interval,
        n_repeats=config.n_repeats,
        n_evaluations=config.num_tests,
        seed=config.seed,
        log_dir=log_dir,
        logger=logger,
    )

    # Make sure policy and solver are connected
    solver.set_policy(policy)

    # Train
    trainer.run(demo_mode=False)

    # Save best model
    src_file = os.path.join(log_dir, "best.npz")
    tar_file = os.path.join(log_dir, "model.npz")
    shutil.copy(src_file, tar_file)
    trainer.model_dir = log_dir
    trainer.run(demo_mode=True)

    # Visualize final policy
    task_reset_fn = jax.jit(test_task.reset)
    policy_reset_fn = jax.jit(policy.reset)
    step_fn = jax.jit(test_task.step)
    action_fn = jax.jit(policy.get_actions)

    # Get best parameters
    best_params = solver.best_params  # This is a tuple of (nodes, conns)
    policy.state = solver.state  # Ensure policy has latest state

    # Get the network representation
    network = policy.genome.network_dict(policy.state, *best_params)

    # 1. Visualize network topology as SVG
    policy.genome.visualize(network, save_path="slimevolley_network.svg")

    # 2. Print network representation as string
    print("Network structure:")
    print(policy.genome.repr(policy.state, *best_params))

    # 3. Optional: Get mathematical formula representation
    sympy_res = policy.genome.sympy_func(
        policy.state, network, sympy_output_transform=policy.genome.output_transform
    )
    latex_code = to_latex_code(*sympy_res)
    print("\nNetwork as LaTeX formula:")
    print(latex_code)

    # Setup states
    key = jax.random.PRNGKey(0)
    key = jnp.expand_dims(key, 0)
    task_state = task_reset_fn(key)
    policy_state = policy_reset_fn(task_state)

    # Run visualization
    screens = []
    for _ in range(max_steps):
        action, policy_state = action_fn(task_state, best_params, policy_state)
        task_state, reward, done = step_fn(task_state, action)
        screens.append(SlimeVolley.render(task_state))

    # Save visualization
    gif_file = os.path.join(log_dir, "slimevolley_neat.gif")
    screens[0].save(
        gif_file, save_all=True, append_images=screens[1:], duration=40, loop=0
    )
    logger.info("GIF saved to {}.".format(gif_file))


if __name__ == "__main__":
    configs = parse_args()
    if configs.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = configs.gpu_id
    main(configs)
