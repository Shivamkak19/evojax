import wandb
import argparse
import os
import shutil
import jax
import jax.numpy as jnp
from datetime import datetime
import uuid
from evojax.task.slimevolley import SlimeVolley
from evojax.policy.tensorneat import NEATPolicy
from evojax.algo.neat_wrapper import NEATWrapper
from evojax import Trainer
from evojax import util


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
        default=2000,
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
    parser.add_argument(
        "--wandb-project", type=str, default="slimevolley-neat", help="W&B project name"
    )
    parser.add_argument("--wandb-entity", type=str, help="W&B entity/username")
    # Generate a unique run name with date/time and 6-digit UUID
    default_run_name = (
        f"neat-{datetime.now().strftime('%Y%m%d_%H%M%S')}-{str(uuid.uuid4())[:6]}"
    )
    parser.add_argument(
        "--wandb-name", type=str, default=default_run_name, help="W&B run name"
    )
    config, _ = parser.parse_known_args()
    return config


def network_to_dict(genome, nodes, conns, state):
    """Convert network structure to a dictionary format for visualization"""
    network = genome.network_dict(state, nodes, conns)
    
    # Get all node information including activation functions
    node_info = {}
    for node_id, node_data in network['nodes'].items():
        node_type = 'hidden'
        if node_id in genome.get_input_idx():
            node_type = 'input'
        elif node_id in genome.get_output_idx():
            node_type = 'output'
            
        node_info[node_id] = {
            'type': node_type,
            'activation': node_data['act'],
            'bias': float(node_data['bias'])
        }
    
    # Get connection information
    connections = []
    for (source, target), conn_data in network['conns'].items():
        connections.append({
            'source': int(source),
            'target': int(target),
            'weight': float(conn_data['weight'])
        })
        
    return {
        'nodes': node_info,
        'connections': connections,
        'layers': network['topo_layers']
    }

def visualize_network(trainer, policy, solver, wandb_run=None, iteration=None):
    """Generate and log network visualization"""
    if wandb_run is None:
        return
    
    # Get best network structure
    idx = solver.best_params[0]
    nodes = solver.current_nodes[idx]
    conns = solver.current_conns[idx]
    
    # Convert to visualization format
    network_data = network_to_dict(policy.genome, nodes, conns, policy.state)
    
    # Log network structure metrics
    metrics = {
        'network/num_nodes': len(network_data['nodes']),
        'network/num_connections': len(network_data['connections']),
        'network/num_layers': len(network_data['layers'])
    }
    
    # Count activation functions
    act_funcs = {}
    for node in network_data['nodes'].values():
        act = node['activation']
        act_funcs[act] = act_funcs.get(act, 0) + 1
    
    for act, count in act_funcs.items():
        metrics[f'network/activation_{act}'] = count
    
    # Log to wandb
    wandb_run.log(metrics, step=iteration)
    
    # Save network visualization
    try:
        # Create network visualization
        network = policy.genome.visualize(
            network_data,
            save_path=None,  # Return figure instead of saving
            size=(600, 400)
        )
        wandb_run.log({"network_visualization": wandb.Image(network)}, step=iteration)
    except Exception as e:
        print(f"Warning: Failed to log network visualization: {str(e)}")


def get_wandb_logging_function(wandb_run, policy, solver):
    """Creates a combined logging function for scores and network visualization."""
    
    def log_network_info(iteration: int, stage: str):
        """Helper function to log network visualization and metrics"""
        try:
            # print("GATE 0")
            # Get best network structure
            idx = solver.best_params[0]
            nodes = solver.current_nodes[idx]
            conns = solver.current_conns[idx]
            
            # Convert to visualization format
            network = policy.genome.network_dict(policy.state, nodes, conns)
            
            # Log network metrics
            metrics = {
                f'{stage}/network/num_nodes': len(network['nodes']),
                f'{stage}/network/num_connections': len(network['conns']),
                f'{stage}/network/num_layers': len(network['topo_layers'])
            }
            
            # Count activation functions
            act_funcs = {}
            for node_id, node_data in network['nodes'].items():
                act = node_data['act'] 
                act_funcs[act] = act_funcs.get(act, 0) + 1
            
            for act, count in act_funcs.items():
                metrics[f'{stage}/network/activation_{act}'] = count
            
            wandb_run.log(metrics)
            
            # Create and log network visualization
            try:
                # print("GATE 1")

                # Use tensorneat's built-in visualization
                policy.genome.visualize(
                    network,
                    save_path=f"temp_network_{stage}_{iteration}.png",
                    with_labels=True,
                    # size=(800, 600)
                )
                
                # Log to wandb with stage-specific name
                wandb_run.log({
                    f"{stage}/network_graph": wandb.Image(f"temp_network_{stage}_{iteration}.png")
                })
                # print("GATE 2")

                # Cleanup temporary file
                import os
                # os.remove(f"temp_network_{stage}_{iteration}.png")
                
            except Exception as e:
                print(f"Warning: Failed to create network visualization for {stage}: {str(e)}")
                
        except Exception as e:
            print(f"Warning: Failed to log network information for {stage}: {str(e)}")

    def log_combined(iteration: int, scores: jnp.ndarray, stage: str):
        if wandb_run is not None:
            # Original score logging

            # print("GATE -2")
            wandb_run.log({
                "iteration": iteration,
                f"{stage}/score_min": float(scores.min()),
                f"{stage}/score_max": float(scores.max()),
                f"{stage}/score_mean": float(scores.mean()),
                f"{stage}/score_std": float(scores.std()),
            })

            # print("GATE -1")
            
            # Log network info for both train and test
            log_network_info(iteration, stage)

    return log_combined

def save_visualization(
    task_state, policy, solver, max_steps, log_dir, wandb_run=None, iteration=None
):
    # print("VIDEO GATE 0")
    """Generates and saves visualization of the trained policy.

    Args:
        task_state: The SlimeVolley task state
        policy: The NEAT policy
        solver: The NEAT solver
        max_steps: Maximum steps to run
        log_dir: Directory to save the GIF
        wandb_run: Optional wandb run object for logging
        iteration: Optional iteration number for the filename
    """
    task_reset_fn = jax.jit(task_state.reset)
    policy_reset_fn = jax.jit(policy.reset)
    step_fn = jax.jit(task_state.step)
    action_fn = jax.jit(policy.get_actions)

    key = jax.random.PRNGKey(0)
    key = jnp.expand_dims(key, 0)
    task_state = task_reset_fn(key)
    policy_state = policy_reset_fn(task_state)

    # Record episode statistics
    total_reward = 0
    steps_survived = 0

    screens = []
    for step in range(max_steps):
        action, policy_state = action_fn(task_state, solver.best_params, policy_state)
        task_state, reward, done = step_fn(task_state, action)
        screens.append(SlimeVolley.render(task_state))

        total_reward += float(reward.item())
        steps_survived = step + 1
        if done:
            break

    # print("VIDEO GATE 1")
    # Create descriptive filename with metrics
    iter_str = f"_iter{iteration}" if iteration is not None else ""
    gif_name = f"slimevolley{iter_str}_r{total_reward:.1f}_s{steps_survived}.gif"
    gif_file = os.path.join(log_dir, gif_name)

    screens[0].save(
        gif_file, save_all=True, append_images=screens[1:], duration=40, loop=0
    )

    # Log to wandb if available
    if wandb_run is not None:
        try:
            # print("VIDEO GATE 2")
            # Log the GIF with metrics
            wandb_run.log(
                {
                    "game_visualization": wandb.Video(gif_file, fps=25, format="gif"),
                    "visualization_reward": total_reward,
                    "visualization_steps": steps_survived,
                },
                step=iteration,
            )
        except Exception as e:
            print(f"Warning: Failed to log visualization to W&B: {str(e)}")

    return gif_file


def main(config):
    # Initialize W&B with error handling
    try:
        run = wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=config.wandb_name,
            config=vars(config),
        )
    except Exception as e:
        print(f"Warning: Failed to initialize W&B: {str(e)}")
        print("Continuing without W&B logging...")
        run = None

    # Set up logging directory
    log_dir = "./log/slimevolley_neat"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    logger = util.create_logger(name="SlimeVolley", log_dir=log_dir, debug=config.debug)
    logger.info("EvoJAX SlimeVolley with NEAT")
    logger.info("=" * 30)

    # Initialize tasks
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

    # Create trainer with wandb logging
    trainer = Trainer(
        policy=policy,
        solver=solver,
        train_task=train_task,
        test_task=test_task,
        max_iter=config.max_iter,
        log_interval=config.log_interval,
        test_interval=config.test_interval,
        n_repeats=config.n_repeats,
        test_n_repeats=1,
        n_evaluations=config.num_tests,
        seed=config.seed,
        log_dir=log_dir,
        logger=logger,
        # Pass both policy and solver to the logging function
        log_scores_fn=get_wandb_logging_function(run, policy, solver),
    )

    # Connect policy and solver
    solver.set_policy(policy)

    # Train
    best_score = trainer.run(demo_mode=False)

    # Log final results to wandb
    if run is not None:
        run.log({"best_final_score": float(best_score)})

        # Save model to wandb
        try:
            src_file = os.path.join(log_dir, "best.npz")
            artifact = wandb.Artifact("model", type="model")
            artifact.add_file(src_file)
            run.log_artifact(artifact)
        except Exception as e:
            print(f"Warning: Failed to log model artifact to W&B: {str(e)}")

    # Save best model
    src_file = os.path.join(log_dir, "best.npz")
    tar_file = os.path.join(log_dir, "model.npz")
    shutil.copy(src_file, tar_file)

    # Run demo and save visualization
    logger.info("Running demo with best model...")
    trainer.model_dir = log_dir
    trainer.run(demo_mode=True)

    gif_file = save_visualization(test_task, policy, solver, max_steps, log_dir)

    # print("MAIN GATE 1")
    logger.info(f"GIF saved to {gif_file}")

    # Log visualization to wandb
    if run is not None:
        try:
            # print("MAIN GATE 2")
            wandb.log({"visualization": wandb.Image(gif_file)})
            run.finish()
            # print("MAIN GATE 3")
        except Exception as e:
            print(f"Warning: Failed to log visualization to W&B: {str(e)}")


if __name__ == "__main__":
    configs = parse_args()
    if configs.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = configs.gpu_id
    main(configs)