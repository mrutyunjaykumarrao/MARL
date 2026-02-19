"""
Random Policy Evaluation
========================

Validates environment difficulty by running random actions.

If random policy achieves ~100% lambda2 reduction, the environment is trivial.

Usage:
    python evaluate_random.py --episodes 50
    python evaluate_random.py --episodes 100 --mode fast
"""

import argparse
import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from training.config import get_debug_config, get_fast_config, get_full_config
from environment.jammer_env import JammerEnv
from physics.fspl import compute_jam_range, db_to_watts
from physics.communication_graph import compute_adjacency_matrix


def count_edges(adj_matrix: np.ndarray) -> int:
    """Count number of edges in adjacency matrix."""
    # Adjacency matrix is symmetric, so count upper triangle
    return int(np.sum(np.triu(adj_matrix, k=1) > 0))


def compute_communication_range(P_tx_dbm: float, P_sens_dbm: float, freq: float) -> float:
    """Compute communication range based on FSPL."""
    c = 3e8
    wavelength = c / freq
    
    # FSPL: P_rx = P_tx - 20*log10(d) - 20*log10(f) + 147.55
    # At threshold: P_sens = P_tx - FSPL
    # Solve for d
    link_budget_db = P_tx_dbm - P_sens_dbm
    
    # FSPL = 20*log10(d) + 20*log10(f) - 147.55
    # link_budget = 20*log10(d) + 20*log10(f) - 147.55
    # 20*log10(d) = link_budget - 20*log10(f) + 147.55
    fspl_component = 20 * np.log10(freq) - 147.55
    log_d = (link_budget_db - fspl_component) / 20
    d = 10 ** log_d
    
    return d


def evaluate_random_policy(config, n_episodes: int = 50, verbose: bool = True):
    """
    Evaluate environment with random policy.
    
    Returns detailed statistics about environment difficulty.
    """
    # Create environment with config
    weights = config.env.reward_weights
    
    env = JammerEnv(
        N=config.env.N,
        M=config.env.M,
        arena_size=config.env.arena_size,
        v_max=config.env.v_max,
        max_steps=config.env.max_steps,
        eps=config.env.eps,
        min_samples=config.env.min_samples,
        v_enemy=config.env.v_enemy,
        enemy_mode='random_walk' if config.env.motion_mode == 'random' else config.env.motion_mode,
        K_recompute=config.env.k_recompute,
        P_tx_dbm=config.env.tx_power_dbm,
        P_sens_dbm=config.env.sensitivity_dbm,
        P_jammer_dbm=config.env.jammer_power_dbm,
        P_jam_thresh_dbm=getattr(config.env, 'jam_thresh_dbm', -40.0),
        omega_1=weights.get("lambda2_reduction", 1.0),
        omega_2=weights.get("band_match", 0.3),
        omega_3=weights.get("proximity", 0.2),
        omega_4=weights.get("energy", 0.1),
        omega_5=weights.get("overlap", 0.2),
        debug_mode=False,
        random_jammer_start=getattr(config.env, 'random_jammer_start', False)
    )
    
    # Compute theoretical values
    BANDS = [433e6, 915e6, 2.4e9, 5.8e9]
    comm_ranges = [compute_communication_range(config.env.tx_power_dbm, 
                                                config.env.sensitivity_dbm, f) 
                   for f in BANDS]
    
    jam_ranges = [compute_jam_range(db_to_watts(config.env.jammer_power_dbm),
                                    db_to_watts(getattr(config.env, 'jam_thresh_dbm', -40.0)),  # jam threshold
                                    f) for f in BANDS]
    
    print("=" * 70)
    print("RANDOM POLICY EVALUATION")
    print("=" * 70)
    print(f"\nEnvironment Configuration:")
    print(f"  N enemies: {config.env.N}")
    print(f"  M jammers: {config.env.M}")
    print(f"  Arena size: {config.env.arena_size}m x {config.env.arena_size}m")
    print(f"  Max steps: {config.env.max_steps}")
    print(f"  Random jammer start: {getattr(config.env, 'random_jammer_start', False)}")
    
    print(f"\nRF Parameters:")
    print(f"  TX power: {config.env.tx_power_dbm} dBm")
    print(f"  Sensitivity: {config.env.sensitivity_dbm} dBm")
    print(f"  Jammer power: {config.env.jammer_power_dbm} dBm")
    
    print(f"\nTheoretical Ranges (per frequency):")
    for i, (freq, comm_r, jam_r) in enumerate(zip(BANDS, comm_ranges, jam_ranges)):
        freq_label = ["433 MHz", "915 MHz", "2.4 GHz", "5.8 GHz"][i]
        print(f"  {freq_label}: Comm={comm_r:.1f}m, Jam={jam_r:.1f}m")
    
    print(f"\nRunning {n_episodes} episodes with RANDOM actions...")
    print("-" * 70)
    
    # Collect statistics
    stats = {
        'lambda2_initial': [],
        'lambda2_final': [],
        'lambda2_reduction': [],
        'edges_initial': [],
        'edges_final': [],
        'edges_disrupted': [],
        'episode_length': [],
        'total_reward': [],
        'steps_to_disconnect': [],  # -1 if never disconnected
        'r_jam': [],
        'n_clusters': [],
    }
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        
        lambda2_initial = info['lambda2_initial']
        
        # Count initial edges
        enemy_positions = env.enemy_swarm.positions
        enemy_freq = env.BANDS[env.enemy_band]
        adj_initial = compute_adjacency_matrix(
            enemy_positions,
            config.env.tx_power_dbm,
            config.env.sensitivity_dbm,
            enemy_freq
        )
        edges_initial = count_edges(adj_initial)
        
        # Episode loop with random actions
        done = False
        step = 0
        total_reward = 0
        lambda2_history = [lambda2_initial]
        first_disconnect_step = -1
        
        while not done:
            # RANDOM ACTION
            action = {
                'velocity': np.random.uniform(-env.v_max, env.v_max, size=(env.M, 2)),
                'band': np.random.randint(0, env.num_bands, size=env.M)
            }
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1
            total_reward += reward
            
            lambda2_current = info['lambda2_current']
            lambda2_history.append(lambda2_current)
            
            # Track first disconnect
            if lambda2_current <= 0 and first_disconnect_step == -1:
                first_disconnect_step = step
        
        # Final statistics
        lambda2_final = lambda2_history[-1]
        
        # Count final edges (with jamming)
        adj_final = compute_adjacency_matrix(
            env.enemy_swarm.positions,
            config.env.tx_power_dbm,
            config.env.sensitivity_dbm,
            env.BANDS[env.enemy_band]
        )
        edges_final = count_edges(adj_final)
        
        # Compute reduction
        if lambda2_initial > 1e-6:
            reduction = (1 - lambda2_final / lambda2_initial) * 100
        else:
            reduction = 0
        
        # Store stats
        stats['lambda2_initial'].append(lambda2_initial)
        stats['lambda2_final'].append(lambda2_final)
        stats['lambda2_reduction'].append(reduction)
        stats['edges_initial'].append(edges_initial)
        stats['edges_final'].append(edges_final)
        stats['edges_disrupted'].append(edges_initial - edges_final)
        stats['episode_length'].append(step)
        stats['total_reward'].append(total_reward)
        stats['steps_to_disconnect'].append(first_disconnect_step)
        stats['r_jam'].append(env.R_jam)
        stats['n_clusters'].append(info.get('n_clusters', 0))
        
        if verbose and (ep < 5 or ep == n_episodes - 1):
            print(f"  Episode {ep+1:3d}: L2_init={lambda2_initial:.2f}, "
                  f"L2_final={lambda2_final:.2f}, "
                  f"Reduction={reduction:.1f}%, "
                  f"Edges: {edges_initial}->{edges_final}, "
                  f"Steps={step}, "
                  f"Disconnect@{first_disconnect_step if first_disconnect_step > 0 else 'never'}")
        elif verbose and ep == 5:
            print("  ...")
    
    # Compute summary statistics
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    mean_reduction = np.mean(stats['lambda2_reduction'])
    std_reduction = np.std(stats['lambda2_reduction'])
    
    print(f"\nLambda-2 Statistics:")
    print(f"  Initial L2: {np.mean(stats['lambda2_initial']):.2f} +/- {np.std(stats['lambda2_initial']):.2f}")
    print(f"  Final L2:   {np.mean(stats['lambda2_final']):.2f} +/- {np.std(stats['lambda2_final']):.2f}")
    print(f"  Reduction:  {mean_reduction:.1f}% +/- {std_reduction:.1f}%")
    
    # Success rate at different thresholds
    print(f"\nSuccess Rates (random policy):")
    for threshold in [50, 70, 90, 100]:
        rate = np.mean(np.array(stats['lambda2_reduction']) >= threshold) * 100
        print(f"  >= {threshold}% reduction: {rate:.1f}% of episodes")
    
    print(f"\nEdge Statistics:")
    print(f"  Initial edges: {np.mean(stats['edges_initial']):.1f} +/- {np.std(stats['edges_initial']):.1f}")
    print(f"  Final edges:   {np.mean(stats['edges_final']):.1f} +/- {np.std(stats['edges_final']):.1f}")
    print(f"  Disrupted:     {np.mean(stats['edges_disrupted']):.1f} +/- {np.std(stats['edges_disrupted']):.1f}")
    
    print(f"\nEpisode Statistics:")
    print(f"  Average length: {np.mean(stats['episode_length']):.1f} steps")
    print(f"  Average reward: {np.mean(stats['total_reward']):.2f}")
    
    # Time to disconnect
    disconnect_steps = [s for s in stats['steps_to_disconnect'] if s > 0]
    never_disconnected = sum(1 for s in stats['steps_to_disconnect'] if s == -1)
    if disconnect_steps:
        print(f"  Mean steps to disconnect: {np.mean(disconnect_steps):.1f}")
    print(f"  Episodes never fully disconnected: {never_disconnected}/{n_episodes}")
    
    print(f"\nJamming Parameters:")
    print(f"  R_jam (mean): {np.mean(stats['r_jam']):.1f}m")
    print(f"  Clusters (mean): {np.mean(stats['n_clusters']):.1f}")
    
    # Diagnosis
    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)
    
    if mean_reduction >= 95:
        print("\n[!!!] ENVIRONMENT IS TRIVIAL")
        print("      Random policy achieves ~100% lambda2 reduction.")
        print("      The jammers are too powerful or the enemies too close.")
        print("\n      Recommended fixes:")
        print("      1. Reduce jammer power (current: {:.0f} dBm)".format(config.env.jammer_power_dbm))
        print("      2. Increase arena size (current: {:.0f}m)".format(config.env.arena_size))
        print("      3. Increase number of enemies (current: {})".format(config.env.N))
        print("      4. Reduce jam threshold sensitivity")
    elif mean_reduction >= 70:
        print("\n[!] ENVIRONMENT IS EASY")
        print("    Random policy achieves significant disruption.")
        print("    PPO should easily converge.")
    elif mean_reduction >= 30:
        print("\n[OK] ENVIRONMENT IS MODERATE")
        print("     Random policy has partial success.")
        print("     PPO should be able to learn meaningful policies.")
    else:
        print("\n[OK] ENVIRONMENT IS CHALLENGING")
        print("     Random policy struggles to disrupt swarm.")
        print("     This is a good learning environment for PPO.")
    
    # Check R_jam vs arena size
    avg_r_jam = np.mean(stats['r_jam'])
    if avg_r_jam > config.env.arena_size / 4:
        print(f"\n[WARNING] R_jam ({avg_r_jam:.1f}m) is > 25% of arena ({config.env.arena_size}m)")
        print("         One jammer can cover too much area!")
    
    # Check enemy density
    enemy_density = config.env.N / (config.env.arena_size ** 2)
    print(f"\n    Enemy density: {enemy_density * 10000:.2f} per 100m^2")
    
    env.close()
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Evaluate random policy baseline")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes")
    parser.add_argument("--mode", choices=["debug", "fast", "full"], default="fast",
                       help="Configuration preset")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()
    
    # Get config
    if args.mode == "debug":
        config = get_debug_config()
    elif args.mode == "fast":
        config = get_fast_config()
    else:
        config = get_full_config()
    
    print(f"\nUsing configuration: {args.mode}")
    
    stats = evaluate_random_policy(config, n_episodes=args.episodes, verbose=args.verbose)
    
    print("\n" + "=" * 70)
    print("Evaluation complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
