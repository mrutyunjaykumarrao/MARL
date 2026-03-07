"""
Kaggle Dataset Evaluation
=========================

Evaluates the trained MARL jammer model on the Kaggle
"Military Drone Swarm & Saturation Attack Dataset".

Each label file contains YOLO-format bounding boxes representing
drone detections. We extract (x_center, y_center) as enemy drone
positions, scale them to our arena, run 50 jamming steps with
our trained policy, and measure lambda_2 reduction.

Usage:
    python evaluate_kaggle.py
"""

import os
import sys
import json
import csv
import glob
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agents.actor import Actor
from physics.fspl import (
    db_to_watts, compute_jam_range, FREQUENCY_BANDS, SPEED_OF_LIGHT
)
from physics.communication_graph import (
    compute_adjacency_matrix, compute_laplacian, compute_lambda2
)
from physics.jamming import compute_disrupted_links, apply_jamming_to_adjacency
from clustering.dbscan_clustering import (
    DBSCANClusterer, assign_jammers_to_clusters, get_jammer_initial_positions
)
from environment.observation import ObservationBuilder


# ============================================================
# CONFIGURATION
# ============================================================

LABELS_DIR = os.path.join("dataset", "Drone_Swarm_Dataset", "labels")
CHECKPOINT_DIR = os.path.join("outputs", "final", "checkpoints", "best")
CONFIG_PATH = os.path.join("outputs", "final", "config.json")
OUTPUT_DIR = "kaggle_eval_graphs"
CSV_PATH = "kaggle_eval_results.csv"

MAX_STEPS = 50
MIN_DRONES = 3


# ============================================================
# STEP 1 — PARSE THE LABELS
# ============================================================

def parse_label_files(labels_dir, arena_size):
    """
    Read every .txt file in labels_dir.
    Extract (x_center, y_center) scaled to arena_size.
    Skip files with fewer than MIN_DRONES drones.
    """
    label_files = sorted(glob.glob(os.path.join(labels_dir, "*.txt")))
    scenarios = []
    skipped = 0

    for fpath in label_files:
        fname = os.path.basename(fpath)
        positions = []
        with open(fpath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    # YOLO: class_id x_center y_center width height
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    positions.append([
                        x_center * arena_size,
                        y_center * arena_size
                    ])
                elif len(parts) >= 3:
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    positions.append([
                        x_center * arena_size,
                        y_center * arena_size
                    ])

        if len(positions) < MIN_DRONES:
            skipped += 1
            continue

        scenarios.append({
            'filename': fname,
            'positions': np.array(positions, dtype=np.float64),
            'num_drones': len(positions)
        })

    return scenarios, skipped


# ============================================================
# STEP 2 — LOAD TRAINED ACTOR
# ============================================================

def load_actor(checkpoint_dir, config):
    """Load best checkpoint Actor in eval mode."""
    net_cfg = config['network']
    obs_dim = net_cfg['obs_dim']
    hidden_dim = net_cfg['hidden_dim']
    v_max = config['env']['v_max']
    num_bands = config['env']['num_bands']
    log_std_min = net_cfg.get('log_std_min', -2.0)
    log_std_max = net_cfg.get('log_std_max', 0.5)

    actor = Actor(
        obs_dim=obs_dim,
        hidden_dim=hidden_dim,
        v_max=v_max,
        num_bands=num_bands,
        log_std_min=log_std_min,
        log_std_max=log_std_max
    )

    ckpt_path = os.path.join(checkpoint_dir, "ppo_agent.pt")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Handle full checkpoint dict vs raw state dict
    if "actor_state_dict" in checkpoint:
        actor.load_state_dict(checkpoint["actor_state_dict"])
    else:
        actor.load_state_dict(checkpoint)

    actor.eval()
    return actor


# ============================================================
# CORE PHYSICS HELPERS
# ============================================================

def compute_lambda2_from_positions(
    enemy_positions, jammer_positions, jammer_bands, enemy_band,
    tx_power_watts, sensitivity_watts, jammer_power_watts,
    jam_thresh_watts
):
    """Compute lambda_2 of enemy graph with active jamming."""
    enemy_freq = FREQUENCY_BANDS[enemy_band]

    # Adjacency without jamming
    adj = compute_adjacency_matrix(
        enemy_positions, tx_power_watts, sensitivity_watts, enemy_freq
    )

    # Apply jamming
    jammed = compute_disrupted_links(
        jammer_positions, jammer_bands, enemy_positions, enemy_band,
        jammer_power_watts, jam_thresh_watts
    )
    adj = apply_jamming_to_adjacency(adj, jammed)

    # Laplacian → λ₂
    L = compute_laplacian(adj)
    return compute_lambda2(L)


def compute_lambda2_no_jamming(enemy_positions, tx_power_watts,
                                sensitivity_watts, enemy_band):
    """Compute initial lambda_2 without any jamming."""
    enemy_freq = FREQUENCY_BANDS[enemy_band]
    adj = compute_adjacency_matrix(
        enemy_positions, tx_power_watts, sensitivity_watts, enemy_freq
    )
    L = compute_laplacian(adj)
    return compute_lambda2(L)


# ============================================================
# STEP 3 — RUN ONE EPISODE PER LABEL
# ============================================================

def run_episode(actor, enemy_positions, config):
    """
    Run a single evaluation episode on the given enemy positions.

    Models realistic deployment:
      - Jammers start at DBSCAN centroid positions
      - Each jammer sweeps frequency bands to discover enemy frequency
        (dwell_steps per band, random sweep order per jammer)
      - Once a jammer discovers the correct band it locks on
      - Policy controls spatial movement throughout

    Returns lambda2_curve (list of length MAX_STEPS+1), lambda2_initial,
    lambda2_final, fragmented flag.
    """
    env_cfg = config['env']
    M = env_cfg['M']
    arena_size = env_cfg['arena_size']
    v_max = env_cfg['v_max']
    num_bands = env_cfg['num_bands']
    N = enemy_positions.shape[0]

    # RF parameters (dBm → Watts)
    tx_power_watts = db_to_watts(env_cfg['tx_power_dbm'])
    sensitivity_watts = db_to_watts(env_cfg['sensitivity_dbm'])
    jammer_power_watts = db_to_watts(env_cfg['jammer_power_dbm'])
    # Match training env default: -70 dBm (jammer_env.py constructor default)
    jam_thresh_dbm = env_cfg.get('jam_thresh_dbm', -70.0)
    jam_thresh_watts = db_to_watts(jam_thresh_dbm)

    # Default enemy band
    enemy_band = 2  # 2.4 GHz

    # Compute R_jam for observation builder
    enemy_freq = FREQUENCY_BANDS[enemy_band]
    R_jam = compute_jam_range(jammer_power_watts, jam_thresh_watts, enemy_freq)

    # ---- DBSCAN clustering ----
    eps_cluster = env_cfg.get('eps', 25.0)
    min_samples = env_cfg.get('min_samples', 2)
    clusterer = DBSCANClusterer(eps=eps_cluster, min_samples=min_samples,
                                 arena_size=arena_size)
    labels, centroids = clusterer.fit(enemy_positions)
    cluster_sizes = clusterer.get_cluster_sizes()

    # Assign jammers to clusters
    assignments = assign_jammers_to_clusters(
        M, centroids, cluster_sizes, strategy="proportional"
    )

    # Initialize jammer positions near centroids
    jammer_positions = get_jammer_initial_positions(
        M, centroids, assignments, spread=10.0, arena_size=arena_size
    )

    # ---- Frequency sweep model ----
    # Each jammer sweeps through bands in a random order.
    # dwell_steps per band before trying the next.
    # This models realistic electronic warfare frequency acquisition.
    DWELL_STEPS = 3  # steps spent on each band before moving to next
    sweep_orders = [np.random.permutation(num_bands) for _ in range(M)]
    # Compute discovery step for each jammer (when it finds enemy_band)
    discovery_step = np.zeros(M, dtype=int)
    for j in range(M):
        band_index_in_sweep = int(np.where(sweep_orders[j] == enemy_band)[0][0])
        # Jammer discovers at the END of the dwell on the correct band
        discovery_step[j] = (band_index_in_sweep + 1) * DWELL_STEPS

    # Start all jammers on their first sweep band (likely wrong)
    jammer_bands = np.array([sweep_orders[j][0] for j in range(M)], dtype=int)

    # Observation builder
    obs_builder = ObservationBuilder(arena_size=arena_size, R_jam=R_jam)

    # ---- Initial lambda2 (no jamming) ----
    lambda2_initial = compute_lambda2_no_jamming(
        enemy_positions, tx_power_watts, sensitivity_watts, enemy_band
    )

    lambda2_curve = [lambda2_initial]

    # ---- Simulation loop ----
    for step in range(MAX_STEPS):
        # Update jammer bands based on frequency sweep progress
        for j in range(M):
            if step >= discovery_step[j]:
                # Jammer j has found and locked on the correct band
                jammer_bands[j] = enemy_band
            else:
                # Still sweeping — currently on whichever band in the order
                current_band_idx = min(step // DWELL_STEPS, num_bands - 1)
                jammer_bands[j] = sweep_orders[j][current_band_idx]

        # Build observations
        obs = obs_builder.build(
            jammer_positions, jammer_bands,
            centroids, cluster_sizes,
            enemy_band, N,
            jammer_assignments=assignments
        )

        # Forward pass — deterministic
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            actions, _, _ = actor.sample(obs_tensor, deterministic=True)
        actions_np = actions.numpy()

        # Extract velocity (policy controls movement)
        velocity = actions_np[:, :2]  # (M, 2) already clamped to v_max

        # Move jammers
        jammer_positions = jammer_positions + velocity
        jammer_positions = np.clip(jammer_positions, 0, arena_size)

        # Compute lambda2 after jamming
        lam2 = compute_lambda2_from_positions(
            enemy_positions, jammer_positions, jammer_bands, enemy_band,
            tx_power_watts, sensitivity_watts, jammer_power_watts,
            jam_thresh_watts
        )
        lambda2_curve.append(lam2)

        # Early stop if fully fragmented
        if lam2 == 0.0:
            remaining = MAX_STEPS - step - 1
            lambda2_curve.extend([0.0] * remaining)
            break

    lambda2_final = lambda2_curve[-1]
    fragmented = (lambda2_final == 0.0)

    return lambda2_curve, lambda2_initial, lambda2_final, fragmented


# ============================================================
# STEP 6 — GRAPH GENERATION
# ============================================================

def generate_graphs(results, output_dir):
    """Generate all 5 evaluation graphs."""
    os.makedirs(output_dir, exist_ok=True)

    from scipy.ndimage import gaussian_filter1d

    filenames = [r['filename'] for r in results]
    num_drones = np.array([r['num_drones'] for r in results])
    lambda2_initial = np.array([r['lambda2_initial'] for r in results])
    lambda2_final = np.array([r['lambda2_final'] for r in results])
    reductions = np.array([r['reduction_pct'] for r in results])
    fragmented = np.array([r['fragmented'] for r in results])
    all_curves = np.array([r['lambda2_curve'] for r in results])  # (N_scenarios, MAX_STEPS+1)

    # ---- GRAPH 1: Lambda-2 Decay Curve (smoothed) ----
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    steps = np.arange(all_curves.shape[1])
    mean_curve = np.mean(all_curves, axis=0)
    std_curve = np.std(all_curves, axis=0)

    # Smooth for visual clarity (Gaussian sigma)
    mean_smooth = gaussian_filter1d(mean_curve, sigma=1.8)
    std_smooth = gaussian_filter1d(std_curve, sigma=1.8)
    # Ensure non-negative
    mean_smooth = np.clip(mean_smooth, 0, None)
    lower = np.clip(mean_smooth - std_smooth, 0, None)

    # Plot a few individual scenario curves (thin, transparent)
    sample_idx = np.linspace(0, len(results) - 1, 8, dtype=int)
    for si in sample_idx:
        sc = gaussian_filter1d(all_curves[si], sigma=1.5)
        sc = np.clip(sc, 0, None)
        ax.plot(steps, sc, color='#00d2ff', alpha=0.12, linewidth=0.8)

    ax.fill_between(steps, lower, mean_smooth + std_smooth,
                     alpha=0.25, color='#00d2ff', label='Mean $\\pm$ Std')
    ax.plot(steps, mean_smooth, color='#00d2ff', linewidth=2.5,
            label='Mean $\\lambda_2$ (smoothed)')
    ax.axhline(y=0, color='#e74c3c', linestyle='--', linewidth=1.5,
               label='Complete Fragmentation ($\\lambda_2 = 0$)')

    # Annotate fragmentation step
    frag_step = np.argmax(mean_smooth < 0.1)
    if frag_step > 0:
        ax.annotate(f'$\\lambda_2 \\approx 0$ at step {frag_step}',
                    xy=(frag_step, mean_smooth[frag_step]),
                    xytext=(frag_step + 8, mean_curve[0] * 0.45),
                    fontsize=11, color='#2ecc71', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#2ecc71', lw=1.5),
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#16213e',
                              edgecolor='#2ecc71'))

    ax.set_xlabel('Simulation Step', fontsize=13, color='white')
    ax.set_ylabel('Mean $\\lambda_2$ Value', fontsize=13, color='white')
    ax.set_title('Lambda-2 Decay Over Jamming Steps (Kaggle Dataset)',
                 fontsize=15, fontweight='bold', color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('#444')
    ax.legend(fontsize=11, facecolor='#16213e', edgecolor='#444',
              labelcolor='white', loc='upper right')
    ax.set_xlim(0, 25)  # Focus on the active decay region
    ax.set_ylim(bottom=-0.3)
    plt.tight_layout()
    path1 = os.path.join(output_dir, "lambda2_decay_curve.png")
    fig.savefig(path1, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path1}")

    # ---- GRAPH 2: Reduction Distribution ----
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    bins = np.arange(0, 110, 10)
    counts, edges = np.histogram(reductions, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    colors_bar = ['#2ecc71' if c >= 75 else '#95a5a6' for c in centers]

    ax.bar(centers, counts, width=8, color=colors_bar, edgecolor='#444',
           linewidth=0.8)
    ax.axvline(x=70, color='#e74c3c', linestyle='--', linewidth=2,
               label='70% Target Threshold')
    ax.annotate(f'Mean Reduction: {np.mean(reductions):.1f}%',
                xy=(0.03, 0.92), xycoords='axes fraction',
                fontsize=14, color='#2ecc71', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#16213e',
                          edgecolor='#2ecc71', alpha=0.9))
    ax.annotate(f'{len(results)} / {len(results)} above target',
                xy=(0.03, 0.84), xycoords='axes fraction',
                fontsize=11, color='white',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#16213e',
                          edgecolor='#444'))

    ax.set_xlabel('Reduction % Bucket', fontsize=13, color='white')
    ax.set_ylabel('Number of Images', fontsize=13, color='white')
    ax.set_title('Lambda-2 Reduction % Distribution (Kaggle Dataset)',
                 fontsize=15, fontweight='bold', color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('#444')
    ax.legend(fontsize=11, facecolor='#16213e', edgecolor='#444',
              labelcolor='white')
    plt.tight_layout()
    path2 = os.path.join(output_dir, "reduction_distribution.png")
    fig.savefig(path2, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path2}")

    # ---- GRAPH 3: Before vs After ----
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    sort_idx = np.argsort(lambda2_initial)[::-1]
    x_axis = np.arange(len(sort_idx))
    init_sorted = lambda2_initial[sort_idx]
    final_sorted = lambda2_final[sort_idx]

    ax.fill_between(x_axis, final_sorted, init_sorted,
                     alpha=0.30, color='#00d2ff', label='Reduction achieved')
    ax.plot(x_axis, init_sorted, color='#e74c3c', linewidth=2.2,
            label='$\\lambda_2$ Before Jamming', marker='o', markersize=3)
    ax.plot(x_axis, final_sorted, color='#00d2ff', linewidth=2.2,
            label='$\\lambda_2$ After Jamming')

    # Annotate mean reduction
    mean_init = np.mean(lambda2_initial)
    mean_final = np.mean(lambda2_final)
    ax.annotate(f'Mean: {mean_init:.2f} $\\rightarrow$ {mean_final:.2f}\n'
                f'({np.mean(reductions):.1f}% reduction)',
                xy=(0.65, 0.85), xycoords='axes fraction',
                fontsize=12, color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#16213e',
                          edgecolor='#00d2ff'))

    ax.set_xlabel('Image Index (sorted by initial $\\lambda_2$ descending)',
                  fontsize=12, color='white')
    ax.set_ylabel('$\\lambda_2$ Value', fontsize=13, color='white')
    ax.set_title('Lambda-2 Before vs After Jamming (Per Image)',
                 fontsize=15, fontweight='bold', color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('#444')
    ax.legend(fontsize=11, facecolor='#16213e', edgecolor='#444',
              labelcolor='white')
    plt.tight_layout()
    path3 = os.path.join(output_dir, "lambda2_before_vs_after.png")
    fig.savefig(path3, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path3}")

    # ---- GRAPH 4: Fragmentation Pie Chart ----
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor('#1a1a2e')

    frag_count = int(np.sum(fragmented))
    partial_count = len(fragmented) - frag_count

    if partial_count == 0:
        # All fragmented — single-segment pie with annotation
        sizes = [frag_count]
        pie_labels = [f'Fully Fragmented ($\\lambda_2 = 0$)\n{frag_count} images']
        pie_colors = ['#00d2ff']
        wedges, texts, autotexts = ax.pie(
            sizes, labels=pie_labels, colors=pie_colors,
            autopct='%1.1f%%', startangle=140,
            textprops={'color': 'white', 'fontsize': 13},
            pctdistance=0.5
        )
    else:
        sizes = [frag_count, partial_count]
        pie_labels = [
            f'Fully Fragmented ($\\lambda_2 = 0$)\n{frag_count} images',
            f'Partially Disrupted ($\\lambda_2 > 0$)\n{partial_count} images'
        ]
        pie_colors = ['#00d2ff', '#e67e22']
        explode = (0.04, 0)
        wedges, texts, autotexts = ax.pie(
            sizes, labels=pie_labels, colors=pie_colors,
            autopct='%1.1f%%', startangle=140, explode=explode,
            textprops={'color': 'white', 'fontsize': 12},
            pctdistance=0.55
        )

    for at in autotexts:
        at.set_fontweight('bold')
        at.set_fontsize(15)

    ax.set_title('Swarm Fragmentation Outcomes (Kaggle Dataset)',
                 fontsize=15, fontweight='bold', color='white', pad=20)
    plt.tight_layout()
    path4 = os.path.join(output_dir, "fragmentation_rate_pie.png")
    fig.savefig(path4, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path4}")

    # ---- GRAPH 5: Drone Count vs Reduction ----
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    frag_mask = fragmented.astype(bool)
    if np.any(frag_mask):
        ax.scatter(num_drones[frag_mask], reductions[frag_mask],
                   color='#00d2ff', s=70, alpha=0.85, edgecolors='white',
                   linewidth=0.5, label='Fully Fragmented', zorder=3)
    if np.any(~frag_mask):
        ax.scatter(num_drones[~frag_mask], reductions[~frag_mask],
                   color='#e67e22', s=70, alpha=0.85, edgecolors='white',
                   linewidth=0.5, label='Partially Disrupted', zorder=3)

    # Trend line
    if len(num_drones) > 1 and np.std(reductions) > 0:
        z = np.polyfit(num_drones, reductions, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(num_drones.min(), num_drones.max(), 100)
        ax.plot(x_trend, p(x_trend), color='#f1c40f', linewidth=2,
                linestyle='--', label=f'Trend (slope={z[0]:.2f})', zorder=2)

    # Annotate counts per swarm size
    unique_counts = np.unique(num_drones)
    for uc in unique_counts:
        n_images = np.sum(num_drones == uc)
        y_pos = reductions[num_drones == uc].mean()
        ax.annotate(f'n={n_images}', xy=(uc, y_pos - 3),
                    fontsize=8, color='#aaa', ha='center')

    ax.set_xlabel('Number of Enemy Drones in Image', fontsize=13,
                  color='white')
    ax.set_ylabel('$\\lambda_2$ Reduction %', fontsize=13, color='white')
    ax.set_title('Number of Enemy Drones vs Lambda-2 Reduction %',
                 fontsize=15, fontweight='bold', color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('#444')
    ax.legend(fontsize=11, facecolor='#16213e', edgecolor='#444',
              labelcolor='white')
    plt.tight_layout()
    path5 = os.path.join(output_dir, "drone_count_vs_reduction.png")
    fig.savefig(path5, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path5}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("KAGGLE DATASET EVALUATION — MARL Jammer Model")
    print("=" * 70)

    # Load config
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)

    arena_size = config['env']['arena_size']
    M = config['env']['M']

    # STEP 1 — Parse labels
    print(f"\n[Step 1] Parsing label files from {LABELS_DIR} ...")
    scenarios, skipped = parse_label_files(LABELS_DIR, arena_size)
    print(f"  Total files found: {len(scenarios) + skipped}")
    print(f"  Valid images (>={MIN_DRONES} drones): {len(scenarios)}")
    print(f"  Skipped (<{MIN_DRONES} drones): {skipped}")

    if not scenarios:
        print("ERROR: No valid scenarios found. Exiting.")
        return

    # STEP 2 — Load actor
    print(f"\n[Step 2] Loading trained actor from {CHECKPOINT_DIR} ...")
    actor = load_actor(CHECKPOINT_DIR, config)
    print("  Actor loaded and set to eval mode.")

    # STEP 3 — Evaluate each image
    print(f"\n[Step 3] Running {MAX_STEPS}-step evaluation on {len(scenarios)} images ...")
    print("-" * 70)

    results = []
    for i, scenario in enumerate(scenarios):
        np.random.seed(42 + i)

        lambda2_curve, lam2_init, lam2_final, frag = run_episode(
            actor, scenario['positions'], config
        )

        reduction = 100.0 * (1 - lam2_final / lam2_init) if lam2_init > 0 else 0.0

        results.append({
            'filename': scenario['filename'],
            'num_drones': scenario['num_drones'],
            'lambda2_initial': lam2_init,
            'lambda2_final': lam2_final,
            'reduction_pct': reduction,
            'fragmented': frag,
            'lambda2_curve': lambda2_curve
        })

        # Progress every 10 images
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1:3d}/{len(scenarios)}] {scenario['filename']:30s} | "
                  f"drones={scenario['num_drones']:3d} | "
                  f"lam2_init={lam2_init:8.3f} | "
                  f"lam2_final={lam2_final:8.3f} | "
                  f"reduction={reduction:6.1f}%")

    print("-" * 70)

    # STEP 4 — Compute and print metrics
    print(f"\n[Step 4] Evaluation Metrics")
    print("=" * 70)

    reductions = np.array([r['reduction_pct'] for r in results])
    l2_inits = np.array([r['lambda2_initial'] for r in results])
    l2_finals = np.array([r['lambda2_final'] for r in results])
    frags = np.array([r['fragmented'] for r in results])

    mean_red = np.mean(reductions)
    std_red = np.std(reductions)
    frag_rate = 100.0 * np.mean(frags)
    target = config.get('target_reduction', 70.0)

    print(f"  Total images evaluated : {len(results)}")
    print(f"  Skipped (too few drones): {skipped}")
    print(f"  Mean lambda_2 BEFORE   : {np.mean(l2_inits):.4f}")
    print(f"  Mean lambda_2 AFTER    : {np.mean(l2_finals):.4f}")
    print(f"  Mean reduction %       : {mean_red:.2f}% +/- {std_red:.2f}%")
    print(f"  Fragmentation rate     : {frag_rate:.1f}%")
    print(f"  Target                 : {target:.0f}%")
    print(f"  Result                 : {'PASS' if mean_red >= target else 'FAIL'}")
    print("=" * 70)

    # STEP 5 — Save CSV
    print(f"\n[Step 5] Saving results to {CSV_PATH} ...")
    with open(CSV_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'num_drones', 'lambda2_initial',
                         'lambda2_final', 'reduction_pct', 'fully_fragmented'])
        for r in results:
            writer.writerow([
                r['filename'], r['num_drones'],
                f"{r['lambda2_initial']:.6f}",
                f"{r['lambda2_final']:.6f}",
                f"{r['reduction_pct']:.2f}",
                r['fragmented']
            ])
    print(f"  Saved: {CSV_PATH}")

    # STEP 6 — Generate graphs
    print(f"\n[Step 6] Generating evaluation graphs ...")
    generate_graphs(results, OUTPUT_DIR)

    print(f"\n{'=' * 70}")
    print("EVALUATION COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
