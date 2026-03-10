#!/usr/bin/env python3
"""
make_video.py — MARL Jammer Swarm Disruption Video Generator
=============================================================

Produces a ~90-second 24 fps MP4 narrating the full jammer-drone scenario:

  Phase 1  (15 s, 360 frames) → ENEMY SWARM APPROACHING
      Enemy drones advance in formation; green jammers sit idle at a base
      in the bottom-left corner; comm links are dense (λ₂ is high).

  Phase 2  ( 5 s, 120 frames) → ANALYZING SWARM TOPOLOGY
      DBSCAN cluster halos expand, centroids marked with ★; λ₂ meter
      shown at full strength; "THREAT ANALYSED" alert pulses.

  Phase 3  (10 s, 240 frames) → DEPLOYING JAMMING UNITS
      Jammers fly from the base toward their assigned cluster centroids;
      enemy swarm continues moving; jamming circles not yet active.

  Phase 4  (~60 s, 12 episodes × 150 steps = 1800 frames) → JAMMING ACTIVE
      Full heuristic-agent loop: jammer positions, comm links, jamming
      radius circles, band labels, live λ₂ bar, reduction-% counter.
      "SWARM DISRUPTED ✓" overlay fires when λ₂ reduction exceeds 60 %.

Usage
-----
    # Full run (~90 s video)
    python make_video.py

    # Quick 10-second preview (first 240 frames only)
    python make_video.py --preview

    # Custom output path
    python make_video.py --output my_video.mp4

Requirements
------------
    pip install numpy scipy scikit-learn matplotlib tqdm
    brew install ffmpeg   (or apt/winget equivalent)

If ffmpeg is not found the script saves individual PNG frames instead.

Author: MARL Jammer Team
"""

import argparse
import os
import subprocess
import sys
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless, no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import to_rgba
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Workspace root & src path injection
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.resolve()
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))

from environment.jammer_env import JammerEnv
from environment.enemy_swarm import EnemySwarm
from clustering.dbscan_clustering import DBSCANClusterer, assign_jammers_to_clusters
from physics.communication_graph import (
    compute_adjacency_matrix, compute_laplacian, compute_lambda2
)
from physics.fspl import db_to_watts

# ---------------------------------------------------------------------------
# ── CONFIGURATION ──────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

# Video
FPS        = 24
VID_W      = 1280
VID_H      = 720

# Arena / simulation  (must match training config)
ARENA      = 150.0   # metres
N_ENEMY    = 30
N_JAMMER   = 6
V_ENEMY    = 2.0
V_MAX      = 5.0
DT         = 0.5
MAX_STEPS  = 150
EPS_DBSCAN = 25.0
MIN_SAMP   = 2

# Physics
P_TX_DBM   = 20.0
P_SENS_DBM = -90.0
P_JAM_DBM  = 30.0
P_JAM_THR  = -70.0

# Phase durations (seconds → frame counts)
DUR_P1 = 30        # enemy approaching
DUR_P2 = 12        # topology analysis
DUR_P3 = 20        # deployment
N_EPISODES_P4 = 20 # jamming active

FRAMES_P1 = DUR_P1 * FPS         # 720
FRAMES_P2 = DUR_P2 * FPS         # 288
FRAMES_P3 = DUR_P3 * FPS         # 480
FRAMES_P4 = N_EPISODES_P4 * MAX_STEPS  # 3000

# Colour palette
C_ENEMY     = "#FF4040"
C_ENEMY_ALT = "#FF8888"
C_JAMMER    = "#40FF80"
C_JAMMER_DIM= "#1A7D3B"
C_LINK      = "#888888"
C_JAM_RING  = "#40FF80"
C_BG        = "#0A0A14"
C_HUD_BG    = "#111120"
C_CENTROID  = "#FFDD00"
C_CLUSTER   = [
    "#FF6B9D", "#00D4FF", "#FFB347", "#B085F5",
    "#9CFF57", "#FF5733", "#33C5FF", "#FFD700",
]

BAND_NAMES  = ["B0 433MHz", "B1 915MHz", "B2 2.4GHz", "B3 5.8GHz"]
BAND_SHORT  = ["B0", "B1", "B2", "B3"]
BAND_FREQS  = [433e6, 915e6, 2.4e9, 5.8e9]

# ---------------------------------------------------------------------------
# ── HEURISTIC AGENT ────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

class HeuristicAgent:
    """
    Smart heuristic policy that reproduces what the trained PPO learned:
      1. Move each jammer toward its assigned cluster centroid.
      2. Use the known enemy frequency band.
      3. Apply slight repulsion between jammers to spread coverage.

    Falls back to random actions if no clustering info is available.
    """

    def __init__(self, M: int, v_max: float, arena_size: float):
        self.M          = M
        self.v_max      = v_max
        self.arena_size = arena_size
        self._rng       = np.random.default_rng(0)

    def get_action(
        self,
        jammer_positions: np.ndarray,   # (M, 2)
        centroids: dict,                # {cid: (2,)}
        jammer_assignments: dict,       # {cid: [jammer_indices]}
        enemy_band: int,
    ) -> dict:
        """
        Returns action dict compatible with JammerEnv:
            {'velocity': (M,2), 'band': (M,)}
        """
        velocity = np.zeros((self.M, 2), dtype=np.float32)
        bands    = np.full(self.M, enemy_band, dtype=np.int32)

        # Build reverse map: jammer_idx → centroid
        jammer_to_centroid = {}
        for cid, members in jammer_assignments.items():
            if cid in centroids:
                for jidx in members:
                    jammer_to_centroid[jidx] = centroids[cid]

        for j in range(self.M):
            if j in jammer_to_centroid:
                target = jammer_to_centroid[j]
            else:
                # No assignment → head to centre
                target = np.array([self.arena_size / 2, self.arena_size / 2])

            direction = target - jammer_positions[j]
            dist      = np.linalg.norm(direction) + 1e-8

            # Attack speed: full v_max until within settling radius, then slow
            SETTLE = 10.0
            if dist > SETTLE:
                speed = self.v_max
            else:
                speed = self.v_max * (dist / SETTLE)

            velocity[j] = (direction / dist) * speed

        # Repulsion between jammers (prevents overlap)
        REPULSE_DIST = 20.0
        for j in range(self.M):
            for k in range(j + 1, self.M):
                diff = jammer_positions[j] - jammer_positions[k]
                d    = np.linalg.norm(diff) + 1e-8
                if d < REPULSE_DIST:
                    force        = (diff / d) * (1.0 - d / REPULSE_DIST) * self.v_max * 0.5
                    velocity[j] += force
                    velocity[k] -= force

        # Clip to v_max
        norms = np.linalg.norm(velocity, axis=1, keepdims=True).clip(min=1e-8)
        too_fast = norms > self.v_max
        velocity = np.where(too_fast, velocity / norms * self.v_max, velocity)

        return {"velocity": velocity, "band": bands}


# ---------------------------------------------------------------------------
# ── ENV FACTORY ────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def make_env(seed: int = 0) -> JammerEnv:
    """Create JammerEnv matching the training configuration."""
    return JammerEnv(
        M=N_JAMMER,
        N=N_ENEMY,
        arena_size=ARENA,
        max_steps=MAX_STEPS,
        dt=DT,
        v_max=V_MAX,
        v_enemy=V_ENEMY,
        enemy_mode="random_walk",
        P_tx_dbm=P_TX_DBM,
        P_sens_dbm=P_SENS_DBM,
        P_jammer_dbm=P_JAM_DBM,
        P_jam_thresh_dbm=P_JAM_THR,
        omega_1=10.0,   # λ₂ reduction only (matches training)
        omega_2=0.0,
        omega_3=0.0,
        omega_4=0.0,
        omega_5=0.0,
        eps=EPS_DBSCAN,
        min_samples=MIN_SAMP,
        K_recompute=10,
        seed=seed,
        random_jammer_start=False,
    )


# ---------------------------------------------------------------------------
# ── FIGURE SETUP ───────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def make_figure():
    """
    Create figure split: arena (left 75%) + HUD (right 25%).
    Returns (fig, ax_arena, ax_hud).
    """
    fig = plt.figure(figsize=(VID_W / 100, VID_H / 100), dpi=100)
    fig.patch.set_facecolor(C_BG)

    # Arena axes — slightly inset for visual padding
    ax_arena = fig.add_axes([0.01, 0.01, 0.72, 0.98])
    ax_arena.set_facecolor(C_BG)
    ax_arena.set_xlim(0, ARENA)
    ax_arena.set_ylim(0, ARENA)
    ax_arena.set_aspect("equal")
    ax_arena.axis("off")

    # HUD axes
    ax_hud = fig.add_axes([0.74, 0.01, 0.25, 0.98])
    ax_hud.set_facecolor(C_HUD_BG)
    ax_hud.set_xlim(0, 1)
    ax_hud.set_ylim(0, 1)
    ax_hud.axis("off")

    return fig, ax_arena, ax_hud


# ---------------------------------------------------------------------------
# ── COMMUNICATION LINKS ────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def get_comm_links(
    enemy_positions: np.ndarray,
    enemy_band: int,
    jammer_positions: np.ndarray = None,
    R_jam: float = 0.0,
) -> list:
    """
    Return list of (i, j) pairs that still have active comm links.
    Optionally applies jamming (if jammer_positions & R_jam given).
    """
    freq    = BAND_FREQS[enemy_band]
    p_tx_w  = db_to_watts(P_TX_DBM)
    p_s_w   = db_to_watts(P_SENS_DBM)

    # Compute jammed links if we have jammer positions
    jammed = None
    if jammer_positions is not None and R_jam > 0:
        N = len(enemy_positions)
        jammed = np.zeros((N, N), dtype=bool)
        for ii in range(N):
            for jj in range(ii + 1, N):
                midpoint = (enemy_positions[ii] + enemy_positions[jj]) / 2.0
                for jp in jammer_positions:
                    dist = np.linalg.norm(jp - midpoint)
                    if dist <= R_jam:
                        jammed[ii, jj] = True
                        jammed[jj, ii] = True
                        break

    A = compute_adjacency_matrix(
        enemy_positions, p_tx_w, p_s_w, freq, jammed_links=jammed
    )
    ii, jj = np.where(np.triu(A, k=1) == 1)
    return list(zip(ii.tolist(), jj.tolist()))


# ---------------------------------------------------------------------------
# ── DRAW HELPERS ───────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def draw_base_grid(ax):
    """Faint grid lines suggesting battlefield grid."""
    for v in np.arange(0, ARENA + 1, 25):
        ax.axhline(v, color="#222233", lw=0.4, zorder=0)
        ax.axvline(v, color="#222233", lw=0.4, zorder=0)


def draw_comm_links(ax, links, enemy_positions, alpha=0.35, color=C_LINK, lw=0.7):
    for i, j in links:
        x = [enemy_positions[i, 0], enemy_positions[j, 0]]
        y = [enemy_positions[i, 1], enemy_positions[j, 1]]
        ax.plot(x, y, color=color, alpha=alpha, lw=lw, zorder=1)


def draw_enemies(ax, positions, labels=None):
    """
    Plot enemy drones.  If DBSCAN labels supplied, colour by cluster.
    """
    if labels is None:
        ax.scatter(
            positions[:, 0], positions[:, 1],
            s=60, color=C_ENEMY, edgecolors="#FF0000",
            linewidths=0.5, zorder=4, alpha=0.9,
        )
    else:
        for cid in np.unique(labels):
            mask = labels == cid
            col  = C_ENEMY if cid == -1 else C_CLUSTER[cid % len(C_CLUSTER)]
            ax.scatter(
                positions[mask, 0], positions[mask, 1],
                s=60, color=col, edgecolors=col,
                linewidths=0.5, zorder=4, alpha=0.9,
            )


def draw_jammers(ax, positions, bands=None, active=True, dim=False):
    """
    Plot jammer drones as green/dim triangles with optional band labels.
    """
    col   = C_JAMMER if active else C_JAMMER_DIM
    ecol  = "#00CC55" if active else "#1A5C30"
    alpha = 0.95 if active else 0.5
    ax.scatter(
        positions[:, 0], positions[:, 1],
        s=90, color=col, marker="^",
        edgecolors=ecol, linewidths=0.7,
        zorder=5, alpha=alpha,
    )
    if bands is not None:
        for j, (px, py) in enumerate(positions):
            label = BAND_SHORT[bands[j]]
            ax.text(
                px, py + 4.5, label,
                color=C_JAMMER, fontsize=5.5, ha="center", va="bottom",
                fontweight="bold", zorder=6,
            )


def draw_jamming_circles(ax, positions, R_jam):
    for px, py in positions:
        circle = plt.Circle(
            (px, py), R_jam,
            color=C_JAM_RING, alpha=0.12,
            linewidth=0.8, linestyle="--",
            fill=True, zorder=2,
        )
        ax.add_patch(circle)
        ring = plt.Circle(
            (px, py), R_jam,
            color=C_JAM_RING, alpha=0.5,
            linewidth=0.8, linestyle="--",
            fill=False, zorder=3,
        )
        ax.add_patch(ring)


def draw_cluster_halos(ax, enemy_positions, labels, centroids, halo_scale=1.0):
    """
    Animated expanding halos for DBSCAN clusters.
    halo_scale in [0, 1] controls the ring radius.
    """
    for cid, centroid in centroids.items():
        mask = labels == cid
        if not np.any(mask):
            continue
        members = enemy_positions[mask]
        max_radius = np.max(np.linalg.norm(members - centroid, axis=1)) + 5.0
        r = max_radius * halo_scale
        col = C_CLUSTER[cid % len(C_CLUSTER)]
        halo = plt.Circle(
            (centroid[0], centroid[1]), r,
            color=col, alpha=0.06 * halo_scale,
            linewidth=1.2, linestyle="--",
            fill=True, zorder=1,
        )
        ax.add_patch(halo)
        ring = plt.Circle(
            (centroid[0], centroid[1]), r,
            color=col, alpha=0.5 * halo_scale,
            linewidth=1.2, linestyle="--",
            fill=False, zorder=2,
        )
        ax.add_patch(ring)
        # Centroid star
        ax.scatter(
            centroid[0], centroid[1],
            s=140 * halo_scale, marker="*",
            color=C_CENTROID, alpha=halo_scale,
            zorder=6,
        )


def draw_hud(
    ax,
    phase_title: str,
    lambda2_initial: float,
    lambda2_current: float,
    lambda2_reduction_pct: float,
    episode_num: int,
    step_num: int,
    enemy_band: int,
    n_clusters: int,
    alert: str = "",
    alert_alpha: float = 0.0,
    frame_num: int = 0,
):
    ax.cla()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_facecolor(C_HUD_BG)
    ax.axis("off")

    # ── Title bar ──────────────────────────────────────────────────────────
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (0.02, 0.88), 0.96, 0.10,
            boxstyle="round,pad=0.01",
            facecolor="#1A1A2E", edgecolor="#444488", lw=1.0,
        )
    )
    ax.text(
        0.5, 0.924, phase_title,
        ha="center", va="center", fontsize=7.5, fontweight="bold",
        color="#88AAFF", transform=ax.transAxes,
        wrap=True,
    )

    # ── λ₂ Meter ───────────────────────────────────────────────────────────
    ax.text(0.5, 0.83, "ALGEBRAIC CONNECTIVITY  λ₂",
            ha="center", va="center", fontsize=6.0,
            color="#AAAACC", transform=ax.transAxes)

    # Bar background
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (0.05, 0.74), 0.90, 0.06,
            boxstyle="round,pad=0.005",
            facecolor="#0A0A1A", edgecolor="#444488", lw=0.8,
        )
    )
    # Bar fill — red when high, transitions to black as disrupted
    if lambda2_initial > 0:
        ratio = max(0.0, min(1.0, lambda2_current / lambda2_initial))
    else:
        ratio = 1.0
    bar_color = (
        ratio * np.array([1.0, 0.2, 0.2]) +
        (1 - ratio) * np.array([0.1, 0.8, 0.3])
    )
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (0.05, 0.74), 0.90 * ratio, 0.06,
            boxstyle="square,pad=0",
            facecolor=bar_color, edgecolor="none",
        )
    )
    ax.text(
        0.5, 0.77, f"{lambda2_current:.3f}",
        ha="center", va="center", fontsize=7.5, fontweight="bold",
        color="white", transform=ax.transAxes,
    )

    # ── Reduction % ────────────────────────────────────────────────────────
    pct_color = "#FF4444" if lambda2_reduction_pct < 30 else (
        "#FFAA00" if lambda2_reduction_pct < 60 else "#40FF80"
    )
    ax.text(
        0.5, 0.685, f"Reduction:  {lambda2_reduction_pct:.1f} %",
        ha="center", va="center", fontsize=8.5, fontweight="bold",
        color=pct_color, transform=ax.transAxes,
    )

    # ── Stats block ────────────────────────────────────────────────────────
    def stat_line(y, label, value, vcol="#DDDDFF"):
        ax.text(0.08, y, label, ha="left", va="center", fontsize=6.2,
                color="#888899", transform=ax.transAxes)
        ax.text(0.92, y, value, ha="right", va="center", fontsize=6.2,
                color=vcol, transform=ax.transAxes)

    stat_line(0.63, "Episode",   f"{episode_num}")
    stat_line(0.59, "Step",      f"{step_num} / {MAX_STEPS}")
    stat_line(0.55, "Clusters",  f"{n_clusters}")
    stat_line(0.51, "Freq Band", BAND_NAMES[enemy_band], "#FFDD88")
    stat_line(0.47, "Jammers",   f"{N_JAMMER}")
    stat_line(0.43, "Enemies",   f"{N_ENEMY}")

    # ── Legend ─────────────────────────────────────────────────────────────
    y0 = 0.35
    for marker, color, label in [
        ("o", C_ENEMY,   "Enemy Drone"),
        ("^", C_JAMMER,  "Jammer Agent"),
        ("*", C_CENTROID,"Cluster Centroid"),
        ("-", C_LINK,    "Comm Link"),
    ]:
        ax.plot(
            0.12, y0, marker=marker if marker != "-" else None,
            linestyle="-" if marker == "-" else "none",
            color=color, markersize=5, lw=1.5,
            transform=ax.transAxes,
        )
        ax.text(0.22, y0, label, ha="left", va="center",
                fontsize=5.8, color="#BBBBCC", transform=ax.transAxes)
        y0 -= 0.055

    # Jamming ring swatch
    patch = mpatches.Circle((0.12, y0), 0.02, color=C_JAM_RING, alpha=0.4,
                             transform=ax.transAxes)
    ax.add_patch(patch)
    ax.text(0.22, y0, "Jamming Radius", ha="left", va="center",
            fontsize=5.8, color="#BBBBCC", transform=ax.transAxes)

    # ── Alert overlay ──────────────────────────────────────────────────────
    if alert_alpha > 0.01 and alert:
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (0.01, 0.04), 0.98, 0.12,
                boxstyle="round,pad=0.01",
                facecolor=(0.0, 0.18, 0.0, alert_alpha * 0.85),
                edgecolor=(0.2, 1.0, 0.4, alert_alpha),
                lw=1.5,
            )
        )
        ax.text(
            0.5, 0.10, alert,
            ha="center", va="center", fontsize=8.0, fontweight="bold",
            color=(0.3, 1.0, 0.5, alert_alpha),
            transform=ax.transAxes,
        )

    # ── Frame counter ──────────────────────────────────────────────────────
    ax.text(
        0.5, 0.01, f"Frame {frame_num:05d}",
        ha="center", va="bottom", fontsize=5.0,
        color="#444455", transform=ax.transAxes,
    )


def draw_phase_title(ax, title: str, subtitle: str = "", alpha: float = 1.0):
    """
    Overlay a large phase title on the arena axes.
    """
    effect = [pe.withStroke(linewidth=3, foreground=C_BG)]
    ax.text(
        ARENA / 2, ARENA * 0.94, title,
        ha="center", va="top", fontsize=13, fontweight="bold",
        color=(1.0, 1.0, 1.0, alpha),
        path_effects=effect, zorder=10,
    )
    if subtitle:
        ax.text(
            ARENA / 2, ARENA * 0.89, subtitle,
            ha="center", va="top", fontsize=7.5,
            color=(0.7, 0.9, 1.0, alpha * 0.85),
            zorder=10,
        )


# ---------------------------------------------------------------------------
# ── FRAME → BYTES ──────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def frame_to_bytes(fig) -> bytes:
    """Render figure to raw RGB24 bytes suitable for ffmpeg stdin."""
    fig.canvas.draw()
    # buffer_rgba() returns a memoryview of RGBA bytes; strip alpha for RGB24
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(VID_H, VID_W, 4)[:, :, :3]   # RGBA → RGB
    return buf.tobytes()


# ---------------------------------------------------------------------------
# ── FFMPEG PIPE ────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def open_ffmpeg_pipe(output_path: str):
    """Open an ffmpeg subprocess that accepts raw RGB24 frames on stdin."""
    cmd = [
        "ffmpeg", "-y",
        "-f",       "rawvideo",
        "-vcodec",  "rawvideo",
        "-s",       f"{VID_W}x{VID_H}",
        "-pix_fmt", "rgb24",
        "-r",       str(FPS),
        "-i",       "pipe:0",
        "-vcodec",  "libx264",
        "-preset",  "fast",
        "-crf",     "22",
        "-pix_fmt", "yuv420p",
        output_path,
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return proc
    except FileNotFoundError:
        return None


def write_frame(proc, fig, frame_dir: Path = None, frame_num: int = 0):
    """
    Write one rendered frame.
    - Always pipes raw RGB24 to ffmpeg if proc is open.
    - Always saves a PNG to frame_dir when it is set (regardless of ffmpeg).
    """
    raw = frame_to_bytes(fig)
    if proc is not None:
        proc.stdin.write(raw)
    if frame_dir is not None:
        # Determine phase sub-folder from frame number ranges (set in main)
        png_path = frame_dir / f"frame_{frame_num:05d}.png"
        fig.savefig(str(png_path), dpi=100, bbox_inches=None)


# ---------------------------------------------------------------------------
# ── PHASE 1 — ENEMY SWARM APPROACHING ─────────────────────────────────────
# ---------------------------------------------------------------------------

def run_phase1(fig, ax_arena, ax_hud, proc, frame_dir, start_frame, n_frames, bar):
    """
    15 s — enemy swarm advancing, jammers dormant at base.
    Returns (enemy_positions, enemy_band) at end of phase.
    """
    swarm = EnemySwarm(
        N=N_ENEMY, mode="random_walk",
        v_enemy=V_ENEMY, dt=DT, arena_size=ARENA, seed=1,
    )
    swarm.reset(seed=1)

    # Jammer base: bottom-left 3×2 grid
    jammer_base = np.array([
        [8.0,  8.0], [18.0,  8.0], [28.0,  8.0],
        [8.0, 18.0], [18.0, 18.0], [28.0, 18.0],
    ])
    jam_bands = np.zeros(N_JAMMER, dtype=np.int32)

    # Pre-compute initial comm links (unjammed)
    p_tx_w = db_to_watts(P_TX_DBM)
    p_s_w  = db_to_watts(P_SENS_DBM)
    ENEMY_BAND_DEFAULT = 2  # 2.4 GHz default for phase 1

    fnum = start_frame
    for f in range(n_frames):
        swarm.step()
        e_pos = swarm.positions.copy()

        # Title pulse in
        title_alpha = min(1.0, f / 30.0)

        ax_arena.cla()
        ax_arena.set_facecolor(C_BG)
        ax_arena.set_xlim(0, ARENA)
        ax_arena.set_ylim(0, ARENA)
        ax_arena.axis("off")

        draw_base_grid(ax_arena)

        links = get_comm_links(e_pos, ENEMY_BAND_DEFAULT)
        draw_comm_links(ax_arena, links, e_pos, alpha=0.3)
        draw_enemies(ax_arena, e_pos)
        draw_jammers(ax_arena, jammer_base, bands=jam_bands, active=False, dim=True)

        # Base marker
        ax_arena.add_patch(
            mpatches.FancyBboxPatch(
                (2, 2), 35, 22,
                boxstyle="round,pad=1",
                facecolor=(0.1, 0.5, 0.25, 0.12),
                edgecolor=(0.2, 0.8, 0.4, 0.4),
                lw=0.8, zorder=1,
            )
        )
        ax_arena.text(
            19, 4, "JAMMER BASE",
            ha="center", va="bottom", fontsize=5.5,
            color=(0.2, 0.8, 0.4, 0.6), zorder=7,
        )

        draw_phase_title(ax_arena, "ENEMY SWARM APPROACHING",
                         f"Tracking {N_ENEMY} hostile drones", alpha=title_alpha)

        # λ₂ for current positions (no jamming yet)
        lam2 = compute_lambda2(
            compute_laplacian(
                compute_adjacency_matrix(e_pos, p_tx_w, p_s_w, BAND_FREQS[ENEMY_BAND_DEFAULT])
            )
        )
        draw_hud(
            ax_hud,
            phase_title="ENEMY SWARM\nAPPROACHING",
            lambda2_initial=lam2 if f == 0 else lam2,
            lambda2_current=lam2,
            lambda2_reduction_pct=0.0,
            episode_num=0,
            step_num=f,
            enemy_band=ENEMY_BAND_DEFAULT,
            n_clusters=0,
            frame_num=fnum,
        )

        write_frame(proc, fig, frame_dir, fnum)
        fnum += 1
        bar.update(1)

    return swarm.positions.copy(), ENEMY_BAND_DEFAULT


# ---------------------------------------------------------------------------
# ── PHASE 2 — ANALYZING SWARM TOPOLOGY ────────────────────────────────────
# ---------------------------------------------------------------------------

def run_phase2(fig, ax_arena, ax_hud, proc, frame_dir, start_frame, n_frames,
               enemy_positions, enemy_band, bar):
    """
    5 s — DBSCAN halos expand; cluster centroids appear.
    Returns (labels, centroids_dict).
    """
    clusterer = DBSCANClusterer(eps=EPS_DBSCAN, min_samples=MIN_SAMP, arena_size=ARENA)
    labels, centroids = clusterer.fit(enemy_positions)
    n_clusters        = clusterer.n_clusters

    jammer_base = np.array([
        [8.0,  8.0], [18.0,  8.0], [28.0,  8.0],
        [8.0, 18.0], [18.0, 18.0], [28.0, 18.0],
    ])
    jam_bands = np.zeros(N_JAMMER, dtype=np.int32)

    p_tx_w = db_to_watts(P_TX_DBM)
    p_s_w  = db_to_watts(P_SENS_DBM)

    lam2 = compute_lambda2(
        compute_laplacian(
            compute_adjacency_matrix(
                enemy_positions, p_tx_w, p_s_w, BAND_FREQS[enemy_band]
            )
        )
    )

    fnum = start_frame
    for f in range(n_frames):
        halo_scale = min(1.0, f / (n_frames * 0.6))   # expand over first 60 %
        # Alert flicker in last 30 %
        alert_alpha = max(0.0, (f - n_frames * 0.7) / (n_frames * 0.3))
        # Phase title pulse
        title_alpha = 1.0

        ax_arena.cla()
        ax_arena.set_facecolor(C_BG)
        ax_arena.set_xlim(0, ARENA)
        ax_arena.set_ylim(0, ARENA)
        ax_arena.axis("off")

        draw_base_grid(ax_arena)
        links = get_comm_links(enemy_positions, enemy_band)
        draw_comm_links(ax_arena, links, enemy_positions, alpha=0.4)
        draw_cluster_halos(ax_arena, enemy_positions, labels, centroids, halo_scale)
        draw_enemies(ax_arena, enemy_positions, labels=labels)
        draw_jammers(ax_arena, jammer_base, bands=jam_bands, active=False, dim=True)

        draw_phase_title(ax_arena, "ANALYZING SWARM TOPOLOGY",
                         f"Detected {n_clusters} enemy clusters via DBSCAN",
                         alpha=title_alpha)

        draw_hud(
            ax_hud,
            phase_title="ANALYZING SWARM\nTOPOLOGY",
            lambda2_initial=lam2,
            lambda2_current=lam2,
            lambda2_reduction_pct=0.0,
            episode_num=0,
            step_num=f,
            enemy_band=enemy_band,
            n_clusters=n_clusters,
            alert="⚠  THREAT ANALYSED",
            alert_alpha=alert_alpha,
            frame_num=fnum,
        )

        write_frame(proc, fig, frame_dir, fnum)
        fnum += 1
        bar.update(1)

    return labels, centroids


# ---------------------------------------------------------------------------
# ── PHASE 3 — DEPLOYING JAMMING UNITS ─────────────────────────────────────
# ---------------------------------------------------------------------------

def run_phase3(fig, ax_arena, ax_hud, proc, frame_dir, start_frame, n_frames,
               enemy_positions, enemy_band, labels, centroids, bar):
    """
    10 s — jammers fly from base → cluster centroids.
    Returns jammer_positions at deployment targets.
    """
    jammer_base = np.array([
        [8.0,  8.0], [18.0,  8.0], [28.0,  8.0],
        [8.0, 18.0], [18.0, 18.0], [28.0, 18.0],
    ], dtype=float)
    jam_bands   = np.zeros(N_JAMMER, dtype=np.int32)

    # Assign jammers to clusters, determine target positions
    from clustering.dbscan_clustering import assign_jammers_to_clusters, get_jammer_initial_positions
    assignments = assign_jammers_to_clusters(
        N_JAMMER, centroids,
        {cid: int(np.sum(labels == cid)) for cid in centroids},
        strategy="proportional"
    )
    target_positions = get_jammer_initial_positions(
        N_JAMMER, centroids, assignments,
        spread=10.0, arena_size=ARENA
    )

    swarm = EnemySwarm(
        N=N_ENEMY, mode="random_walk",
        v_enemy=V_ENEMY, dt=DT, arena_size=ARENA, seed=2,
    )
    swarm.positions = enemy_positions.copy()
    swarm.velocities = np.zeros((N_ENEMY, 2))

    p_tx_w = db_to_watts(P_TX_DBM)
    p_s_w  = db_to_watts(P_SENS_DBM)
    lam2   = compute_lambda2(
        compute_laplacian(
            compute_adjacency_matrix(
                enemy_positions, p_tx_w, p_s_w, BAND_FREQS[enemy_band]
            )
        )
    )

    fnum = start_frame
    for f in range(n_frames):
        t = f / max(1, n_frames)   # 0 → 1

        # Ease-in-out interpolation
        ease = t * t * (3 - 2 * t)
        j_pos = jammer_base + ease * (target_positions - jammer_base)

        swarm.step()
        e_pos = swarm.positions.copy()

        ax_arena.cla()
        ax_arena.set_facecolor(C_BG)
        ax_arena.set_xlim(0, ARENA)
        ax_arena.set_ylim(0, ARENA)
        ax_arena.axis("off")

        draw_base_grid(ax_arena)
        links = get_comm_links(e_pos, enemy_band)
        draw_comm_links(ax_arena, links, e_pos, alpha=0.35)
        draw_cluster_halos(ax_arena, e_pos, labels,
                           {cid: c for cid, c in centroids.items()},
                           halo_scale=0.6)
        draw_enemies(ax_arena, e_pos, labels=labels)
        draw_jammers(ax_arena, j_pos, bands=jam_bands, active=True)

        # Trajectory lines from base
        for j in range(N_JAMMER):
            ax_arena.plot(
                [jammer_base[j, 0], j_pos[j, 0]],
                [jammer_base[j, 1], j_pos[j, 1]],
                color=C_JAMMER, alpha=0.15, lw=0.6, zorder=1,
            )

        title_alpha = 1.0
        draw_phase_title(ax_arena, "DEPLOYING JAMMING UNITS",
                         "Positioning jammers at cluster centroids",
                         alpha=title_alpha)

        draw_hud(
            ax_hud,
            phase_title="DEPLOYING\nJAMMING UNITS",
            lambda2_initial=lam2,
            lambda2_current=lam2,
            lambda2_reduction_pct=0.0,
            episode_num=0,
            step_num=f,
            enemy_band=enemy_band,
            n_clusters=len(centroids),
            alert="→ WEAPONS FREE",
            alert_alpha=min(1.0, (f - n_frames * 0.6) / (n_frames * 0.4)) if f > n_frames * 0.6 else 0.0,
            frame_num=fnum,
        )

        write_frame(proc, fig, frame_dir, fnum)
        fnum += 1
        bar.update(1)

    return target_positions


# ---------------------------------------------------------------------------
# ── PHASE 4 — JAMMING ACTIVE ───────────────────────────────────────────────
# ---------------------------------------------------------------------------

def run_phase4(fig, ax_arena, ax_hud, proc, frame_dir, start_frame,
               bar, preview=False):
    """
    12 episodes × 150 steps = 1800 frames (~75 s of content).
    In preview mode only runs 60 frames.
    """
    env   = make_env(seed=42)
    agent = HeuristicAgent(N_JAMMER, V_MAX, ARENA)

    max_frames_p4 = 60 if preview else FRAMES_P4

    episode_num   = 0
    total_frames  = 0
    fnum          = start_frame

    # Alert state
    alert_text    = ""
    alert_alpha   = 0.0
    disrupted_ctr = 0

    obs, info = env.reset(seed=42)

    lam2_init = info["lambda2_initial"]
    lam2_curr = lam2_init
    reduction = 0.0

    while total_frames < max_frames_p4:
        # --- Heuristic action ---
        action = agent.get_action(
            env.jammer_positions.copy(),
            env.centroids,
            env.jammer_assignments,
            env.enemy_band,
        )

        obs, reward, terminated, truncated, info = env.step(action)

        lam2_init = info["lambda2_initial"]
        lam2_curr = info["lambda2_current"]
        reduction = info.get("lambda2_reduction", 0.0) * 100.0  # to %

        e_pos = info["enemy_positions"].copy()
        j_pos = info["jammer_positions"].copy()
        j_bands   = env.jammer_bands.copy()
        n_clusters = info["n_clusters"]
        enemy_band = info["enemy_band"]
        labels     = env.clusterer.labels.copy() if env.clusterer.labels is not None else np.zeros(N_ENEMY, dtype=int)
        centroids  = dict(env.centroids)
        R_jam      = env.R_jam

        # Alert management
        if reduction > 60.0:
            disrupted_ctr = min(disrupted_ctr + 1, FPS * 3)  # hold 3 s
            alert_text    = "✓  SWARM DISRUPTED"
            alert_alpha   = min(1.0, disrupted_ctr / (FPS * 0.5))
        else:
            disrupted_ctr = max(0, disrupted_ctr - 1)
            alert_alpha   = max(0.0, disrupted_ctr / (FPS * 0.5))
            if disrupted_ctr == 0:
                alert_text = ""

        # ── render ────────────────────────────────────────────────────────
        ax_arena.cla()
        ax_arena.set_facecolor(C_BG)
        ax_arena.set_xlim(0, ARENA)
        ax_arena.set_ylim(0, ARENA)
        ax_arena.axis("off")

        draw_base_grid(ax_arena)

        links = get_comm_links(e_pos, enemy_band, j_pos, R_jam)
        draw_comm_links(ax_arena, links, e_pos,
                        alpha=max(0.05, 0.45 - reduction / 250.0))
        draw_cluster_halos(ax_arena, e_pos, labels, centroids, halo_scale=0.5)
        draw_jamming_circles(ax_arena, j_pos, R_jam)
        draw_enemies(ax_arena, e_pos, labels=labels)
        draw_jammers(ax_arena, j_pos, bands=j_bands, active=True)

        draw_phase_title(ax_arena, "JAMMING ACTIVE",
                         f"Episode {episode_num + 1}  |  Step {env.step_count}",
                         alpha=1.0)

        draw_hud(
            ax_hud,
            phase_title="JAMMING ACTIVE",
            lambda2_initial=lam2_init,
            lambda2_current=lam2_curr,
            lambda2_reduction_pct=reduction,
            episode_num=episode_num + 1,
            step_num=env.step_count,
            enemy_band=enemy_band,
            n_clusters=n_clusters,
            alert=alert_text,
            alert_alpha=alert_alpha,
            frame_num=fnum,
        )

        write_frame(proc, fig, frame_dir, fnum)
        fnum        += 1
        total_frames += 1
        bar.update(1)

        if terminated or truncated:
            episode_num += 1
            if episode_num >= N_EPISODES_P4 and not preview:
                break
            obs, info = env.reset(seed=42 + episode_num)
            lam2_init = info["lambda2_initial"]


# ---------------------------------------------------------------------------
# ── MAIN ───────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MARL Jammer Swarm Video Generator")
    parser.add_argument("--output",  default="swarm_disruption_video.mp4",
                        help="Output MP4 path  (default: swarm_disruption_video.mp4)")
    parser.add_argument("--preview", action="store_true",
                        help="Quick ~10-second preview of all 4 phases")
    parser.add_argument("--no-frames", action="store_true",
                        help="Skip saving individual PNG frames (faster, less disk)")
    parser.add_argument("--frames-dir", default="",
                        help="Custom folder for PNG frames (default: outputs/video_frames/<name>)")
    args = parser.parse_args()

    output_path = str(ROOT / args.output) if not Path(args.output).is_absolute() else args.output
    preview     = args.preview
    save_frames = not args.no_frames

    # Frame counts for each phase
    if preview:
        frames_p1     = 72    # 3 s
        frames_p2     = 48    # 2 s
        frames_p3     = 72    # 3 s
        frames_p4_run = 60    # ~2.5 s
    else:
        frames_p1     = FRAMES_P1
        frames_p2     = FRAMES_P2
        frames_p3     = FRAMES_P3
        frames_p4_run = FRAMES_P4

    total_frames = frames_p1 + frames_p2 + frames_p3 + frames_p4_run

    # Derive a clean run name from the output filename
    run_name = Path(output_path).stem   # e.g. "swarm_disruption_video"

    # -- Frame directory -----------------------------------------------------
    if save_frames:
        if args.frames_dir:
            frame_dir = Path(args.frames_dir)
        else:
            frame_dir = ROOT / "outputs" / "video_frames" / run_name
        frame_dir.mkdir(parents=True, exist_ok=True)
    else:
        frame_dir = None

    print(f"\n{'='*60}")
    print(" MARL Jammer Swarm Disruption — Video Generator")
    print(f"{'='*60}")
    print(f"  Resolution : {VID_W} × {VID_H}  @  {FPS} fps")
    print(f"  Mode       : {'PREVIEW (fast)' if preview else 'FULL'}")
    print(f"  Frames     : {total_frames}  (~{total_frames/FPS:.0f} seconds)")
    print(f"  Output     : {output_path}")
    print(f"  PNG frames : {'disabled' if frame_dir is None else str(frame_dir)}")
    print(f"{'='*60}\n")

    # -- Try ffmpeg ----------------------------------------------------------
    proc = open_ffmpeg_pipe(output_path)
    if proc is None:
        print("[WARNING] ffmpeg not found — video will not be compiled.")
        if frame_dir is None:
            # Force frame saving so the run produces something
            frame_dir = ROOT / "outputs" / "video_frames" / run_name
            frame_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO]    PNG frames → {frame_dir}")
        print(f"[INFO]    Compile with:  ffmpeg -r {FPS} -i "
              f"{frame_dir}/frame_%05d.png -vcodec libx264 -pix_fmt yuv420p {output_path}\n")
    else:
        print("[OK] ffmpeg pipe opened.")
    if frame_dir is not None:
        print(f"[OK] PNG frames will be saved to: {frame_dir}")

    # -- Build figure --------------------------------------------------------
    fig, ax_arena, ax_hud = make_figure()
    # Agg backend set via matplotlib.use("Agg") at top of file

    # -- Progress bar --------------------------------------------------------
    bar = tqdm(total=total_frames, desc="Rendering", unit="frame",
               bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

    # ── PHASE 1 ─────────────────────────────────────────────────────────────
    print("Phase 1 — ENEMY SWARM APPROACHING …")
    enemy_pos, enemy_band = run_phase1(
        fig, ax_arena, ax_hud, proc, frame_dir,
        start_frame=0, n_frames=frames_p1, bar=bar)

    # ── PHASE 2 ─────────────────────────────────────────────────────────────
    print("Phase 2 — ANALYZING SWARM TOPOLOGY …")
    labels, centroids = run_phase2(
        fig, ax_arena, ax_hud, proc, frame_dir,
        start_frame=frames_p1, n_frames=frames_p2,
        enemy_positions=enemy_pos, enemy_band=enemy_band, bar=bar)

    # ── PHASE 3 ─────────────────────────────────────────────────────────────
    print("Phase 3 — DEPLOYING JAMMING UNITS …")
    run_phase3(
        fig, ax_arena, ax_hud, proc, frame_dir,
        start_frame=frames_p1 + frames_p2, n_frames=frames_p3,
        enemy_positions=enemy_pos, enemy_band=enemy_band,
        labels=labels, centroids=centroids, bar=bar)

    # ── PHASE 4 ─────────────────────────────────────────────────────────────
    print("Phase 4 — JAMMING ACTIVE …")
    run_phase4(fig, ax_arena, ax_hud, proc, frame_dir,
               start_frame=frames_p1 + frames_p2 + frames_p3,
               bar=bar, preview=preview)

    # -- Finalize ------------------------------------------------------------
    bar.close()
    plt.close(fig)

    if proc is not None:
        proc.stdin.close()
        proc.wait()
        print(f"\n[✓] Video saved → {output_path}")
        print(f"    Duration  : {total_frames/FPS:.1f} s  ({total_frames} frames)")
        print(f"\nPlay with:  open {output_path}")
    else:
        print(f"\n[✓] No video compiled (ffmpeg unavailable).")

    if frame_dir is not None:
        saved = len(list(frame_dir.glob('frame_*.png')))
        print(f"[✓] {saved} PNG frames saved → {frame_dir}")
        if proc is not None:
            print(f"    Re-compile any time:")
            print(f"    ffmpeg -r {FPS} -i {frame_dir}/frame_%05d.png "
                  f"-vcodec libx264 -pix_fmt yuv420p {output_path}")


if __name__ == "__main__":
    main()
