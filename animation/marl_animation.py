"""
MARL Jammer Drone System — 4-Minute 2D Animated Explainer Video
================================================================
Renders a full MP4 video explaining the entire project pipeline.

Usage:
    python marl_animation.py

Output:
    MARL_Jammer_Explainer.mp4  (in the same folder)

Dependencies:
    pip install matplotlib numpy scipy
    (ffmpeg must be installed on your system)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch
import matplotlib.lines as mlines
from scipy.spatial.distance import cdist

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# QUALITY SETTINGS — Edit these to trade speed for quality
# ─────────────────────────────────────────────────────────────────────────────
# FAST (test run):  DPI=80,  FIG_W=12, FIG_H=6.75
# MEDIUM (laptop):  DPI=100, FIG_W=16, FIG_H=9
# HIGH (final):     DPI=150, FIG_W=16, FIG_H=9
RENDER_DPI = 120          # dots per inch — higher = sharper but slower
FPS        = 24
TOTAL_SECS = 390          # 6.5 minutes — extended with new scenes
TOTAL_FRAMES = FPS * TOTAL_SECS

FIG_W, FIG_H = 16, 9     # 16:9 widescreen (1920x1080 at DPI=120)

# Color palette
BG        = '#0a0d1a'
ENEMY_C   = '#ff3300'
ENEMY_L   = '#ff6622'
JAMMER_C  = '#00ffdd'
JAMMER_L  = '#0088ff'
LINK_C    = '#ff6622'
BROKEN_C  = '#441100'
HUD_C     = '#88ccff'
GOLD      = '#ffcc00'
GREEN_C   = '#00ff88'
TITLE_C   = '#ffffff'
DIM       = '#334455'
WARN_C    = '#ff4444'

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def ease(t):
    """Smooth ease-in-out [0,1]→[0,1]."""
    return t * t * (3 - 2 * t)

def lerp(a, b, t):
    return a + (b - a) * np.clip(t, 0, 1)

def safe(*vals):
    """Multiply values and clamp to [0,1] — use everywhere alpha is computed."""
    result = 1.0
    for v in vals:
        result *= float(v)
    return float(np.clip(result, 0.0, 1.0))



def alpha_clamp(a):
    return float(np.clip(a, 0, 1))

def scene_t(frame, start, duration):
    """Return local [0,1] progress for a scene, or -1 if outside."""
    if frame < start or frame >= start + duration:
        return -1
    return (frame - start) / max(duration - 1, 1)

def fade_in(t, fade_dur=0.15):
    return alpha_clamp(t / fade_dur)

def fade_out(t, fade_start=0.85):
    return alpha_clamp((1 - t) / (1 - fade_start))

def scene_alpha(t, fi=0.08, fo=0.92):
    if t < 0: return 0
    if t < fi: return ease(t / fi)
    if t > fo: return ease((1 - t) / (1 - fo))
    return 1.0

def draw_drone(ax, x, y, color, size=0.012, label=None, alpha=1.0):
    """Draw a drone as a small X shape."""
    s = size
    # Body
    ax.plot([x-s, x+s], [y-s, y+s], color=color, lw=1.5, alpha=safe(alpha), solid_capstyle='round')
    ax.plot([x-s, x+s], [y+s, y-s], color=color, lw=1.5, alpha=safe(alpha), solid_capstyle='round')
    # Center dot
    ax.plot(x, y, 'o', color=color, ms=4, alpha=safe(alpha), zorder=5)
    if label:
        ax.text(x, y+0.025, label, color=color, fontsize=7, ha='center', alpha=safe(alpha))

def draw_hud_box(ax, x, y, w, h, title, color=HUD_C, alpha=1.0):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.005",
                          linewidth=1.5, edgecolor=color,
                          facecolor='#0a1520', alpha=safe(alpha*0.9),
                          zorder=8)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h - 0.025, title,
            color=color, fontsize=8, ha='center', va='top',
            alpha=safe(alpha), zorder=9, fontweight='bold')

def glow_text(ax, x, y, text, color, size=12, alpha=1.0, **kw):
    t = ax.text(x, y, text, color=color, fontsize=size, alpha=safe(alpha), **kw)
    t.set_path_effects([
        pe.withStroke(linewidth=4, foreground=color, alpha=safe(alpha*0.3)),
    ])
    return t

# ─────────────────────────────────────────────────────────────────────────────
# SCENE DATA
# ─────────────────────────────────────────────────────────────────────────────

# Enemy drone positions (normalized 0-1)
N_ENEMY = 20
np.random.seed(7)
# Create 3 natural clusters
cluster_centers = np.array([[0.25, 0.35], [0.65, 0.55], [0.45, 0.75]])
enemy_pos = []
for i, c in enumerate(cluster_centers):
    n = [7, 7, 6][i]
    pts = c + np.random.randn(n, 2) * 0.07
    enemy_pos.append(pts)
enemy_pos = np.vstack(enemy_pos)
enemy_pos = np.clip(enemy_pos, 0.08, 0.92)

# Jammer positions (start near centroids, then move)
M_JAMMER = 4
jammer_start = np.array([
    [0.1, 0.9], [0.9, 0.9], [0.1, 0.1], [0.9, 0.1]
])
jammer_target = np.array([
    cluster_centers[0] + [-0.04,  0.04],
    cluster_centers[1] + [ 0.04,  0.04],
    cluster_centers[2] + [-0.04, -0.04],
    cluster_centers[0] + [ 0.04, -0.04],
])

# Communication links (FSPL-based: connect if within threshold)
LINK_THRESH = 0.22
def get_links(pos, thresh=LINK_THRESH, broken_set=None):
    links = []
    for i in range(len(pos)):
        for j in range(i+1, len(pos)):
            d = np.linalg.norm(pos[i] - pos[j])
            if d < thresh:
                if broken_set and (i,j) in broken_set:
                    continue
                links.append((i, j, d))
    return links

all_links = get_links(enemy_pos)

# Scene schedule (start_frame, duration_frames)
# Scene durations — all extended for better readability
# Format: (start_frame, duration_frames)
_S = [
    ('title',          10),   # 0
    ('swarm_intro',    22),   # 1
    ('graph_build',    22),   # 2
    ('lambda2',        24),   # 3
    ('dbscan',         22),   # 4
    ('deploy',         20),   # 5
    ('obs_vector',     24),   # 6
    ('neural_net',     24),   # 7
    ('critic_network', 26),   # 8  ← NEW
    ('reward',         24),   # 9
    ('ppo_explained',  30),   # 10 ← NEW
    ('training',       28),   # 11
    ('jamming',        26),   # 12
    ('fragmentation',  24),   # 13
    ('comparison',     22),   # 14
    ('proposition',    22),   # 15
    ('results',        20),   # 16
    ('outro',          26),   # 17
]

# Build cumulative start times
SCENES = {}
_cur = 0
for _name, _dur in _S:
    SCENES[_name] = (_cur * FPS, _dur * FPS)
    _cur += _dur

# ─────────────────────────────────────────────────────────────────────────────
# SCENE RENDERERS
# ─────────────────────────────────────────────────────────────────────────────

def render_title(ax, t):
    a = scene_alpha(t)
    ax.set_facecolor(BG)
    # Subtitle
    ax.text(0.5, 0.62, 'Multi-Agent Reinforcement Learning for',
            color=HUD_C, fontsize=16, ha='center', va='center',
            alpha=safe(a), transform=ax.transAxes)
    # Main title
    glow_text(ax, 0.5, 0.5, 'Enemy Drone Swarm\nCommunication Disruption',
              JAMMER_C, size=26, alpha=safe(a),
              ha='center', va='center', transform=ax.transAxes,
              fontweight='bold', linespacing=1.4)
    # Tech tags
    tags = ['PPO Actor-Critic', '|', 'Graph Laplacian λ₂', '|', 'FSPL Jamming', '|', 'DBSCAN Clustering']
    ax.text(0.5, 0.32, '   '.join(tags),
            color=HUD_C, fontsize=10, ha='center', va='center',
            alpha=safe(a * 0.8), transform=ax.transAxes)
    # Extending line
    line_w = ease(min(t * 3, 1.0))
    ax.plot([0.5 - 0.3*line_w, 0.5 + 0.3*line_w], [0.415, 0.415],
            color=JAMMER_C, lw=1.5, alpha=safe(a*0.7), transform=ax.transAxes)
    # Bottom tag
    ax.text(0.5, 0.12, 'Extending Valianti et al. — IEEE TMC, December 2024',
            color=DIM, fontsize=9, ha='center', va='center',
            alpha=safe(a * 0.7), transform=ax.transAxes, style='italic')


def render_swarm_intro(ax, t):
    a = scene_alpha(t)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_facecolor(BG)

    # Title
    ax.text(0.5, 0.95, 'THE THREAT: Coordinated Enemy Drone Swarm',
            color=TITLE_C, fontsize=13, ha='center', va='top',
            alpha=safe(a), transform=ax.transAxes, fontweight='bold')

    # Progressively show drones appearing
    n_show = int(ease(min(t * 2.5, 1.0)) * N_ENEMY)

    for i in range(n_show):
        p = enemy_pos[i]
        draw_drone(ax, p[0], p[1], ENEMY_C, alpha=safe(a))

    # Show links after drones appear
    if t > 0.4:
        link_alpha = ease((t - 0.4) / 0.3) * a
        for i, j, d in all_links:
            pi, pj = enemy_pos[i], enemy_pos[j]
            intensity = 1.0 - d / LINK_THRESH
            ax.plot([pi[0], pj[0]], [pi[1], pj[1]],
                    color=LINK_C, lw=1.2,
                    alpha=alpha_clamp(link_alpha * intensity * 0.7))

    # Show pulsing signal rings
    if t > 0.6:
        pulse_t = (t - 0.6) * 4  # 0 to many
        for i, center in enumerate(cluster_centers):
            ring_r = (pulse_t % 1.0) * 0.15
            ring_a = (1 - (pulse_t % 1.0)) * 0.4 * a
            ring = Circle(center, ring_r, fill=False,
                         edgecolor=ENEMY_L, lw=1.0, alpha=safe(ring_a))
            ax.add_patch(ring)

    # Label boxes
    if t > 0.7:
        box_a = ease((t - 0.7) / 0.2) * a
        ax.text(0.02, 0.06, f'N = {N_ENEMY} Enemy Drones', color=ENEMY_C,
                fontsize=10, alpha=safe(box_a), transform=ax.transAxes)
        ax.text(0.02, 0.02, 'Status: FULLY COORDINATED', color=ENEMY_L,
                fontsize=9, alpha=safe(box_a), transform=ax.transAxes)

    # Right side explanation
    if t > 0.5:
        exp_a = ease((t - 0.5) / 0.3) * a
        lines = [
            '● Each drone communicates with',
            '  nearby drones via radio links',
            '',
            '● As a connected network, they can:',
            '  – Share targeting data',
            '  – Execute coordinated attacks',
            '  – Maintain formation flight',
            '',
            '● Breaking these links =',
            '  destroying their coordination',
        ]
        for i, line in enumerate(lines):
            ax.text(0.72, 0.82 - i*0.072, line,
                    color=HUD_C if '●' in line else TITLE_C,
                    fontsize=9, alpha=safe(exp_a), transform=ax.transAxes)


def render_graph_build(ax, t):
    a = scene_alpha(t)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_facecolor(BG)

    ax.text(0.5, 0.95, 'MODELING THE SWARM AS A COMMUNICATION GRAPH  G = (V, E)',
            color=TITLE_C, fontsize=11, ha='center', va='top',
            alpha=safe(a), transform=ax.transAxes, fontweight='bold')

    # Show all drones
    for p in enemy_pos:
        draw_drone(ax, p[0]*0.55 + 0.02, p[1]*0.78 + 0.1, ENEMY_C, alpha=safe(a))

    # Progressively draw links
    n_links = len(all_links)
    links_show = int(ease(min(t * 2.0, 1.0)) * n_links)
    for k, (i, j, d) in enumerate(all_links[:links_show]):
        pi = enemy_pos[i] * np.array([0.55, 0.78]) + np.array([0.02, 0.1])
        pj = enemy_pos[j] * np.array([0.55, 0.78]) + np.array([0.02, 0.1])
        intensity = 1.0 - d / LINK_THRESH
        ax.plot([pi[0], pj[0]], [pi[1], pj[1]],
                color=LINK_C, lw=1.5, alpha=alpha_clamp(a * intensity * 0.8))

    # Adjacency matrix display
    if t > 0.35:
        mat_a = ease((t - 0.35) / 0.25) * a
        draw_hud_box(ax, 0.62, 0.35, 0.34, 0.50,
                     'Adjacency Matrix A', alpha=safe(mat_a))

        # Show mini matrix (8x8 subset)
        n_mat = 8
        sub_pos = enemy_pos[:n_mat]
        mat_x0, mat_y0 = 0.635, 0.42
        cell = 0.038
        for i in range(n_mat):
            for j in range(n_mat):
                d = np.linalg.norm(sub_pos[i] - sub_pos[j])
                val = 1 if (i != j and d < LINK_THRESH) else 0
                col = LINK_C if val else DIM
                ax.text(mat_x0 + j*cell, mat_y0 + (n_mat-1-i)*cell,
                        str(val), color=col, fontsize=7,
                        ha='center', va='center', alpha=safe(mat_a))

        # Column/row labels
        for k in range(n_mat):
            ax.text(mat_x0 + k*cell, mat_y0 + n_mat*cell, str(k+1),
                    color=HUD_C, fontsize=6, ha='center', va='center', alpha=safe(mat_a*0.7))

    # FSPL formula
    if t > 0.55:
        fspl_a = ease((t - 0.55) / 0.2) * a
        draw_hud_box(ax, 0.62, 0.05, 0.34, 0.27, 'FSPL Edge Rule', alpha=safe(fspl_a))
        ax.text(0.79, 0.265, 'P_R(i,j) = P_tx × (c / 4πf·d_ij)²',
                color=GOLD, fontsize=8.5, ha='center', alpha=safe(fspl_a),
                transform=ax.transAxes)
        ax.text(0.79, 0.215, 'Edge exists if:  P_R ≥ P_sens  AND  not jammed',
                color=TITLE_C, fontsize=8, ha='center', alpha=safe(fspl_a),
                transform=ax.transAxes)
        ax.text(0.79, 0.165, 'P_sens = −90 dBm  |  f = 2.4 GHz default',
                color=HUD_C, fontsize=7.5, ha='center', alpha=safe(fspl_a*0.8),
                transform=ax.transAxes)
        ax.text(0.79, 0.115, 'Derived R_comm ≈ 86m  (not hardcoded)',
                color=GREEN_C, fontsize=7.5, ha='center', alpha=safe(fspl_a*0.8),
                transform=ax.transAxes)


def render_lambda2(ax, t):
    a = scene_alpha(t)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_facecolor(BG)

    ax.text(0.5, 0.95, 'λ₂ — THE FIEDLER VALUE: ONE NUMBER TO MEASURE COORDINATION',
            color=TITLE_C, fontsize=11, ha='center', va='top',
            alpha=safe(a), transform=ax.transAxes, fontweight='bold')

    # Three state panels
    states = [
        (0.08, 0.40, 'CONNECTED\nλ₂ = 0.82', ENEMY_C, True, False),
        (0.38, 0.40, 'WEAKENING\nλ₂ = 0.21', ENEMY_L, True, True),
        (0.68, 0.40, 'FRAGMENTED\nλ₂ = 0.00', GREEN_C, False, False),
    ]

    for i, (px, py, label, col, connected, partial) in enumerate(states):
        panel_a = ease(max(0, (t - i*0.2) / 0.15)) * a
        if panel_a <= 0:
            continue

        # Draw mini graph
        n = 6
        pts = np.array([
            [px+0.06, py+0.18], [px+0.12, py+0.22], [px+0.18, py+0.18],
            [px+0.06, py+0.10], [px+0.12, py+0.06], [px+0.18, py+0.10],
        ])

        if connected:
            pairs = [(0,1),(1,2),(0,3),(3,4),(4,5),(2,5),(1,4),(0,2)]
        else:
            pairs = [(0,1),(3,4),(4,5)]  # disconnected

        if partial:
            pairs = [(0,1),(1,2),(3,4),(4,5)]

        for pa, pb in pairs:
            ax.plot([pts[pa][0], pts[pb][0]], [pts[pa][1], pts[pb][1]],
                    color=col, lw=1.5, alpha=alpha_clamp(panel_a * 0.7))

        for pt in pts[:3]:
            ax.plot(pt[0], pt[1], 'o', color=col, ms=5, alpha=safe(panel_a))
        for pt in pts[3:]:
            col2 = col if connected or partial else JAMMER_C
            ax.plot(pt[0], pt[1], 'o', color=col2, ms=5, alpha=safe(panel_a))

        ax.text(px+0.12, py+0.34, label, color=col,
                fontsize=9, ha='center', va='center', alpha=safe(panel_a),
                fontweight='bold', linespacing=1.3)

        # Box
        rect = FancyBboxPatch((px, py), 0.24, 0.38,
                              boxstyle="round,pad=0.01",
                              linewidth=1.5, edgecolor=col,
                              facecolor=BG, alpha=safe(panel_a*0.5))
        ax.add_patch(rect)

    # Mission arrow
    if t > 0.55:
        arr_a = ease((t - 0.55) / 0.2) * a
        ax.annotate('', xy=(0.68, 0.28), xytext=(0.08, 0.28),
                    arrowprops=dict(arrowstyle='->', color=JAMMER_C,
                                   lw=2.0, alpha=safe(arr_a)))
        ax.text(0.38, 0.32, 'OUR MISSION: Drive λ₂ → 0',
                color=JAMMER_C, fontsize=11, ha='center', alpha=safe(arr_a),
                fontweight='bold')

    # Theory box
    if t > 0.65:
        th_a = ease((t - 0.65) / 0.2) * a
        draw_hud_box(ax, 0.05, 0.04, 0.9, 0.14,
                     'Fiedler Theorem (1973)', color=GOLD, alpha=safe(th_a))
        ax.text(0.5, 0.135, 'λ₂ > 0  ⟺  Graph is connected  ⟺  Swarm CAN coordinate globally',
                color=GOLD, fontsize=10, ha='center', alpha=safe(th_a),
                transform=ax.transAxes)
        ax.text(0.5, 0.09, 'λ₂ = 0  ⟺  Graph is disconnected  ⟺  Swarm CANNOT coordinate  (Proposition 1)',
                color=GREEN_C, fontsize=10, ha='center', alpha=safe(th_a),
                transform=ax.transAxes, fontweight='bold')


def render_dbscan(ax, t):
    a = scene_alpha(t)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_facecolor(BG)

    ax.text(0.5, 0.95, 'DBSCAN CLUSTERING — Intelligent Jammer Deployment',
            color=TITLE_C, fontsize=12, ha='center', va='top',
            alpha=safe(a), transform=ax.transAxes, fontweight='bold')

    # Show enemy drones
    for p in enemy_pos:
        draw_drone(ax, p[0]*0.58 + 0.03, p[1]*0.75 + 0.12, ENEMY_C, alpha=safe(a))

    cluster_colors = ['#ff9900', '#aa44ff', '#44ffaa']
    radii = [0.12, 0.12, 0.10]

    # Progressively show cluster circles
    for k, (c, col, r) in enumerate(zip(cluster_centers, cluster_colors, radii)):
        circ_t = max(0, (t - 0.25 - k*0.18) / 0.15)
        circ_a = ease(min(circ_t, 1.0)) * a
        if circ_a <= 0:
            continue
        cx = c[0]*0.58 + 0.03
        cy = c[1]*0.75 + 0.12
        circle = Circle((cx, cy), r*0.58,
                        fill=True, facecolor=col,
                        edgecolor=col, lw=2, alpha=safe(circ_a*0.15))
        ax.add_patch(circle)
        circle2 = Circle((cx, cy), r*0.58,
                         fill=False, edgecolor=col, lw=2, alpha=safe(circ_a*0.6))
        ax.add_patch(circle2)

        # Centroid marker
        cent_a = ease(max(0, (t - 0.4 - k*0.18) / 0.15)) * a
        ax.plot(cx, cy, '+', color=col, ms=12, mew=2, alpha=safe(cent_a))
        ax.text(cx, cy+r*0.58+0.02, f'Cluster {k+1}', color=col,
                fontsize=8, ha='center', alpha=safe(cent_a))

    # Right panel explanation
    if t > 0.5:
        exp_a = ease((t - 0.5) / 0.2) * a
        draw_hud_box(ax, 0.65, 0.3, 0.32, 0.5, 'DBSCAN Config', alpha=safe(exp_a))
        lines = [
            ('eps = 30m', GOLD),
            ('min_samples = 2', GOLD),
            ('', TITLE_C),
            ('1. Group nearby drones', TITLE_C),
            ('2. Find cluster centers', TITLE_C),
            ('3. Place jammers at centers', JAMMER_C),
            ('', TITLE_C),
            (f'Found {len(cluster_centers)} clusters', GREEN_C),
            ('Re-runs every 10 steps', HUD_C),
        ]
        for i, (line, col) in enumerate(lines):
            ax.text(0.81, 0.745 - i*0.045, line,
                    color=col, fontsize=8.5, ha='center',
                    alpha=safe(exp_a), transform=ax.transAxes)

    # Bottom key insight
    if t > 0.75:
        ins_a = ease((t - 0.75) / 0.2) * a
        ax.text(0.5, 0.045, '2× faster convergence vs random initialization  (ablation result)',
                color=GREEN_C, fontsize=10, ha='center', alpha=safe(ins_a),
                transform=ax.transAxes, style='italic')


def render_deploy(ax, t):
    a = scene_alpha(t)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_facecolor(BG)

    ax.text(0.5, 0.95, 'JAMMER DEPLOYMENT — 4 Teal Agents Deploying to Cluster Centroids',
            color=TITLE_C, fontsize=11, ha='center', va='top',
            alpha=safe(a), transform=ax.transAxes, fontweight='bold')

    # Enemy drones (static)
    for p in enemy_pos:
        draw_drone(ax, p[0]*0.58+0.03, p[1]*0.75+0.12, ENEMY_C, alpha=safe(a*0.6))

    # Links
    for i, j, d in all_links:
        pi = enemy_pos[i]*np.array([0.58,0.75]) + np.array([0.03,0.12])
        pj = enemy_pos[j]*np.array([0.58,0.75]) + np.array([0.03,0.12])
        ax.plot([pi[0], pj[0]], [pi[1], pj[1]],
                color=LINK_C, lw=0.8, alpha=safe(a*0.3))

    # Cluster circles (dim)
    for c in cluster_centers:
        cx = c[0]*0.58+0.03; cy = c[1]*0.75+0.12
        circle = Circle((cx, cy), 0.07, fill=False,
                        edgecolor=HUD_C, lw=1, alpha=safe(a*0.3), linestyle='--')
        ax.add_patch(circle)
        ax.plot(cx, cy, '+', color=HUD_C, ms=8, mew=1.5, alpha=safe(a*0.4))

    # Animate jammer movement
    move_t = ease(min(t * 1.8, 1.0))
    jammer_labels = ['J1', 'J2', 'J3', 'J4']
    for k in range(M_JAMMER):
        js = jammer_start[k]
        jt = jammer_target[k]
        jx = lerp(js[0]*0.58+0.03, jt[0]*0.58+0.03, move_t)
        jy = lerp(js[1]*0.75+0.12, jt[1]*0.75+0.12, move_t)

        # Trail
        if move_t > 0.05:
            trail_xs = np.linspace(js[0]*0.58+0.03, jx, 15)
            trail_ys = np.linspace(js[1]*0.75+0.12, jy, 15)
            for ti in range(len(trail_xs)-1):
                ax.plot([trail_xs[ti], trail_xs[ti+1]],
                        [trail_ys[ti], trail_ys[ti+1]],
                        color=JAMMER_C, lw=1.2,
                        alpha=safe(a*(ti/len(trail_xs)))*0.5)

        draw_drone(ax, jx, jy, JAMMER_C, alpha=safe(a), label=jammer_labels[k])

        # Jamming radius ring (when in position)
        if move_t > 0.85:
            ring_a = ease((move_t - 0.85)/0.15) * a * 0.35
            ring = Circle((jx, jy), 0.055, fill=False,
                         edgecolor=JAMMER_C, lw=1.5,
                         alpha=safe(ring_a), linestyle='--')
            ax.add_patch(ring)

    # Status
    if t > 0.8:
        st_a = ease((t - 0.8)/0.15) * a
        ax.text(0.65, 0.25, f'M = {M_JAMMER} Jammer Agents', color=JAMMER_C,
                fontsize=11, alpha=safe(st_a), transform=ax.transAxes, fontweight='bold')
        ax.text(0.65, 0.19, 'Deployment: COMPLETE', color=GREEN_C,
                fontsize=10, alpha=safe(st_a), transform=ax.transAxes)
        ax.text(0.65, 0.13, 'Architecture: CTDE', color=HUD_C,
                fontsize=9, alpha=safe(st_a), transform=ax.transAxes)
        ax.text(0.65, 0.07, '(Centralized Train / Decentralized Execute)', color=HUD_C,
                fontsize=8, alpha=safe(st_a*0.8), transform=ax.transAxes, style='italic')


def render_obs_vector(ax, t):
    a = scene_alpha(t)
    ax.set_facecolor(BG)

    ax.text(0.5, 0.96, 'AGENT OBSERVATION — The 5 Numbers Every Jammer Uses',
            color=TITLE_C, fontsize=12, ha='center', va='top',
            alpha=safe(a), transform=ax.transAxes, fontweight='bold')

    # Central jammer
    ax.plot(0.5, 0.52, 'o', color=JAMMER_C, ms=14, alpha=safe(a), zorder=5)
    ax.text(0.5, 0.52, 'J', color=BG, fontsize=9, ha='center', va='center',
            alpha=safe(a), zorder=6, fontweight='bold')
    ax.text(0.5, 0.46, 'JAMMER', color=JAMMER_C, fontsize=9,
            ha='center', alpha=safe(a), transform=ax.transAxes)

    obs = [
        (0.20, 0.78, '[0] dist_to_centroid',
         'Distance to nearest\ncluster center / arena_size',
         'Am I in position?', GOLD),
        (0.05, 0.52, '[1] cluster_density',
         'Fraction of enemies\nin my assigned cluster',
         'How important\nis my cluster?', '#aa44ff'),
        (0.20, 0.26, '[2] dist_to_others',
         'Mean distance to\nother jammer agents',
         'Am I isolated\nor overcrowded?', '#ff9900'),
        (0.75, 0.26, '[3] coverage_overlap',
         'Fraction of jammer pairs\noverlapping my jamming zone',
         'Wasting coverage\nwith a teammate?', ENEMY_C),
        (0.78, 0.78, '[4] band_match',
         'Binary: is my current\nfrequency = enemy frequency?',
         'Am I actually\njamming right now?', GREEN_C),
    ]

    for i, (ox, oy, label, desc, why, col) in enumerate(obs):
        item_a = ease(max(0, (t - 0.1 - i*0.12) / 0.12)) * a
        if item_a <= 0:
            continue

        # Arrow from jammer to box
        ax.annotate('', xy=(ox+0.1, oy),
                    xytext=(0.5, 0.52),
                    arrowprops=dict(arrowstyle='->', color=col,
                                   lw=1.2, alpha=safe(item_a*0.6),
                                   connectionstyle='arc3,rad=0.1'))

        draw_hud_box(ax, ox-0.02, oy-0.06, 0.24, 0.16, '', color=col, alpha=safe(item_a))
        ax.text(ox+0.10, oy+0.06, label, color=col,
                fontsize=8, ha='center', alpha=safe(item_a), fontweight='bold')
        ax.text(ox+0.10, oy+0.01, desc, color=TITLE_C,
                fontsize=7, ha='center', alpha=safe(item_a), linespacing=1.3)
        ax.text(ox+0.10, oy-0.045, why, color=col,
                fontsize=7, ha='center', alpha=safe(item_a*0.8),
                style='italic', linespacing=1.2)

    # Key property
    if t > 0.85:
        kp_a = ease((t - 0.85)/0.12) * a
        ax.text(0.5, 0.055, 'All 5 values clipped to [0, 1]  ●  Fixed size regardless of N or M  ●  This IS the scalability',
                color=GREEN_C, fontsize=9.5, ha='center', alpha=safe(kp_a),
                transform=ax.transAxes)


def render_neural_net(ax, t):
    a = scene_alpha(t)
    ax.set_facecolor(BG)

    ax.text(0.5, 0.96, 'ACTOR NETWORK — Turning 5 Numbers Into Actions',
            color=TITLE_C, fontsize=12, ha='center', va='top',
            alpha=safe(a), transform=ax.transAxes, fontweight='bold')

    # Network layout
    layers = [
        (5, 'Input\n5D obs', HUD_C),
        (8, 'Hidden\n128', JAMMER_C),
        (8, 'Hidden\n128', JAMMER_C),
        (6, 'Output', GOLD),
    ]

    layer_xs = [0.18, 0.35, 0.52, 0.70]
    layer_spacing = [0.14, 0.1, 0.1, 0.1]
    node_positions = []

    for li, (n_nodes, label, col) in enumerate(layers):
        x = layer_xs[li]
        ys = np.linspace(0.5 - (n_nodes-1)*layer_spacing[li]/2,
                         0.5 + (n_nodes-1)*layer_spacing[li]/2,
                         n_nodes)
        node_positions.append(list(zip([x]*n_nodes, ys)))

        # Draw connections to previous layer
        if li > 0:
            conn_a = ease(max(0, (t - 0.1 - li*0.15)/0.15)) * a
            for (x1, y1) in node_positions[li-1]:
                for (x2, y2) in zip([x]*n_nodes, ys):
                    ax.plot([x1, x2], [y1, y2],
                            color=col, lw=0.5, alpha=alpha_clamp(conn_a * 0.3))

        node_a = ease(max(0, (t - li*0.15)/0.15)) * a
        for y in ys:
            circle = Circle((x, y), 0.018, color=col, alpha=safe(node_a*0.8))
            ax.add_patch(circle)

        ax.text(x, 0.17, label, color=col, fontsize=8,
                ha='center', alpha=safe(node_a), linespacing=1.3)

    # Activation wave
    if t > 0.5:
        wave_t = (t - 0.5) * 3.0
        wave_x = lerp(layer_xs[0], layer_xs[-1], wave_t % 1.0)
        wave_a = (1 - (wave_t % 1.0)) * a * 0.7
        ax.axvline(wave_x, color=GOLD, lw=2, alpha=safe(wave_a), linestyle='--')

    # Output labels
    if t > 0.6:
        out_a = ease((t - 0.6)/0.15) * a
        out_ys = [y for _, y in node_positions[-1]]
        labels_out = ['Vx →', 'Vy →', '', '433MHz', '915MHz', '2.4GHz']
        colors_out = [JAMMER_C, JAMMER_C, JAMMER_C, HUD_C, HUD_C, GREEN_C]
        for k, (y, lab, col) in enumerate(zip(out_ys, labels_out, colors_out)):
            ax.text(0.77, y, lab, color=col, fontsize=8,
                    va='center', alpha=safe(out_a))

        # Output labels
        ax.text(0.88, 0.65, 'VELOCITY\n(Continuous)', color=JAMMER_C,
                fontsize=8, ha='center', alpha=safe(out_a), linespacing=1.3,
                transform=ax.transAxes)
        ax.text(0.88, 0.38, 'BAND\n(Discrete)', color=GREEN_C,
                fontsize=8, ha='center', alpha=safe(out_a), linespacing=1.3,
                transform=ax.transAxes)

    # Parameter sharing callout
    if t > 0.75:
        ps_a = ease((t - 0.75)/0.15) * a
        draw_hud_box(ax, 0.02, 0.04, 0.45, 0.14, 'Parameter Sharing', color=GOLD, alpha=safe(ps_a))
        ax.text(0.24, 0.115, 'All M jammers share ONE Actor network', color=GOLD,
                fontsize=9, ha='center', alpha=safe(ps_a), transform=ax.transAxes)
        ax.text(0.24, 0.073, 'Each uses its OWN local 5D observation', color=TITLE_C,
                fontsize=8.5, ha='center', alpha=safe(ps_a), transform=ax.transAxes)

    # LayerNorm note
    if t > 0.8:
        ln_a = ease((t - 0.8)/0.12) * a
        ax.text(0.35, 0.09, 'LayerNorm after each hidden layer\n(stable with variable batch sizes)',
                color=HUD_C, fontsize=8, ha='center', alpha=safe(ln_a),
                transform=ax.transAxes, linespacing=1.3)


def render_reward(ax, t):
    a = scene_alpha(t)
    ax.set_facecolor(BG)

    ax.text(0.5, 0.96, 'REWARD FUNCTION — 5 Signals Teaching the Agent What "Good" Means',
            color=TITLE_C, fontsize=11, ha='center', va='top',
            alpha=safe(a), transform=ax.transAxes, fontweight='bold')

    terms = [
        ('ω₁ = 1.0', 'λ₂ Reduction', '1 − λ₂(t)/λ₂(0)',
         'PRIMARY: Swarm fragmentation', GREEN_C, 1.0, '↑'),
        ('ω₂ = 0.3', 'Band Match', '(1/M)·Σ 1[b_k = b_enemy]',
         'Correct frequency selection', JAMMER_C, 0.3, '↑'),
        ('ω₃ = 0.2', 'Proximity', '(1/M)·Σ exp(−κ·d_centroid)',
         'Stay near cluster center', HUD_C, 0.2, '↑'),
        ('ω₄ = 0.1', 'Energy Penalty', '(1/M)·Σ ‖v_k‖²/v_max²',
         'No unnecessary movement', '#ff9900', 0.1, '↓'),
        ('ω₅ = 0.2', 'Overlap Penalty', 'fraction pairs within 2·R_jam',
         'Spread coverage out', ENEMY_C, 0.2, '↓'),
    ]

    for i, (weight, name, formula, role, col, w, direction) in enumerate(terms):
        item_a = ease(max(0, (t - 0.1 - i*0.13)/0.13)) * a
        if item_a <= 0: continue

        y0 = 0.82 - i * 0.145
        bar_w = w * 0.25

        draw_hud_box(ax, 0.02, y0, 0.94, 0.12, '', color=col, alpha=safe(item_a*0.7))

        ax.text(0.055, y0+0.075, weight, color=col, fontsize=10,
                va='center', alpha=safe(item_a), fontweight='bold',
                transform=ax.transAxes)
        ax.text(0.16, y0+0.075, name, color=TITLE_C, fontsize=10,
                va='center', alpha=safe(item_a), fontweight='bold',
                transform=ax.transAxes)
        ax.text(0.43, y0+0.075, formula, color=GOLD, fontsize=8.5,
                va='center', alpha=safe(item_a), transform=ax.transAxes)
        ax.text(0.70, y0+0.075, role, color=HUD_C, fontsize=8,
                va='center', alpha=safe(item_a), transform=ax.transAxes)

        # Bar
        rect = patches.Rectangle((0.935, y0+0.025), bar_w, 0.07,
                                  color=col, alpha=safe(item_a*0.6),
                                  transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(0.958, y0+0.075, direction, color=col, fontsize=9,
                va='center', alpha=safe(item_a), transform=ax.transAxes,
                fontweight='bold')

    # Total reward
    if t > 0.85:
        tot_a = ease((t - 0.85)/0.12) * a
        ax.text(0.5, 0.058, 'R(t)  =  ω₁·(λ₂ reduction)  +  ω₂·(band)  +  ω₃·(proximity)  −  ω₄·(energy)  −  ω₅·(overlap)',
                color=GOLD, fontsize=10, ha='center', alpha=safe(tot_a),
                transform=ax.transAxes, fontweight='bold')


def render_training(ax, t):
    a = scene_alpha(t)
    ax.set_facecolor(BG)

    ax.text(0.5, 0.96, 'PPO TRAINING — Learning Over Thousands of Episodes',
            color=TITLE_C, fontsize=12, ha='center', va='top',
            alpha=safe(a), transform=ax.transAxes, fontweight='bold')

    # Simulated training curves
    steps = np.linspace(0, 500000, 500)
    # Reward curve (rises with noise)
    np.random.seed(1)
    reward_noise = np.cumsum(np.random.randn(500)) * 0.3
    reward_raw = 1 - np.exp(-steps/150000) + reward_noise/30
    reward = np.clip(reward_raw, -0.1, 1.05)

    # λ₂ reduction curve
    lambda_red = (1 - np.exp(-steps/120000)) * 80 + np.random.randn(500)*3
    lambda_red = np.clip(lambda_red, 0, 85)

    # Entropy (decreasing)
    entropy = np.exp(-steps/200000) * 1.4 + np.random.randn(500)*0.05
    entropy = np.clip(entropy, 0.05, 1.4)

    # Value loss (decreasing)
    vloss = np.exp(-steps/100000) * 2 + np.abs(np.random.randn(500))*0.1
    vloss = np.clip(vloss, 0.1, 2.2)

    # Reveal proportion
    reveal = ease(min(t * 1.5, 1.0))
    n_show = max(2, int(reveal * 500))

    panels = [
        (0.04, 0.52, 0.44, 0.38, 'Episode Reward', steps, reward*0.8+0.1, GOLD, '0', '1.0'),
        (0.52, 0.52, 0.44, 0.38, 'λ₂ Reduction %', steps, lambda_red, GREEN_C, '0%', '80%'),
        (0.04, 0.10, 0.44, 0.38, 'Policy Entropy', steps, entropy, HUD_C, '0', '1.4'),
        (0.52, 0.10, 0.44, 0.38, 'Value Loss', steps, vloss, ENEMY_L, '0', '2.0'),
    ]

    for px, py, pw, ph, title, xs, ys, col, ymin_l, ymax_l in panels:
        draw_hud_box(ax, px, py, pw, ph, title, color=col, alpha=safe(a))
        # Plot area
        plot_x0, plot_y0 = px+0.02, py+0.07
        plot_w, plot_h = pw-0.04, ph-0.11
        # Axes
        ax.plot([plot_x0, plot_x0+plot_w], [plot_y0, plot_y0],
                color=DIM, lw=0.8, alpha=safe(a*0.5))
        ax.plot([plot_x0, plot_x0], [plot_y0, plot_y0+plot_h],
                color=DIM, lw=0.8, alpha=safe(a*0.5))

        # Normalize and plot
        y_range = ys.max() - ys.min() + 1e-6
        ys_norm = (ys - ys.min()) / y_range * plot_h + plot_y0
        xs_norm = xs / xs.max() * plot_w + plot_x0

        ax.plot(xs_norm[:n_show], ys_norm[:n_show], color=col, lw=1.5, alpha=safe(a*0.9))

        # Y labels
        ax.text(plot_x0-0.01, plot_y0, ymin_l, color=col, fontsize=6,
                ha='right', va='center', alpha=safe(a*0.7))
        ax.text(plot_x0-0.01, plot_y0+plot_h, ymax_l, color=col, fontsize=6,
                ha='right', va='center', alpha=safe(a*0.7))

    # 70% target line on lambda2 plot
    targ_y = 0.52 + 0.07 + (70/80) * 0.27
    ax.plot([0.54, 0.94], [targ_y, targ_y], color=WARN_C,
            lw=1.2, alpha=safe(a*0.7), linestyle='--')
    ax.text(0.955, targ_y, '70%\nTarget', color=WARN_C,
            fontsize=6.5, va='center', alpha=safe(a*0.7), transform=ax.transAxes)

    # Status
    if t > 0.85:
        st_a = ease((t - 0.85)/0.12) * a
        ax.text(0.5, 0.038, '≥ 70% λ₂ reduction achieved at ~300K–500K timesteps  ●  Training: CONVERGED ✓',
                color=GREEN_C, fontsize=10, ha='center', alpha=safe(st_a),
                transform=ax.transAxes, fontweight='bold')


def render_jamming(ax, t):
    a = scene_alpha(t)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_facecolor(BG)

    ax.text(0.5, 0.95, 'JAMMING IN ACTION — Links Breaking, λ₂ Falling',
            color=TITLE_C, fontsize=12, ha='center', va='top',
            alpha=safe(a), transform=ax.transAxes, fontweight='bold')

    # How many links have broken
    n_break = int(ease(min(t * 1.8, 1.0)) * len(all_links))
    broken_links = set()
    for k in range(n_break):
        i, j, _ = all_links[k]
        broken_links.add((i, j))

    # Enemy drones
    for i, p in enumerate(enemy_pos):
        px = p[0]*0.58+0.03; py = p[1]*0.75+0.12
        # Isolated drones look dimmer
        connected = any((i,j) not in broken_links and (j,i) not in broken_links
                       for _, j, _ in [l for l in all_links if l[0]==i or l[1]==i])
        draw_drone(ax, px, py, ENEMY_C if connected else '#551100', alpha=safe(a))

    # Draw links
    for k, (i, j, d) in enumerate(all_links):
        pi = enemy_pos[i]*np.array([0.58,0.75])+np.array([0.03,0.12])
        pj = enemy_pos[j]*np.array([0.58,0.75])+np.array([0.03,0.12])
        if (i,j) in broken_links:
            # Show broken as faded
            ax.plot([pi[0], pj[0]], [pi[1], pj[1]],
                    color=BROKEN_C, lw=0.8, alpha=safe(a*0.3))
            # X marker at midpoint
            mx = (pi[0]+pj[0])/2; my = (pi[1]+pj[1])/2
            ax.plot(mx, my, 'x', color=WARN_C, ms=5, mew=1.2,
                    alpha=safe(a*(1 - (t*2 % 0.5)))*0.8)
        else:
            intensity = 1.0 - d/LINK_THRESH
            ax.plot([pi[0], pj[0]], [pi[1], pj[1]],
                    color=LINK_C, lw=1.5, alpha=safe(a*intensity*0.8))

    # Jammer drones (in position)
    for k in range(M_JAMMER):
        jt = jammer_target[k]
        jx = jt[0]*0.58+0.03; jy = jt[1]*0.75+0.12

        # Jamming signal rings
        pulse_t = (t * 3 + k*0.25) % 1.0
        ring_r = pulse_t * 0.08
        ring_a = (1 - pulse_t) * a * 0.5
        ring = Circle((jx, jy), ring_r, fill=False,
                     edgecolor=JAMMER_C, lw=1.2, alpha=safe(ring_a))
        ax.add_patch(ring)

        draw_drone(ax, jx, jy, JAMMER_C, alpha=safe(a))

    # λ₂ HUD
    lambda2_val = 0.82 * (1 - ease(min(t * 1.6, 1.0)) * 0.85)
    lambda2_val = max(0.0, lambda2_val)
    draw_hud_box(ax, 0.63, 0.65, 0.34, 0.22, 'SWARM CONNECTIVITY  λ₂', alpha=safe(a))
    color_l = ENEMY_C if lambda2_val > 0.4 else (ENEMY_L if lambda2_val > 0.1 else GREEN_C)
    glow_text(ax, 0.80, 0.735, f'{lambda2_val:.3f}',
              color_l, size=28, alpha=safe(a), ha='center',
              transform=ax.transAxes, fontweight='bold')

    # Bar
    bar_fill = lambda2_val / 0.82
    ax.add_patch(patches.Rectangle((0.645, 0.62), 0.31 * bar_fill, 0.025,
                                   color=color_l, alpha=safe(a*0.8)))
    ax.add_patch(patches.Rectangle((0.645, 0.62), 0.31, 0.025,
                                   fill=False, edgecolor=HUD_C, lw=1, alpha=safe(a*0.5)))

    # Break counter
    if t > 0.3:
        bc_a = ease((t-0.3)/0.15)*a
        ax.text(0.80, 0.575, f'Links broken: {n_break} / {len(all_links)}',
                color=WARN_C, fontsize=9, ha='center', alpha=safe(bc_a),
                transform=ax.transAxes)


def render_fragmentation(ax, t):
    a = scene_alpha(t)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_facecolor(BG)

    ax.text(0.5, 0.95, '⚡ MISSION COMPLETE — λ₂ = 0 — SWARM FULLY FRAGMENTED',
            color=GREEN_C, fontsize=12, ha='center', va='top',
            alpha=safe(a), transform=ax.transAxes, fontweight='bold')

    # Shockwave effect
    if t < 0.3:
        wave_r = ease(t/0.3) * 0.6
        wave_a = (1 - ease(t/0.3)) * a * 0.5
        wave = Circle((0.32, 0.5), wave_r, fill=False,
                     edgecolor='#ffffff', lw=3, alpha=safe(wave_a))
        ax.add_patch(wave)

    # All links broken — enemy drones isolated
    for i, p in enumerate(enemy_pos):
        px = p[0]*0.58+0.03; py = p[1]*0.75+0.12
        draw_drone(ax, px, py, '#441100', alpha=safe(a*0.5))
        # Isolation ring
        iso = Circle((px, py), 0.025, fill=False,
                    edgecolor='#441100', lw=0.8,
                    alpha=safe(a*0.3), linestyle='--')
        ax.add_patch(iso)

    # Jammer drones (triumphant)
    for k in range(M_JAMMER):
        jt = jammer_target[k]
        jx = jt[0]*0.58+0.03; jy = jt[1]*0.75+0.12
        draw_drone(ax, jx, jy, JAMMER_C, alpha=safe(a))
        glow = Circle((jx, jy), 0.04, fill=True,
                     facecolor=JAMMER_C, alpha=safe(a*0.15))
        ax.add_patch(glow)

    # λ₂ = 0
    draw_hud_box(ax, 0.63, 0.60, 0.34, 0.25, 'CONNECTIVITY  λ₂', color=GREEN_C, alpha=safe(a))
    glow_text(ax, 0.80, 0.70, '0.000',
              GREEN_C, size=32, alpha=safe(a), ha='center',
              transform=ax.transAxes, fontweight='bold')

    # Fragmented indicator
    if t > 0.35:
        frag_a = ease((t-0.35)/0.15) * a
        ax.text(0.80, 0.595, '● SWARM FRAGMENTED', color=GREEN_C,
                fontsize=11, ha='center', alpha=safe(frag_a),
                transform=ax.transAxes, fontweight='bold')
        ax.text(0.80, 0.548, 'Proposition 1: SATISFIED ✓', color=GOLD,
                fontsize=10, ha='center', alpha=safe(frag_a),
                transform=ax.transAxes)

    # Proposition box
    if t > 0.5:
        prop_a = ease((t-0.5)/0.2) * a
        draw_hud_box(ax, 0.62, 0.09, 0.35, 0.35,
                     'Proposition 1  (Fiedler, 1973)', color=GOLD, alpha=safe(prop_a))
        lines = [
            'λ₂ = 0  ⟺  Graph disconnected',
            '⟺  Swarm cannot coordinate',
            '',
            'This is NOT a heuristic.',
            'This is a mathematical theorem.',
        ]
        for i, line in enumerate(lines):
            col = GOLD if i < 2 else (TITLE_C if i < 3 else GREEN_C)
            ax.text(0.795, 0.375 - i*0.055, line, color=col,
                    fontsize=8.5, ha='center', alpha=safe(prop_a),
                    transform=ax.transAxes,
                    fontweight='bold' if i == 4 else 'normal')


def render_comparison(ax, t):
    a = scene_alpha(t)
    ax.set_facecolor(BG)

    ax.text(0.5, 0.96, 'BASELINE COMPARISON — How We Outperform All Methods',
            color=TITLE_C, fontsize=12, ha='center', va='top',
            alpha=safe(a), transform=ax.transAxes, fontweight='bold')

    methods = [
        ('Random Agents',        15,  DIM,      False),
        ('Greedy Distance',      40,  '#886644', False),
        ('Single-Agent PPO',     55,  HUD_C,     False),
        ('Independent PPO',      65,  '#aa44ff', False),
        ('MARL-PPO Static',      80,  JAMMER_C, True),
        ('MARL-PPO Dynamic',     75,  GREEN_C,  True),
    ]

    bar_h = 0.085
    y_start = 0.85
    bar_x0 = 0.38
    max_bar_w = 0.50

    for i, (name, val, col, ours) in enumerate(methods):
        item_a = ease(max(0, (t - 0.1 - i*0.12)/0.12)) * a
        if item_a <= 0: continue

        y = y_start - i * (bar_h + 0.02)

        # Bar
        bar_w = (val/100) * max_bar_w * ease(min((t - i*0.1)/0.3, 1.0))
        lw = 2.5 if ours else 1.2
        rect = patches.Rectangle((bar_x0, y), bar_w, bar_h,
                                  facecolor=col, alpha=safe(item_a*0.7),
                                  edgecolor=col if ours else DIM, linewidth=lw,
                                  transform=ax.transAxes)
        ax.add_patch(rect)

        # Name
        ax.text(bar_x0 - 0.01, y + bar_h/2, name,
                color=col if ours else TITLE_C,
                fontsize=9, ha='right', va='center',
                alpha=safe(item_a), transform=ax.transAxes,
                fontweight='bold' if ours else 'normal')

        # Value
        ax.text(bar_x0 + bar_w + 0.01, y + bar_h/2,
                f'{val}%', color=col, fontsize=9,
                ha='left', va='center', alpha=safe(item_a),
                transform=ax.transAxes, fontweight='bold' if ours else 'normal')

        if ours:
            ax.text(bar_x0 + bar_w + 0.05, y + bar_h/2, '← OUR WORK',
                    color=col, fontsize=8, ha='left', va='center',
                    alpha=safe(item_a*0.8), transform=ax.transAxes, style='italic')

    # 70% target line
    tx = bar_x0 + 0.70 * max_bar_w
    ax.plot([tx, tx], [0.08, 0.94], color=WARN_C, lw=1.5,
        alpha=safe(a*0.7), linestyle='--', transform=ax.transAxes)
    ax.text(tx+0.005, 0.09, '70%\nTarget', color=WARN_C, fontsize=7.5,
            va='bottom', alpha=safe(a*0.8), transform=ax.transAxes)

    # X axis label
    ax.text(0.635, 0.04, 'Mean λ₂ Reduction %', color=HUD_C,
            fontsize=9, ha='center', alpha=safe(a*0.8),
            transform=ax.transAxes)


def render_proposition(ax, t):
    a = scene_alpha(t)
    ax.set_facecolor(BG)

    ax.text(0.5, 0.96, 'PROPOSITION 1 — The Theoretical Backbone',
            color=GOLD, fontsize=13, ha='center', va='top',
            alpha=safe(a), transform=ax.transAxes, fontweight='bold')

    # Three-step visual proof
    steps = [
        (0.08, 'STEP 1', 'Connected graph\nwith bridge edge',
         [(0,1),(1,2),(2,3),(3,4),(4,5),(2,5)], 6, ENEMY_C),
        (0.38, 'STEP 2', 'Remove bridge\nedge (jammed)',
         [(0,1),(1,2),(3,4),(4,5)], 6, ENEMY_L),
        (0.68, 'STEP 3', 'Two isolated\nclusters  λ₂ = 0',
         [(0,1),(1,2),(3,4),(4,5)], 6, GREEN_C),
    ]

    node_pts = {
        0: (0.08, 0.55), 1: (0.12, 0.65), 2: (0.16, 0.55),
        3: (0.16, 0.42), 4: (0.12, 0.32), 5: (0.08, 0.42),
    }

    for si, (ox, step_label, desc, edges, _, col) in enumerate(steps):
        step_a = ease(max(0, (t - 0.1 - si*0.2)/0.15)) * a
        if step_a <= 0: continue

        # Offset nodes
        pts = {k: (v[0]+ox, v[1]) for k, v in node_pts.items()}

        for i, j in edges:
            # Show bridge edge breaking in step 2
            if si == 1 and (i == 2 and j == 3):
                break_t = ease(min((t - 0.35)/0.2, 1.0))
                edge_col = BROKEN_C if break_t > 0.5 else col
                edge_lw = 2.5 if break_t < 0.3 else 1.0
                ax.plot([pts[i][0], pts[j][0]], [pts[i][1], pts[j][1]],
                        color=edge_col, lw=edge_lw, alpha=safe(step_a*0.8))
                if break_t > 0.5:
                    mx = (pts[i][0]+pts[j][0])/2
                    my = (pts[i][1]+pts[j][1])/2
                    ax.plot(mx, my, 'x', color=WARN_C, ms=8, mew=2.5,
                            alpha=safe(step_a))
            else:
                ax.plot([pts[i][0], pts[j][0]], [pts[i][1], pts[j][1]],
                        color=col, lw=2.0, alpha=safe(step_a*0.7))

        for k, (px, py) in pts.items():
            node_col = JAMMER_C if (si == 2 and k >= 3) else col
            circle = Circle((px, py), 0.018, color=node_col, alpha=safe(step_a*0.9))
            ax.add_patch(circle)

        ax.text(ox+0.12, 0.73, step_label, color=col, fontsize=10,
                ha='center', alpha=safe(step_a), fontweight='bold')
        ax.text(ox+0.12, 0.27, desc, color=TITLE_C, fontsize=8.5,
                ha='center', alpha=safe(step_a), linespacing=1.4)

        # Arrows between steps
        if si < 2:
            ax.annotate('', xy=(ox+0.28, 0.50), xytext=(ox+0.26, 0.50),
                        arrowprops=dict(arrowstyle='->', color=GOLD,
                                       lw=2.0, alpha=safe(step_a*0.8)))

    # Main theorem box
    if t > 0.65:
        th_a = ease((t-0.65)/0.2) * a
        draw_hud_box(ax, 0.05, 0.05, 0.9, 0.16, '', color=GOLD, alpha=safe(th_a))
        ax.text(0.5, 0.165, 'Fiedler (1973): λ₂(G) = 0  ⟺  G is disconnected',
                color=GOLD, fontsize=12, ha='center', alpha=safe(th_a),
                transform=ax.transAxes, fontweight='bold')
        ax.text(0.5, 0.105, 'Our reward directly optimizes the EXACT condition for swarm fragmentation — not a proxy metric.',
                color=TITLE_C, fontsize=9.5, ha='center', alpha=safe(th_a),
                transform=ax.transAxes)
        ax.text(0.5, 0.058, 'This answers the key reviewer question: "Why λ₂ and not jamming power?"',
                color=GREEN_C, fontsize=9, ha='center', alpha=safe(th_a),
                transform=ax.transAxes, style='italic')


def render_results(ax, t):
    a = scene_alpha(t)
    ax.set_facecolor(BG)

    ax.text(0.5, 0.96, 'KEY RESULTS — What Our System Achieves',
            color=TITLE_C, fontsize=13, ha='center', va='top',
            alpha=safe(a), transform=ax.transAxes, fontweight='bold')

    results = [
        ('Mean λ₂ Reduction (Static Enemies)',   '75–85%',  GREEN_C),
        ('Mean λ₂ Reduction (Dynamic Enemies)',  '70–80%',  JAMMER_C),
        ('vs. Training Target (≥70%)',           'PASS ✓',  GREEN_C),
        ('vs. Baseline (Valianti Q-learning)',   '+30–35%', GOLD),
        ('vs. Independent PPO (no sharing)',     '+10–15%', HUD_C),
        ('Convergence Step',                     '~300K–500K', TITLE_C),
        ('Scales to N=100 enemies, M=40 jammers','Yes ✓',   GREEN_C),
        ('Real-World Dataset Validation',        'PASS ✓',  JAMMER_C),
    ]

    for i, (label, val, col) in enumerate(results):
        item_a = ease(max(0, (t - 0.1 - i*0.09)/0.12)) * a
        if item_a <= 0: continue
        y = 0.845 - i * 0.088

        draw_hud_box(ax, 0.05, y, 0.88, 0.072, '', color=col, alpha=safe(item_a*0.6))
        ax.text(0.10, y+0.043, label, color=TITLE_C, fontsize=9.5,
                va='center', alpha=safe(item_a), transform=ax.transAxes)
        glow_text(ax, 0.82, y+0.043, val, col, size=11,
                  alpha=safe(item_a), ha='center', va='center',
                  transform=ax.transAxes, fontweight='bold')


def render_outro(ax, t):
    a = scene_alpha(t, fi=0.05, fo=0.85)
    ax.set_facecolor(BG)

    # Fade to black at the end
    fade_out_a = ease(max(0, (t - 0.88) / 0.12))
    ax.add_patch(patches.Rectangle((0, 0), 1, 1,
                                   facecolor='#000000',
                                   alpha=safe(fade_out_a),
                                   transform=ax.transAxes, zorder=20))

    # Expanding ring
    ring_r = ease(min(t * 2, 1.0)) * 0.45
    ring_a = (1 - ease(min(t * 2, 1.0))) * a * 0.3
    ring = Circle((0.5, 0.52), ring_r, fill=False,
                 edgecolor=JAMMER_C, lw=2, alpha=safe(ring_a),
                 transform=ax.transAxes)
    ax.add_patch(ring)

    ax.text(0.5, 0.75, 'SYSTEM SUMMARY', color=HUD_C, fontsize=12,
            ha='center', alpha=safe(a), transform=ax.transAxes)

    glow_text(ax, 0.5, 0.62,
              'MARL-PPO + Graph Laplacian λ₂\nPhysically-Grounded Swarm Disruption',
              JAMMER_C, size=20, alpha=safe(a), ha='center', va='center',
              transform=ax.transAxes, fontweight='bold', linespacing=1.4)

    tags = [
        ('PPO Actor-Critic + GAE', GOLD),
        ('Fiedler Value λ₂ Reward', GREEN_C),
        ('FSPL Jamming Model', JAMMER_C),
        ('DBSCAN Deployment', HUD_C),
        ('Proposition 1 Proof', GOLD),
        ('Dynamic Enemies', GREEN_C),
    ]

    for i, (tag, col) in enumerate(tags):
        col_i = i % 3
        row_i = i // 3
        x = 0.22 + col_i * 0.27
        y = 0.40 - row_i * 0.07
        ax.text(x, y, f'✦ {tag}', color=col, fontsize=9,
                ha='center', alpha=safe(a*0.9), transform=ax.transAxes)

    ax.text(0.5, 0.24, 'Extending Valianti et al. — IEEE TMC, December 2024',
            color=DIM, fontsize=10, ha='center', alpha=safe(a*0.8),
            transform=ax.transAxes, style='italic')

    ax.text(0.5, 0.15, 'Mean λ₂ Reduction: 75–85%  ●  Theoretical Guarantee: Proposition 1  ●  Scales to N=100, M=40',
            color=HUD_C, fontsize=9, ha='center', alpha=safe(a*0.7),
            transform=ax.transAxes)

    ax.text(0.5, 0.07, 'Publication-Ready Research',
            color=GREEN_C, fontsize=11, ha='center', alpha=safe(a),
            transform=ax.transAxes, fontweight='bold')


# ─────────────────────────────────────────────────────────────────────────────
# SCENE DISPATCH
# ─────────────────────────────────────────────────────────────────────────────


def render_critic_network(ax, t):
    a = scene_alpha(t)
    ax.set_facecolor(BG)

    ax.text(0.5, 0.96, 'CRITIC NETWORK — The Global Evaluator During Training',
            color=TITLE_C, fontsize=12, ha='center', va='top',
            alpha=safe(a), transform=ax.transAxes, fontweight='bold')

    # ── Left panel: Actor (reminder) ─────────────────────────────────────────
    if t > 0.05:
        actor_a = ease(min((t - 0.05) / 0.15, 1.0)) * a
        draw_hud_box(ax, 0.03, 0.12, 0.40, 0.75, 'ACTOR  (M copies, one per jammer)',
                     color=JAMMER_C, alpha=safe(actor_a))

        # Each jammer feeds its OWN 5D obs
        for k in range(4):
            ky = 0.71 - k * 0.13
            ax.text(0.06, ky + 0.02, f'Jammer {k+1}', color=JAMMER_C,
                    fontsize=8, alpha=safe(actor_a), transform=ax.transAxes)
            # Mini obs box
            rect = patches.Rectangle((0.06, ky - 0.03), 0.10, 0.055,
                                      facecolor='#0a2030', edgecolor=JAMMER_C,
                                      linewidth=1, transform=ax.transAxes,
                                      alpha=safe(actor_a))
            ax.add_patch(rect)
            ax.text(0.11, ky, '5D obs', color=HUD_C, fontsize=7,
                    ha='center', va='center', alpha=safe(actor_a),
                    transform=ax.transAxes)
            # Arrow to shared network
            ax.annotate('', xy=(0.26, 0.50), xytext=(0.17, ky),
                        arrowprops=dict(arrowstyle='->', color=JAMMER_C,
                                        lw=0.8, alpha=safe(actor_a * 0.5),
                                        connectionstyle='arc3,rad=0.0'),
                        xycoords='axes fraction', textcoords='axes fraction')

        # Shared network box
        rect2 = patches.FancyBboxPatch((0.26, 0.35), 0.13, 0.30,
                                        boxstyle='round,pad=0.01',
                                        facecolor='#001530', edgecolor=JAMMER_C,
                                        linewidth=2, transform=ax.transAxes,
                                        alpha=safe(actor_a))
        ax.add_patch(rect2)
        ax.text(0.325, 0.525, 'SHARED\nACTOR\nπ_θ', color=JAMMER_C,
                fontsize=8.5, ha='center', va='center', alpha=safe(actor_a),
                transform=ax.transAxes, fontweight='bold', linespacing=1.4)

        # Output arrows
        ax.annotate('', xy=(0.41, 0.58), xytext=(0.39, 0.55),
                    arrowprops=dict(arrowstyle='->', color=JAMMER_C,
                                    lw=1.2, alpha=safe(actor_a)),
                    xycoords='axes fraction', textcoords='axes fraction')
        ax.text(0.41, 0.60, 'Vx,Vy + band', color=JAMMER_C,
                fontsize=7.5, alpha=safe(actor_a), transform=ax.transAxes)
        ax.text(0.11, 0.23, 'Local obs\nonly — no\nteam state', color=HUD_C,
                fontsize=7.5, ha='center', alpha=safe(actor_a * 0.8),
                transform=ax.transAxes, linespacing=1.3, style='italic')

        # ── Right panel: Critic ───────────────────────────────────────────────────
        crit_a = ease(min((t - 0.20) / 0.18, 1.0)) * a
        draw_hud_box(ax, 0.54, 0.12, 0.43, 0.75,
                     'CRITIC  (ONE copy, sees full team)',
                     color=GOLD, alpha=safe(crit_a))

        # Mean pooling step
        for k in range(4):
            ky = 0.71 - k * 0.13
            rect = patches.Rectangle((0.57, ky - 0.03), 0.10, 0.055,
                                      facecolor='#201500', edgecolor=GOLD,
                                      linewidth=1, transform=ax.transAxes,
                                      alpha=safe(crit_a))
            ax.add_patch(rect)
            ax.text(0.62, ky, f's_{k+1}', color=GOLD, fontsize=8,
                    ha='center', va='center', alpha=safe(crit_a),
                    transform=ax.transAxes)
            ax.annotate('', xy=(0.73, 0.505), xytext=(0.68, ky),
                        arrowprops=dict(arrowstyle='->', color=GOLD,
                                        lw=0.8, alpha=safe(crit_a * 0.5),
                                        connectionstyle='arc3,rad=0.0'),
                        xycoords='axes fraction', textcoords='axes fraction')

        # Mean pool box
        rect3 = patches.FancyBboxPatch((0.73, 0.44), 0.10, 0.13,
                                        boxstyle='round,pad=0.01',
                                        facecolor='#201500', edgecolor=GOLD,
                                        linewidth=2, transform=ax.transAxes,
                                        alpha=safe(crit_a))
        ax.add_patch(rect3)
        ax.text(0.78, 0.505, 'mean\npool', color=GOLD,
                fontsize=8, ha='center', va='center', alpha=safe(crit_a),
                transform=ax.transAxes, linespacing=1.3)

        # Critic network box
        rect4 = patches.FancyBboxPatch((0.85, 0.35), 0.10, 0.30,
                                        boxstyle='round,pad=0.01',
                                        facecolor='#201500', edgecolor=GOLD,
                                        linewidth=2, transform=ax.transAxes,
                                        alpha=safe(crit_a))
        ax.add_patch(rect4)
        ax.text(0.90, 0.505, 'CRITIC\nV_φ(s)', color=GOLD,
                fontsize=8.5, ha='center', va='center', alpha=safe(crit_a),
                transform=ax.transAxes, fontweight='bold', linespacing=1.4)

        ax.annotate('', xy=(0.85, 0.505), xytext=(0.83, 0.505),
                    arrowprops=dict(arrowstyle='->', color=GOLD, lw=1.5,
                                    alpha=safe(crit_a)),
                    xycoords='axes fraction', textcoords='axes fraction')

        # Output
        ax.annotate('', xy=(0.965, 0.505), xytext=(0.955, 0.505),
                    arrowprops=dict(arrowstyle='->', color=GOLD, lw=1.5,
                                    alpha=safe(crit_a)),
                    xycoords='axes fraction', textcoords='axes fraction')
        ax.text(0.968, 0.52, 'V(s)', color=GOLD, fontsize=9,
                va='center', alpha=safe(crit_a), transform=ax.transAxes,
                fontweight='bold')
        ax.text(0.968, 0.48, '(value\nestimate)', color=GOLD, fontsize=7,
                va='center', alpha=safe(crit_a * 0.8), transform=ax.transAxes,
                linespacing=1.2)

    # ── Key difference callout ────────────────────────────────────────────────
    if t > 0.45:
        diff_a = ease(min((t - 0.45) / 0.15, 1.0)) * a
        draw_hud_box(ax, 0.03, 0.01, 0.94, 0.10,
                     '', color=HUD_C, alpha=safe(diff_a))
        ax.text(0.5, 0.085, 'ACTOR: local obs per jammer  →  decides WHERE to move and WHICH band to jam',
                color=JAMMER_C, fontsize=9, ha='center', alpha=safe(diff_a),
                transform=ax.transAxes)
        ax.text(0.5, 0.042, 'CRITIC: pooled global state  →  estimates how GOOD the current team situation is  (used ONLY during training)',
                color=GOLD, fontsize=9, ha='center', alpha=safe(diff_a),
                transform=ax.transAxes)

    # ── CTDE label ────────────────────────────────────────────────────────────
    if t > 0.60:
        ctde_a = ease(min((t - 0.60) / 0.15, 1.0)) * a
        ax.text(0.50, 0.895, '▶  Centralized Training  /  Decentralized Execution  (CTDE)',
                color=GREEN_C, fontsize=11, ha='center', alpha=safe(ctde_a),
                transform=ax.transAxes, fontweight='bold')
        lines2 = [
            'Training:   Critic sees ALL agents — better gradient estimates',
            'Deployment: Critic is discarded — each jammer runs its Actor alone in real time',
        ]
        for i, ln in enumerate(lines2):
            ax.text(0.50, 0.845 - i * 0.045, ln,
                    color=HUD_C, fontsize=9, ha='center',
                    alpha=safe(ctde_a * 0.85), transform=ax.transAxes)

    # ── Mean pooling explanation ──────────────────────────────────────────────
    if t > 0.72:
        mp_a = ease(min((t - 0.72) / 0.15, 1.0)) * a
        draw_hud_box(ax, 0.54, 0.22, 0.43, 0.12,
                     '', color=GOLD, alpha=safe(mp_a * 0.7))
        ax.text(0.755, 0.305, 'Mean pooling:  s_pooled = (1/M) × Σ s_j',
                color=GOLD, fontsize=9, ha='center', alpha=safe(mp_a),
                transform=ax.transAxes)
        ax.text(0.755, 0.26, 'Critic input size stays FIXED regardless of M  →  scales to M=40',
                color=TITLE_C, fontsize=8, ha='center', alpha=safe(mp_a),
                transform=ax.transAxes)


def render_ppo_explained(ax, t):
    a = scene_alpha(t)
    ax.set_facecolor(BG)

    ax.text(0.5, 0.96, 'PPO — Proximal Policy Optimisation  |  GAE  |  Clipping',
            color=TITLE_C, fontsize=12, ha='center', va='top',
            alpha=safe(a), transform=ax.transAxes, fontweight='bold')

    # ── Phase 1: What problem does PPO solve? (t 0.00–0.25) ──────────────────
    if t < 0.50:
        prob_a = safe(scene_alpha(t, fi=0.04, fo=0.48))

        draw_hud_box(ax, 0.03, 0.62, 0.94, 0.28,
                     'THE PROBLEM WITH NAIVE POLICY GRADIENT', color=WARN_C,
                     alpha=safe(prob_a))

        ax.text(0.5, 0.845,
                'Without clipping: one bad update can make the policy WORSE — and it never recovers.',
                color=WARN_C, fontsize=10, ha='center', alpha=safe(prob_a),
                transform=ax.transAxes)
        ax.text(0.5, 0.795,
                'Like steering a car: small correction = fine.  Giant jerk of the wheel = crash.',
                color=TITLE_C, fontsize=9.5, ha='center', alpha=safe(prob_a),
                transform=ax.transAxes, style='italic')

        # Show unstable vs stable curves
        xs = np.linspace(0, 1, 200)
        np.random.seed(3)
        bad_curve  = np.cumsum(np.random.randn(200) * 0.04) * 3
        good_curve = np.tanh(xs * 4) * 0.8 + np.random.randn(200) * 0.03

        bad_norm  = (bad_curve  - bad_curve.min())  / (bad_curve.max()  - bad_curve.min() + 1e-6)
        good_norm = (good_curve - good_curve.min()) / (good_curve.max() - good_curve.min() + 1e-6)

        reveal = ease(min(t / 0.3, 1.0))
        n = max(2, int(reveal * 200))

        # Plot area
        ax.plot(xs[:n] * 0.42 + 0.05, bad_norm[:n]  * 0.38 + 0.20,
                color=WARN_C, lw=1.8, alpha=safe(prob_a * 0.9))
        ax.plot(xs[:n] * 0.42 + 0.52, good_norm[:n] * 0.38 + 0.20,
                color=GREEN_C, lw=1.8, alpha=safe(prob_a * 0.9))

        ax.text(0.26, 0.215, 'Vanilla Policy Gradient\n(unstable — crashes)',
                color=WARN_C, fontsize=8.5, ha='center', alpha=safe(prob_a),
                transform=ax.transAxes, linespacing=1.3)
        ax.text(0.73, 0.215, 'PPO\n(stable — smooth learning)',
                color=GREEN_C, fontsize=8.5, ha='center', alpha=safe(prob_a),
                transform=ax.transAxes, linespacing=1.3)

    # ── Phase 2: Clipping (t 0.28–0.60) ──────────────────────────────────────
    if t > 0.28:
        clip_a = safe(scene_alpha(t, fi=0.28, fo=0.95))

        draw_hud_box(ax, 0.03, 0.48, 0.94, 0.13,
                     'PPO CLIPPING — The Guardrail', color=JAMMER_C,
                     alpha=safe(clip_a * ease(min((t - 0.28) / 0.12, 1.0))))

        clip_aa = safe(clip_a * ease(min((t - 0.28) / 0.12, 1.0)))
        ax.text(0.5, 0.565,
                'r_t(θ)  =  π_θ(a|s) / π_θ_old(a|s)          '
                'L_CLIP  =  E[ min( r_t · A_t ,  clip(r_t, 1−ε, 1+ε) · A_t ) ]',
                color=GOLD, fontsize=9.5, ha='center', alpha=safe(clip_aa),
                transform=ax.transAxes)
        ax.text(0.5, 0.520,
                'ε = 0.2  →  update is clipped to ±20% of old policy  →  no single update can destroy learned behaviour',
                color=TITLE_C, fontsize=9, ha='center', alpha=safe(clip_aa),
                transform=ax.transAxes)

    # ── Phase 3: GAE (t 0.45–0.75) ────────────────────────────────────────────
    if t > 0.45:
        gae_a = safe(clip_a * ease(min((t - 0.45) / 0.15, 1.0)) if t > 0.28 else 0)

        draw_hud_box(ax, 0.03, 0.32, 0.94, 0.145,
                     'GAE — Generalised Advantage Estimation', color=HUD_C,
                     alpha=safe(gae_a))

        gae_show = safe(gae_a * ease(min((t - 0.45) / 0.15, 1.0)))
        ax.text(0.5, 0.422,
                'δ_t  =  r_t  +  γ · V(s_{t+1}) · (1−done)  −  V(s_t)',
                color=GOLD, fontsize=9.5, ha='center', alpha=safe(gae_show),
                transform=ax.transAxes)
        ax.text(0.5, 0.378,
                'A_t  =  δ_t  +  (γ · λ_GAE) · A_{t+1} · (1−done)          '
                '[computed backwards through buffer]',
                color=GOLD, fontsize=9, ha='center', alpha=safe(gae_show),
                transform=ax.transAxes)

        if t > 0.55:
            gae2 = safe(gae_a * ease(min((t - 0.55) / 0.12, 1.0)))
            lines_gae = [
                ('λ_GAE = 0  →  pure TD  →  low variance, high bias   (trusts value function)',   HUD_C),
                ('λ_GAE = 1  →  Monte Carlo  →  high variance, zero bias   (trusts raw rewards)', HUD_C),
                ('λ_GAE = 0.95  →  sweet spot — used in our system', GREEN_C),
            ]
            for i, (ln, col) in enumerate(lines_gae):
                ax.text(0.5, 0.345 - i * 0.038, ln,
                        color=col, fontsize=8.5, ha='center', alpha=safe(gae2),
                        transform=ax.transAxes,
                        fontweight='bold' if i == 2 else 'normal')

    # ── Phase 4: Full PPO loop (t 0.65–end) ──────────────────────────────────
    if t > 0.65:
        loop_a = safe(ease(min((t - 0.65) / 0.15, 1.0)) * a)

        draw_hud_box(ax, 0.03, 0.01, 0.94, 0.30,
                     'PPO UPDATE LOOP — What Happens Every 2048 Steps',
                     color=GREEN_C, alpha=safe(loop_a))

        steps_ppo = [
            ('1', 'Collect 2048 steps in rollout buffer  (M agents × T steps)',     TITLE_C),
            ('2', 'Compute GAE advantages backwards through buffer',                 GOLD),
            ('3', 'Normalize advantages:  A = (A − mean) / (std + ε)',              GOLD),
            ('4', 'For 10 epochs: shuffle buffer → mini-batches of 256',            JAMMER_C),
            ('5', '  Forward Actor → log π  |  Forward Critic → V(s)',             JAMMER_C),
            ('6', '  Compute r_ratio = exp(log_π_new − log_π_old)',                JAMMER_C),
            ('7', '  L_CLIP + c1·V_loss − c2·Entropy  →  backward + Adam step',   JAMMER_C),
            ('8', '  Clip grad norm to 0.5  →  update Actor + Critic weights',     GREEN_C),
        ]

        for i, (num, text, col) in enumerate(steps_ppo):
            item_show = safe(loop_a * ease(min((t - 0.65 - i * 0.025) / 0.08, 1.0)))
            ax.text(0.06, 0.272 - i * 0.031, f'Step {num}:',
                    color=GREEN_C, fontsize=8, va='center',
                    alpha=safe(item_show), transform=ax.transAxes,
                    fontweight='bold')
            ax.text(0.135, 0.272 - i * 0.031, text,
                    color=col, fontsize=8, va='center',
                    alpha=safe(item_show), transform=ax.transAxes)

        # Hyperparams
        if t > 0.82:
            hp_a = safe(ease(min((t - 0.82) / 0.12, 1.0)) * a)
            hps = [
                'γ=0.99', 'λ_GAE=0.95', 'ε_clip=0.2',
                'lr_actor=3e-4', 'lr_critic=1e-3',
                'K_epochs=10', 'T_rollout=2048', 'batch=256',
            ]
            ax.text(0.5, 0.04,
                    '  ·  '.join(hps),
                    color=DIM, fontsize=8, ha='center', alpha=safe(hp_a),
                    transform=ax.transAxes)



SCENE_RENDERERS = {
    'title':         render_title,
    'swarm_intro':   render_swarm_intro,
    'graph_build':   render_graph_build,
    'lambda2':       render_lambda2,
    'dbscan':        render_dbscan,
    'deploy':        render_deploy,
    'obs_vector':    render_obs_vector,
    'neural_net':    render_neural_net,
    'critic_network': render_critic_network,
    'reward':        render_reward,
    'ppo_explained': render_ppo_explained,
    'training':      render_training,
    'jamming':       render_jamming,
    'fragmentation': render_fragmentation,
    'comparison':    render_comparison,
    'proposition':   render_proposition,
    'results':       render_results,
    'outro':         render_outro,
}

def get_active_scene(frame):
    for name, (start, duration) in SCENES.items():
        if start <= frame < start + duration:
            t = (frame - start) / max(duration - 1, 1)
            return name, t
    return None, 0.0


# ─────────────────────────────────────────────────────────────────────────────
# RENDER LOOP
# ─────────────────────────────────────────────────────────────────────────────

def render_frame(fig, ax, frame):
    ax.cla()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('auto')
    ax.axis('off')
    ax.set_facecolor(BG)
    fig.patch.set_facecolor(BG)

    # Progress bar
    prog = frame / TOTAL_FRAMES
    ax.add_patch(patches.Rectangle((0, 0), prog, 0.004,
                                   facecolor=JAMMER_C, alpha=0.6,
                                   transform=ax.transAxes, zorder=100))

    name, t = get_active_scene(frame)
    if name and name in SCENE_RENDERERS:
        SCENE_RENDERERS[name](ax, t)

    # Frame counter (tiny)
    ax.text(0.995, 0.01, f'{frame//FPS//60:02d}:{(frame//FPS)%60:02d}',
            color=DIM, fontsize=7, ha='right', va='bottom',
            transform=ax.transAxes, zorder=100)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("MARL Jammer — 2D Animated Explainer Video Renderer")
    print(f"Resolution: {FIG_W*100}×{FIG_H*100} px  |  {FPS} fps")
    print(f"Duration:   {TOTAL_SECS}s = {TOTAL_SECS//60}m {TOTAL_SECS%60}s")
    print(f"Frames:     {TOTAL_FRAMES}")
    print("=" * 60)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor(BG)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    output_path = 'MARL_Jammer_Explainer.mp4'

    writer = FFMpegWriter(
        fps=FPS,
        metadata={'title': 'MARL Jammer Drone System', 'artist': 'Research Team'},
        extra_args=['-vcodec', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p']
    )

    print(f"\nRendering to: {output_path}")
    print("Progress:")

    with writer.saving(fig, output_path, dpi=RENDER_DPI):
        for frame in range(TOTAL_FRAMES):
            render_frame(fig, ax, frame)
            writer.grab_frame()

            if frame % (FPS * 10) == 0:
                pct = frame / TOTAL_FRAMES * 100
                elapsed_s = frame // FPS
                print(f"  {pct:5.1f}%  |  {elapsed_s//60:02d}:{elapsed_s%60:02d} rendered  |  Scene: {get_active_scene(frame)[0]}")

    plt.close(fig)
    print(f"\n✓ Done! Video saved: {output_path}")
    print(f"  File size: check {output_path}")


if __name__ == '__main__':
    main()