"""
=================================================================
  Multi-Agent Jammer Drone System — Complete 2D Visualization
  All 9 Phases | MARL-PPO | Graph Laplacian | FSPL Jamming
  ~3.5 minutes | 20 FPS | Saves as MP4
=================================================================
  pip install matplotlib numpy scipy
  For MP4: winget install Gyan.FFmpeg  (admin PowerShell)
=================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
FPS = 20

# ── Color palette ─────────────────────────────────────────
BG     = '#060e1c'
BLUE   = '#00aaff'
TEAL   = '#00ffcc'
RED    = '#ff3344'
GOLD   = '#ffcc00'
GREEN  = '#00ff88'
PURPLE = '#bb55ff'
ORANGE = '#ff7700'
WHITE  = '#ddeeff'
DIM    = '#1a2f4a'
GRIDC  = '#0d1f33'
PINK   = '#ff44aa'

# ── Scene table  (name, start_frame, end_frame) ───────────
SCENES = [
    ('title',    0,    100),   # 5s
    ('swarm',    100,  360),   # 13s  Phase 1: detection
    ('dbscan',   360,  660),   # 15s  Phase 1: clustering
    ('graph',    660,  940),   # 14s  Phase 2: graph build
    ('lambda2',  940,  1160),  # 11s  Phase 2: λ₂
    ('deploy',   1160, 1380),  # 11s  Phase 3: jammers
    ('state',    1380, 1640),  # 13s  Phase 4: observation
    ('action',   1640, 1880),  # 12s  Phase 4: actor network
    ('reward',   1880, 2160),  # 14s  Phase 4: reward fn
    ('jamming',  2160, 2500),  # 17s  Phase 4: live jamming
    ('gae',      2500, 2740),  # 12s  Phase 5: GAE
    ('ppo',      2740, 2980),  # 12s  Phase 6: PPO
    ('training', 2980, 3320),  # 17s  Phases 5-6: curves
    ('deploy2',  3320, 3580),  # 13s  Phase 8: real-time
    ('done',     3580, 3720),  # 7s   Phase 9: complete
]
TOTAL = 3720   # 186s ≈ 3 min 6s

def get_scene(frame):
    for name, s, e in SCENES:
        if s <= frame < e:
            t = (frame - s) / max(1, e - s - 1)
            return name, float(np.clip(t, 0, 1)), frame - s
    return SCENES[-1][0], 1.0, frame - SCENES[-1][1]

def sc(t):
    """Smooth s-curve easing"""
    t = float(np.clip(t, 0, 1))
    return t * t * (3 - 2 * t)

def fi(t, start=0.0, dur=0.2):
    """Fade in"""
    if t < start: return 0.0
    return float(np.clip((t - start) / max(dur, 1e-6), 0, 1))

# ── Core geometry ─────────────────────────────────────────
ARENA = 10.0
enemy_pos = np.array([
    [2.5,6.5],[3.2,7.0],[2.9,6.0],[3.7,6.5],[3.1,7.3],
    [6.4,6.2],[7.1,6.8],[6.8,5.8],[7.5,6.3],[6.9,7.1],
], dtype=float)
C1c = enemy_pos[:5].mean(0)
C2c = enemy_pos[5:].mean(0)
COMM_R = 1.8

jammer_base = np.array([
    [1.8,0.8],[3.6,0.8],[6.0,0.8],[7.8,0.8]], dtype=float)
jammer_tgt = np.array([
    [C1c[0]-0.7, C1c[1]-0.5],[C1c[0]+0.7, C1c[1]+0.5],
    [C2c[0]-0.7, C2c[1]-0.5],[C2c[0]+0.7, C2c[1]+0.5],
], dtype=float)

def get_links(pos, broken=None):
    broken = broken or set()
    return [(i,j) for i in range(len(pos)) for j in range(i+1,len(pos))
            if np.linalg.norm(pos[i]-pos[j]) <= COMM_R and (i,j) not in broken]

# ── Simulated training curves ─────────────────────────────
EP = 500
ep_x = np.arange(EP)
rew_c = -14 + 17*(1-np.exp(-ep_x/130)) + np.random.randn(EP)*0.7
rew_c = np.clip(rew_c, -16, 4)
l2r_c = 78*(1-np.exp(-ep_x/160)) + np.random.randn(EP)*1.5
l2r_c = np.clip(l2r_c, 0, 82)
ent_c = 1.7*np.exp(-ep_x/210)+0.05+np.random.randn(EP)*0.02
ent_c = np.clip(ent_c, 0, 2)
vl_c  = 7.5*np.exp(-ep_x/110)+0.08+np.abs(np.random.randn(EP)*0.08)

# ── Figure ────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 9), facecolor=BG)
_prev_scene = ['']

def grid_bg(ax):
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(DIM)
        spine.set_linewidth(1)

def title_strip(fig_ref, txt, sub='', phase_label=''):
    """Add a consistent top title bar to the current figure."""
    ax_t = fig_ref.add_axes([0.0, 0.91, 1.0, 0.09])
    ax_t.set_facecolor(DIM)
    ax_t.axis('off')
    ax_t.text(0.5, 0.65, txt, color=BLUE, fontsize=15,
              ha='center', va='center', fontweight='bold')
    ax_t.text(0.5, 0.20, sub, color=TEAL, fontsize=9,
              ha='center', va='center')
    if phase_label:
        ax_t.text(0.01, 0.5, phase_label, color=GOLD, fontsize=9,
                  ha='left', va='center', fontweight='bold')

# ═══════════════════════════════════════════════════════════
#  SCENE 0 — TITLE
# ═══════════════════════════════════════════════════════════
def draw_title(t, lf):
    ax = fig.add_subplot(111)
    grid_bg(ax); ax.axis('off')
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    a1=sc(t/0.3); a2=sc(fi(t,0.2,0.25)); a3=sc(fi(t,0.45,0.25))
    ax.text(0.5,0.66,'Multi-Agent Jammer Drone System',
            color=BLUE,fontsize=30,ha='center',fontweight='bold',alpha=a1)
    ax.text(0.5,0.52,'MARL-PPO  ·  Graph Laplacian Reward  ·  FSPL Jamming',
            color=TEAL,fontsize=13,ha='center',alpha=a2)
    ax.text(0.5,0.40,'Complete System Walkthrough  —  All 9 Phases',
            color=WHITE,fontsize=10,ha='center',alpha=a3,style='italic')
    ax.text(0.5,0.25,'λ₂  →  0',color=GOLD,fontsize=20,ha='center',
            alpha=sc(fi(t,0.6,0.3)),fontweight='bold')
    for i,(x,y) in enumerate([(0.15,0.75),(0.85,0.75),(0.12,0.25),(0.88,0.25)]):
        angle = t*3.14+i*1.57
        ax.plot(x+0.03*np.cos(angle),y+0.02*np.sin(angle),
                'D',color=BLUE,ms=12,alpha=a1*0.7)

# ═══════════════════════════════════════════════════════════
#  SCENE 1 — ENEMY SWARM DETECTION  (Phase 1, steps 1-2)
# ═══════════════════════════════════════════════════════════
def draw_swarm(t, lf):
    title_strip(fig,'PHASE 1 — ENEMY SWARM DETECTION',
                'Sensor detects positions x₁...x₁₀ for all enemy drones','PHASE 1')
    ax = fig.add_axes([0.05,0.08,0.55,0.82])
    grid_bg(ax)
    ax.set_xlim(0,ARENA); ax.set_ylim(0,ARENA)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title('Detection Grid — 200m × 200m Arena',
                 color=WHITE, fontsize=10, pad=4)
    for x in np.arange(0,ARENA+1,2):
        ax.axvline(x,color=GRIDC,lw=0.4)
        ax.axhline(x,color=GRIDC,lw=0.4)

    # Drones appear one by one
    n_visible = min(10, int(t * 12) + 1)
    # Drift toward detection zone
    drift = np.array([0, -0.3*sc(t)])
    for i in range(n_visible):
        pos = enemy_pos[i] + drift + np.array([0.0, 0.8*(1-sc(min(1,t*4)))])
        a = sc(min(1, (t*12 - i) / 1.5))
        ax.plot(*pos, 'o', color=RED, ms=14, alpha=a,
                mec='#ff8888', mew=1.5)
        ax.text(pos[0]+0.25, pos[1]+0.2, f'x{i+1}',
                color=RED, fontsize=7, alpha=a*0.9)
        # Ping ring
        ring_r = (lf % 30) / 30 * 0.8
        ring_a = (1 - (lf % 30) / 30) * 0.4 * a
        c = Circle(pos, ring_r, fill=False, edgecolor=RED,
                   lw=1.2, alpha=ring_a)
        ax.add_patch(c)

    # Radar sweep line from center top
    sweep_angle = (t * 3 * np.pi) % (2*np.pi)
    r = 4.5
    ax.plot([5, 5+r*np.cos(sweep_angle)],
            [9.5, 9.5+r*np.sin(sweep_angle)],
            color=GREEN, lw=1.5, alpha=0.5)

    # Right info panel
    ax_r = fig.add_axes([0.63,0.08,0.34,0.82])
    grid_bg(ax_r); ax_r.axis('off')
    ax_r.set_xlim(0,1); ax_r.set_ylim(0,1)
    ax_r.text(0.5,0.95,'DETECTION LOG',color=TEAL,fontsize=11,
              ha='center',fontweight='bold')
    n_vis = min(10, int(t*12)+1)
    for i in range(n_vis):
        pos = enemy_pos[i]
        a = sc(min(1,(t*12-i)/1.5))
        y_pos = 0.88 - i*0.08
        ax_r.text(0.05, y_pos,
                  f'  Drone {i+1:02d}:  ({pos[0]:.1f}, {pos[1]:.1f}) m',
                  color=GREEN if i<5 else PURPLE,
                  fontsize=8.5, alpha=a, va='center')
    if t > 0.85:
        ax_r.add_patch(FancyBboxPatch((0.05,0.04),0.9,0.12,
                       boxstyle='round,pad=0.02',
                       facecolor=DIM,edgecolor=GREEN,lw=1.5))
        ax_r.text(0.5,0.10,f'✓  {n_vis}/10 drones detected',
                  color=GREEN,fontsize=10,ha='center',fontweight='bold',
                  alpha=sc(fi(t,0.85,0.1)))

# ═══════════════════════════════════════════════════════════
#  SCENE 2 — DBSCAN CLUSTERING  (Phase 1, steps 3-7)
# ═══════════════════════════════════════════════════════════
def draw_dbscan(t, lf):
    title_strip(fig,'PHASE 1 — DBSCAN SPATIAL CLUSTERING',
                'ε = 20m, MinPts = 4  →  Identifies C₁ and C₂, computes centroids','PHASE 1')
    ax = fig.add_axes([0.04,0.08,0.56,0.82])
    grid_bg(ax); ax.set_xlim(0,ARENA); ax.set_ylim(0,ARENA)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for x in np.arange(0,ARENA+1,2):
        ax.axvline(x,color=GRIDC,lw=0.4)
        ax.axhline(x,color=GRIDC,lw=0.4)

    # Phase 1: show epsilon circles (t < 0.35)
    eps_a = sc(fi(t,0.0,0.2)) * (1-sc(fi(t,0.3,0.12)))
    if eps_a > 0.01:
        eps_r = 0.9  # scaled epsilon
        for i, pos in enumerate(enemy_pos):
            c = Circle(pos, eps_r, fill=True, facecolor=BLUE+'18',
                       edgecolor=BLUE, lw=0.7, alpha=eps_a*0.6, ls='--')
            ax.add_patch(c)
        ax.text(3.0,2.5,'ε-neighborhood circles',color=BLUE,
                fontsize=8,ha='center',alpha=eps_a)

    # All drones
    for i, pos in enumerate(enemy_pos):
        col = BLUE if i < 5 else PURPLE
        ax.plot(*pos,'o',color=col,ms=13,mec='white',mew=0.8,zorder=5)
        ax.text(pos[0]+0.22,pos[1]+0.2,f'{i+1}',
                color=WHITE,fontsize=7,zorder=6)

    # Cluster boundaries appear
    c1_a = sc(fi(t,0.35,0.2))
    c2_a = sc(fi(t,0.45,0.2))
    if c1_a > 0:
        c1 = Circle(C1c, 1.4, fill=True, facecolor=BLUE+'22',
                    edgecolor=BLUE, lw=2.5, alpha=c1_a, ls='--')
        ax.add_patch(c1)
        ax.text(C1c[0], C1c[1]+1.65,'C₁',color=BLUE,
                fontsize=14,ha='center',fontweight='bold',alpha=c1_a)
    if c2_a > 0:
        c2 = Circle(C2c, 1.4, fill=True, facecolor=PURPLE+'22',
                    edgecolor=PURPLE, lw=2.5, alpha=c2_a, ls='--')
        ax.add_patch(c2)
        ax.text(C2c[0], C2c[1]+1.65,'C₂',color=PURPLE,
                fontsize=14,ha='center',fontweight='bold',alpha=c2_a)

    # Centroids appear
    cent_a = sc(fi(t,0.65,0.2))
    if cent_a > 0:
        ax.plot(*C1c,'*',color=GOLD,ms=16,zorder=8,alpha=cent_a)
        ax.plot(*C2c,'*',color=GOLD,ms=16,zorder=8,alpha=cent_a)
        ax.text(C1c[0]-0.1,C1c[1]-0.5,'μ₁',color=GOLD,
                fontsize=11,ha='center',fontweight='bold',alpha=cent_a)
        ax.text(C2c[0]-0.1,C2c[1]-0.5,'μ₂',color=GOLD,
                fontsize=11,ha='center',fontweight='bold',alpha=cent_a)

    # Right panel — DBSCAN explanation
    ax_r = fig.add_axes([0.62,0.08,0.36,0.82])
    grid_bg(ax_r); ax_r.axis('off')
    ax_r.set_xlim(0,1); ax_r.set_ylim(0,1)
    ax_r.text(0.5,0.95,'DBSCAN ALGORITHM',color=TEAL,fontsize=11,
              ha='center',fontweight='bold')

    steps = [
        (0.05, '① Compute Euclidean distances\n   d(i,j) for all pairs',  BLUE),
        (0.3,  '② ε = 20m neighborhood\n   MinPts = 4 core point rule', TEAL),
        (0.5,  '③ Form clusters:\n   C₁ = {1,2,3,4,5}',                 BLUE),
        (0.55, '   C₂ = {6,7,8,9,10}',                                  PURPLE),
        (0.7,  '④ Compute centroids:\n   μₖ = (1/|Cₖ|) Σ xᵢ',         GOLD),
    ]
    y = 0.84
    for thresh, txt, col in steps:
        a = sc(fi(t, thresh, 0.15))
        ax_r.text(0.08, y, txt, color=col, fontsize=8.5, alpha=a, va='top')
        y -= 0.14 + txt.count('\n')*0.055

    if t > 0.75:
        a = sc(fi(t,0.75,0.15))
        ax_r.add_patch(FancyBboxPatch((0.05,0.04),0.9,0.22,
                       boxstyle='round,pad=0.02',
                       facecolor=DIM,edgecolor=GOLD,lw=1.5,alpha=a))
        ax_r.text(0.5,0.20,'C₁: centroid μ₁ = (3.08, 6.68)',
                  color=BLUE,fontsize=9,ha='center',alpha=a)
        ax_r.text(0.5,0.11,'C₂: centroid μ₂ = (6.94, 6.44)',
                  color=PURPLE,fontsize=9,ha='center',alpha=a)

# ═══════════════════════════════════════════════════════════
#  SCENE 3 — COMMUNICATION GRAPH  (Phase 2, steps 8-11)
# ═══════════════════════════════════════════════════════════
def draw_graph(t, lf):
    title_strip(fig,'PHASE 2 — COMMUNICATION GRAPH CONSTRUCTION',
                'G=(V,E): Adjacency A, Degree D, Laplacian L = D − A','PHASE 2')

    # Left: network visualization
    ax = fig.add_axes([0.04,0.08,0.44,0.82])
    grid_bg(ax); ax.set_xlim(-0.5,ARENA+0.5); ax.set_ylim(-0.5,ARENA+0.5)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title('Communication Graph  G = (V, E)',color=WHITE,fontsize=10,pad=4)
    for x in np.arange(0,ARENA+1,2):
        ax.axvline(x,color=GRIDC,lw=0.4)
        ax.axhline(x,color=GRIDC,lw=0.4)

    links = get_links(enemy_pos)
    n_links_show = int(sc(fi(t,0.1,0.5)) * len(links))
    link_a = sc(fi(t,0.1,0.5))

    for k,(i,j) in enumerate(links[:n_links_show]):
        intensity = 0.5 + 0.5*(1 - np.linalg.norm(enemy_pos[i]-enemy_pos[j])/COMM_R)
        ax.plot([enemy_pos[i,0],enemy_pos[j,0]],
                [enemy_pos[i,1],enemy_pos[j,1]],
                '-', color=BLUE, lw=1.2+intensity*0.8,
                alpha=link_a*intensity*0.8, zorder=3)

    for i,pos in enumerate(enemy_pos):
        col = BLUE if i<5 else PURPLE
        deg = sum(1 for (a,b) in links if a==i or b==i)
        ax.plot(*pos,'o',color=col,ms=14,mec=WHITE,mew=0.8,zorder=5)
        ax.text(pos[0]+0.25,pos[1]+0.2,f'{i+1}',
                color=WHITE,fontsize=7,zorder=6)

    n_links_show_label = int(sc(fi(t,0.1,0.5))*len(links))
    ax.text(5,0.4,f'Active links: {n_links_show_label} / {len(links)}',
            color=TEAL,fontsize=9,ha='center',
            alpha=sc(fi(t,0.4,0.2)))

    # Right top: adjacency matrix (small)
    ax_m = fig.add_axes([0.51,0.50,0.22,0.40])
    grid_bg(ax_m); ax_m.set_xticks([]); ax_m.set_yticks([])
    ax_m.set_title('Adjacency Matrix A\n(Aᵢⱼ = 1 if link exists)',
                   color=WHITE, fontsize=8, pad=3)
    mat_a = sc(fi(t,0.05,0.3))
    A = np.zeros((10,10))
    for (i,j) in links:
        A[i,j] = A[j,i] = 1
    ax_m.imshow(A, cmap='Blues', vmin=0, vmax=1.2, alpha=mat_a)
    ax_m.set_xlabel('Node j', color=WHITE, fontsize=7)
    ax_m.set_ylabel('Node i', color=WHITE, fontsize=7)
    for spine in ax_m.spines.values():
        spine.set_edgecolor(BLUE)

    # Right bottom: Laplacian explanation
    ax_e = fig.add_axes([0.51,0.08,0.46,0.38])
    grid_bg(ax_e); ax_e.axis('off')
    ax_e.set_xlim(0,1); ax_e.set_ylim(0,1)
    eqs = [
        (0.0, 'Degree Matrix:  Dᵢᵢ = Σⱼ Aᵢⱼ',         GOLD,  11),
        (0.2, 'Laplacian:      L  =  D  −  A',          GREEN, 11),
        (0.4, 'L is symmetric positive semi-definite',   WHITE,  9),
        (0.6, 'One zero eigenvalue per connected component', WHITE, 8),
        (0.75,'→ λ₁ = 0 always for connected graph',    TEAL,   8),
    ]
    for thresh, txt, col, sz in eqs:
        a = sc(fi(t, thresh+0.3, 0.15))
        ax_e.text(0.05, 0.88 - thresh*0.9, txt,
                  color=col, fontsize=sz, alpha=a, va='center')

    # Right top right: degree distribution
    ax_d = fig.add_axes([0.76,0.50,0.21,0.40])
    grid_bg(ax_d)
    ax_d.set_title('Node Degrees',color=WHITE,fontsize=8,pad=3)
    degs = [sum(1 for(a,b) in links if a==i or b==i) for i in range(10)]
    cols = [BLUE if i<5 else PURPLE for i in range(10)]
    bar_a = sc(fi(t,0.5,0.3))
    ax_d.bar(range(10), degs, color=cols, alpha=bar_a*0.85, edgecolor=WHITE, lw=0.5)
    ax_d.set_xlabel('Node', color=WHITE, fontsize=7)
    ax_d.set_ylabel('Degree', color=WHITE, fontsize=7)
    ax_d.tick_params(colors=WHITE, labelsize=7)
    ax_d.set_facecolor(BG)
    for spine in ax_d.spines.values(): spine.set_edgecolor(DIM)

# ═══════════════════════════════════════════════════════════
#  SCENE 4 — λ₂ FIEDLER VALUE  (Phase 2, steps 12-14)
# ═══════════════════════════════════════════════════════════
def draw_lambda2(t, lf):
    title_strip(fig,'PHASE 2 — FIEDLER VALUE  λ₂  COMPUTATION',
                'Eigenvalue decomposition of L  →  λ₂ quantifies algebraic connectivity','PHASE 2')

    ax = fig.add_axes([0.05,0.08,0.55,0.82])
    grid_bg(ax); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')

    # Step 1: eigenvalue decomposition text
    a1 = sc(fi(t,0.0,0.2))
    ax.text(0.5,0.90,'Eigenvalue Decomposition of L:',
            color=WHITE,fontsize=13,ha='center',alpha=a1,fontweight='bold')
    ax.text(0.5,0.82,'L · v  =  λ · v',
            color=GOLD,fontsize=18,ha='center',alpha=a1,fontweight='bold')

    # Eigenvalue spectrum (simulated)
    eigenvalues = [0.0, 0.82, 1.21, 1.55, 1.87, 2.10, 2.34, 2.56, 2.78, 3.12]
    n_show = min(10, int(t * 16) + 1) if t > 0.15 else 0
    spec_a = sc(fi(t,0.15,0.25))

    ax_spec = fig.add_axes([0.07,0.20,0.50,0.38])
    grid_bg(ax_spec)
    ax_spec.set_title('Eigenvalue Spectrum of L',color=WHITE,fontsize=10,pad=4)
    colors_eig = [GREEN if i==1 else (BLUE if i==0 else WHITE)
                  for i in range(len(eigenvalues))]
    labels_eig = [f'λ{i+1}' for i in range(len(eigenvalues))]
    bars = ax_spec.bar(range(n_show),
                       [eigenvalues[i] for i in range(n_show)],
                       color=[colors_eig[i] for i in range(n_show)],
                       alpha=spec_a*0.9, edgecolor=WHITE, lw=0.7)
    ax_spec.set_xticks(range(n_show))
    ax_spec.set_xticklabels([labels_eig[i] for i in range(n_show)],
                             fontsize=8, color=WHITE)
    ax_spec.set_ylabel('Eigenvalue', color=WHITE, fontsize=8)
    ax_spec.tick_params(colors=WHITE, labelsize=8)
    ax_spec.set_facecolor(BG)
    for spine in ax_spec.spines.values(): spine.set_edgecolor(DIM)
    if n_show > 1:
        ax_spec.annotate('← λ₂ = Fiedler value', xy=(1, 0.82),
                         xytext=(3, 1.5),
                         arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.5),
                         color=GREEN, fontsize=9, alpha=spec_a)

    # λ₂ big display
    box_a = sc(fi(t,0.55,0.2))
    ax.add_patch(FancyBboxPatch((0.15,0.04),0.70,0.22,
                 boxstyle='round,pad=0.02',
                 facecolor=DIM,edgecolor=GREEN,lw=2,alpha=box_a))
    ax.text(0.5,0.14,'λ₂  =  0.82   →   HIGH CONNECTIVITY',
            color=GREEN,fontsize=14,ha='center',
            fontweight='bold',alpha=box_a)
    ax.text(0.5,0.07,'λ₂ > 0  ⟺  Swarm is connected',
            color=TEAL,fontsize=9,ha='center',alpha=box_a)

    # Right panel: meaning explanation
    ax_r = fig.add_axes([0.63,0.08,0.35,0.82])
    grid_bg(ax_r); ax_r.axis('off')
    ax_r.set_xlim(0,1); ax_r.set_ylim(0,1)
    ax_r.text(0.5,0.96,'WHY λ₂ MATTERS',
              color=TEAL,fontsize=11,ha='center',fontweight='bold')
    facts = [
        (0.0,  'λ₂ > 0',        '→ Swarm is connected',         GREEN),
        (0.12, 'Larger λ₂',     '→ More fault-tolerant swarm',  BLUE),
        (0.25, 'λ₂ → 0',        '→ Swarm near fragmentation',   ORANGE),
        (0.38, 'λ₂ = 0',        '→ Swarm FULLY disconnected',   RED),
        (0.52, 'Our mission:',   'Drive λ₂(t) → 0',             GOLD),
    ]
    y = 0.87
    for thresh, label, desc, col in facts:
        a = sc(fi(t, thresh+0.35, 0.18))
        ax_r.text(0.08, y, label, color=col, fontsize=10,
                  fontweight='bold', alpha=a)
        ax_r.text(0.42, y, desc, color=WHITE, fontsize=9, alpha=a)
        y -= 0.11

    # Mission objective box
    mo_a = sc(fi(t,0.82,0.15))
    ax_r.add_patch(FancyBboxPatch((0.05,0.04),0.90,0.18,
                   boxstyle='round,pad=0.02',
                   facecolor=DIM,edgecolor=GOLD,lw=2,alpha=mo_a))
    ax_r.text(0.5,0.16,'MISSION OBJECTIVE',
              color=GOLD,fontsize=10,ha='center',fontweight='bold',alpha=mo_a)
    ax_r.text(0.5,0.09,'Reduce  λ₂(T) / λ₂(0)  ≥ 80%',
              color=WHITE,fontsize=9,ha='center',alpha=mo_a)

# ═══════════════════════════════════════════════════════════
#  SCENE 5 — JAMMER DEPLOYMENT  (Phase 3, steps 15-20)
# ═══════════════════════════════════════════════════════════
def draw_deploy(t, lf):
    title_strip(fig,'PHASE 3 — JAMMER DEPLOYMENT & NETWORK INIT',
                '4 jammer agents deployed (2 per cluster)  ·  Actor & Critic initialized','PHASE 3')

    ax = fig.add_axes([0.04,0.08,0.55,0.82])
    grid_bg(ax); ax.set_xlim(0,ARENA); ax.set_ylim(0,ARENA)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for x in np.arange(0,ARENA+1,2):
        ax.axvline(x,color=GRIDC,lw=0.4)
        ax.axhline(x,color=GRIDC,lw=0.4)

    # Enemy clusters
    for i,pos in enumerate(enemy_pos):
        col = BLUE if i<5 else PURPLE
        ax.plot(*pos,'o',color=col,ms=11,mec=WHITE,mew=0.6,alpha=0.7,zorder=4)
    for (c,r,col) in [(C1c,1.4,BLUE),(C2c,1.4,PURPLE)]:
        ax.add_patch(Circle(c,r,fill=False,edgecolor=col,
                            lw=1.5,ls='--',alpha=0.5))

    # Jammers fly from base to targets
    launch_t = np.clip(t*1.5, 0, 1)
    jammer_colors = [GREEN, TEAL, GREEN, TEAL]
    for i in range(4):
        delay = i*0.08
        lt = sc(np.clip((t-delay)*1.6, 0, 1))
        pos = jammer_base[i] + (jammer_tgt[i] - jammer_base[i]) * lt
        # Trail
        if lt > 0.05:
            n_trail = 6
            for k in range(n_trail):
                kt = sc(np.clip((t-delay-k*0.03)*1.6,0,1))
                tp = jammer_base[i]+(jammer_tgt[i]-jammer_base[i])*kt
                ax.plot(*tp,'D',color=jammer_colors[i],
                        ms=3+k*0.5,alpha=0.08*(n_trail-k),zorder=3)
        ax.plot(*pos,'D',color=jammer_colors[i],ms=14,
                mec=WHITE,mew=1.0,zorder=6)
        ax.text(pos[0]+0.25,pos[1]+0.2,f'J{i+1}',
                color=jammer_colors[i],fontsize=8,fontweight='bold')

    # Assignment arrows at end
    if t > 0.7:
        aa = sc(fi(t,0.7,0.2))
        ax.annotate('2 per cluster',xy=C1c,xytext=(C1c[0]-2.5,C1c[1]+2),
                    arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.5),
                    color=BLUE, fontsize=9, alpha=aa)
        ax.annotate('2 per cluster',xy=C2c,xytext=(C2c[0]+0.5,C2c[1]+2),
                    arrowprops=dict(arrowstyle='->', color=PURPLE, lw=1.5),
                    color=PURPLE, fontsize=9, alpha=aa)

    # Right: initialization checklist
    ax_r = fig.add_axes([0.62,0.08,0.36,0.82])
    grid_bg(ax_r); ax_r.axis('off')
    ax_r.set_xlim(0,1); ax_r.set_ylim(0,1)
    ax_r.text(0.5,0.95,'INITIALIZATION CHECKLIST',
              color=TEAL,fontsize=10,ha='center',fontweight='bold')
    checks = [
        (0.0,  '4 jammer agents deployed',       '✓', GREEN),
        (0.15, '2 jammers → Cluster C₁',         '✓', BLUE),
        (0.25, '2 jammers → Cluster C₂',         '✓', PURPLE),
        (0.4,  'Actor network πθ initialized',   '✓', TEAL),
        (0.55, 'Critic network Vφ initialized',  '✓', GOLD),
        (0.65, 'Rollout buffer (empty)',         '⬜', WHITE),
        (0.75, 'Copy policy: πθ_old ← πθ',       '✓', TEAL),
        (0.85, 'Episode counter: 0',             '✓', WHITE),
    ]
    y = 0.86
    for thresh, txt, mark, col in checks:
        a = sc(fi(t, thresh+0.1, 0.15))
        ax_r.text(0.08, y, f'{mark}  {txt}',
                  color=col, fontsize=9, alpha=a)
        y -= 0.09
    if t > 0.9:
        a = sc(fi(t,0.9,0.1))
        ax_r.add_patch(FancyBboxPatch((0.05,0.03),0.9,0.10,
                       boxstyle='round,pad=0.02',
                       facecolor=DIM,edgecolor=GREEN,lw=2,alpha=a))
        ax_r.text(0.5,0.08,'EPISODE 0  —  TRAINING BEGINS',
                  color=GREEN,fontsize=10,ha='center',fontweight='bold',alpha=a)

# ═══════════════════════════════════════════════════════════
#  SCENE 6 — STATE SPACE  (Phase 4, step 21)
# ═══════════════════════════════════════════════════════════
def draw_state(t, lf):
    title_strip(fig,'PHASE 4 — STATE OBSERVATION  sⱼ',
                'Each jammer observes 5 features from its local environment','PHASE 4')

    ax = fig.add_axes([0.04,0.08,0.50,0.82])
    grid_bg(ax); ax.set_xlim(0,ARENA); ax.set_ylim(0,ARENA)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for x in np.arange(0,ARENA+1,2):
        ax.axvline(x,color=GRIDC,lw=0.4)
        ax.axhline(x,color=GRIDC,lw=0.4)

    # Show one cluster + one jammer
    for i in range(5):
        ax.plot(*enemy_pos[i],'o',color=RED,ms=11,mec='#ff8888',mew=1.0)
    ax.add_patch(Circle(C1c,1.4,fill=False,edgecolor=BLUE,lw=1.5,ls='--',alpha=0.5))
    ax.plot(*C1c,'*',color=GOLD,ms=14,zorder=5)
    ax.text(C1c[0]+0.2,C1c[1]+0.3,'μ₁',color=GOLD,fontsize=10,fontweight='bold')

    # Jammer position (orbiting)
    angle = t * 2 * np.pi * 0.8
    jx = C1c[0] + 0.9*np.cos(angle)
    jy = C1c[1] + 0.7*np.sin(angle)
    jpos = np.array([jx, jy])
    ax.plot(*jpos,'D',color=GREEN,ms=16,mec=WHITE,mew=1.2,zorder=8)
    ax.text(jx+0.3,jy+0.2,'J₁',color=GREEN,fontsize=10,fontweight='bold')

    # Second jammer
    ax.plot(*jammer_tgt[1],'D',color=TEAL,ms=12,mec=WHITE,mew=0.8,alpha=0.6)

    # 5 observation arrows + labels
    obs_items = [
        ('d(μ₁, J₁)\nDist to centroid',  C1c,       GOLD,   0.0),
        ('ρₖ\nCluster density',           enemy_pos[2], ORANGE, 0.15),
        ('d(J₁, J₂)\nJammer spacing',    jammer_tgt[1],TEAL,  0.30),
        ('Oⱼ\nCoverage overlap',         jpos+np.array([0.5,0.5]), PINK, 0.45),
        ('E=[e₁,e₂,e₃,e₄]\nFreq spectrum',jpos+np.array([-0.5,1.0]),PURPLE,0.60),
    ]
    for txt, target, col, thresh in obs_items:
        a = sc(fi(t, thresh+0.05, 0.20))
        if a > 0.01:
            ax.annotate('', xy=target, xytext=jpos,
                        arrowprops=dict(arrowstyle='->', color=col,
                                        lw=1.8, connectionstyle='arc3,rad=0.2'))
            mid = (jpos + np.array(target)) / 2
            ax.text(mid[0]+0.1,mid[1]+0.1,txt,color=col,
                    fontsize=7.5,alpha=a,
                    bbox=dict(facecolor=BG,edgecolor=col,
                              boxstyle='round,pad=0.2',alpha=0.7))

    # Right: state vector display
    ax_r = fig.add_axes([0.57,0.08,0.41,0.82])
    grid_bg(ax_r); ax_r.axis('off')
    ax_r.set_xlim(0,1); ax_r.set_ylim(0,1)
    ax_r.text(0.5,0.95,'STATE VECTOR  sⱼ  ∈  ℝ⁵',
              color=TEAL,fontsize=11,ha='center',fontweight='bold')

    jd = np.linalg.norm(jpos - C1c)
    states = [
        ('s₁', 'd(μₖ, xⱼ)',   f'{jd:.2f} m',      'Distance to centroid', GOLD),
        ('s₂', 'ρₖ',           '5.00',              'Cluster density',      ORANGE),
        ('s₃', "d(j,j')",      '1.41 m',            'Inter-jammer distance', TEAL),
        ('s₄', 'Oⱼ',          '0.23',              'Coverage overlap',      PINK),
        ('s₅', 'E[band]',      '[0.8,0.2,0.9,0.4]','Energy spectrum',       PURPLE),
    ]
    y = 0.86
    for i,(sym,name,val,desc,col) in enumerate(states):
        a = sc(fi(t, i*0.12, 0.18))
        ax_r.add_patch(FancyBboxPatch((0.04,y-0.065),0.92,0.075,
                       boxstyle='round,pad=0.01',
                       facecolor=DIM,edgecolor=col,lw=1,alpha=a*0.6))
        ax_r.text(0.09, y-0.025, f'{sym}  {name}',
                  color=col, fontsize=10, fontweight='bold', alpha=a)
        ax_r.text(0.62, y-0.025, val,
                  color=WHITE, fontsize=9, alpha=a)
        ax_r.text(0.09, y-0.053, desc,
                  color=WHITE+'88', fontsize=7.5, alpha=a*0.8)
        y -= 0.105

    if t > 0.75:
        a = sc(fi(t,0.75,0.15))
        ax_r.text(0.5,0.12,'sⱼ → [Actor Network πθ] → action aⱼ',
                  color=GREEN,fontsize=9.5,ha='center',alpha=a,fontweight='bold')

# ═══════════════════════════════════════════════════════════
#  SCENE 7 — ACTION SELECTION  (Phase 4, steps 22-27)
# ═══════════════════════════════════════════════════════════
def draw_action(t, lf):
    title_strip(fig,'PHASE 4 — ACTOR NETWORK  πθ  →  ACTION SELECTION',
                'Continuous: Vx,Vy ~ N(μθ,σθ)    Discrete: Pband ~ Categorical','PHASE 4')

    ax = fig.add_axes([0.03,0.08,0.60,0.82])
    grid_bg(ax); ax.axis('off'); ax.set_xlim(0,10); ax.set_ylim(0,8)

    layer_x = [1.0, 3.2, 5.5, 7.8, 9.5]
    layer_n = [5, 8, 8, 6, 3]   # nodes per layer: [in, h1, h2, h3, out]
    layer_labels = ['Input\n(State s)', 'Hidden\nLayer 1',
                    'Hidden\nLayer 2', 'Hidden\nLayer 3', 'Output']
    layer_colors = [BLUE, TEAL, TEAL, TEAL, GREEN]

    node_positions = {}
    # Draw connections first
    conn_a = sc(fi(t,0.1,0.3))
    for li in range(len(layer_x)-1):
        n1, n2 = layer_n[li], layer_n[li+1]
        for i in range(n1):
            y1 = 4.0 + (i-(n1-1)/2)*0.72
            for j in range(n2):
                y2 = 4.0 + (j-(n2-1)/2)*0.72
                intensity = 0.15 + 0.3*np.random.rand()
                ax.plot([layer_x[li], layer_x[li+1]], [y1, y2],
                        '-', color=TEAL, lw=0.5,
                        alpha=conn_a*intensity*0.8, zorder=1)

    # Animate data flow
    flow_a = sc(fi(t,0.3,0.2))
    if flow_a > 0.01:
        phase_offset = (lf / 8) % 1.0
        for li in range(len(layer_x)-1):
            x_flow = layer_x[li] + (layer_x[li+1]-layer_x[li])*phase_offset
            n1 = layer_n[li]
            for i in range(n1):
                y_flow = 4.0 + (i-(n1-1)/2)*0.72
                ax.plot(x_flow, y_flow, 'o', color=GOLD,
                        ms=4, alpha=flow_a*0.6, zorder=4)

    # Draw nodes
    node_a = sc(fi(t,0.0,0.2))
    input_labels = ['d(μ,x)', 'ρₖ', "d(j,j')", 'Oⱼ', 'E[b]']
    for li, (lx, n, col) in enumerate(zip(layer_x, layer_n, layer_colors)):
        for i in range(n):
            y = 4.0 + (i-(n-1)/2)*0.72
            circle = Circle((lx, y), 0.22, facecolor=BG,
                            edgecolor=col, lw=1.5, zorder=5, alpha=node_a)
            ax.add_patch(circle)
            if li == 0:
                ax.text(lx-0.7, y, input_labels[i],
                        color=BLUE, fontsize=7, ha='right', va='center',
                        alpha=node_a)
        ax.text(lx, 1.3, layer_labels[li],
                color=col, fontsize=8, ha='center', alpha=node_a)

    # Output labels
    out_a = sc(fi(t,0.5,0.25))
    out_items = [
        (4.0 + 0.72, 'Vx', GOLD, 'velocity x'),
        (4.0,        'Vy', GOLD, 'velocity y'),
        (4.0 - 0.72, 'Pband','#ff88ff','freq band'),
    ]
    lx = layer_x[-1]
    for (y, label, col, desc) in out_items:
        ax.text(lx+0.6, y, f'{label}',
                color=col, fontsize=11, ha='left', va='center',
                fontweight='bold', alpha=out_a)
        ax.text(lx+0.6, y-0.32, f'({desc})',
                color=WHITE, fontsize=7.5, ha='left', va='center',
                alpha=out_a*0.8)

    # Right panels: output distributions
    ax_vx = fig.add_axes([0.66,0.60,0.16,0.28])
    grid_bg(ax_vx)
    ax_vx.set_title('Vx ~ N(μ,σ)',color=GOLD,fontsize=9,pad=3)
    dist_a = sc(fi(t,0.55,0.25))
    xg = np.linspace(-3,3,100)
    ax_vx.plot(xg, np.exp(-xg**2/2)/np.sqrt(2*np.pi),
               color=GOLD, lw=2, alpha=dist_a)
    ax_vx.fill_between(xg, np.exp(-xg**2/2)/np.sqrt(2*np.pi),
                       alpha=dist_a*0.3, color=GOLD)
    ax_vx.axvline(0.8, color=RED, lw=2, ls='--', alpha=dist_a,
                  label='sampled')
    ax_vx.set_facecolor(BG); ax_vx.tick_params(colors=WHITE,labelsize=6)
    for sp in ax_vx.spines.values(): sp.set_edgecolor(DIM)
    ax_vx.legend(fontsize=6, labelcolor=WHITE, facecolor=DIM)

    ax_band = fig.add_axes([0.84,0.60,0.14,0.28])
    grid_bg(ax_band)
    ax_band.set_title('Pband Categorical',color=PURPLE,fontsize=8,pad=3)
    bands = ['433\nMHz','915\nMHz','2.4\nGHz','5.8\nGHz']
    probs = [0.10, 0.15, 0.60, 0.15]
    bcols = [BLUE,BLUE,GREEN,BLUE]
    ax_band.bar(range(4), probs, color=bcols, alpha=dist_a*0.85,
                edgecolor=WHITE, lw=0.5)
    ax_band.set_xticks(range(4))
    ax_band.set_xticklabels(bands, fontsize=6, color=WHITE)
    ax_band.set_ylabel('P', color=WHITE, fontsize=7)
    ax_band.set_facecolor(BG)
    ax_band.tick_params(colors=WHITE,labelsize=6)
    for sp in ax_band.spines.values(): sp.set_edgecolor(DIM)

    ax_eq = fig.add_axes([0.66,0.08,0.32,0.46])
    grid_bg(ax_eq); ax_eq.axis('off')
    ax_eq.set_xlim(0,1); ax_eq.set_ylim(0,1)
    ax_eq.text(0.5,0.95,'Action Space  aⱼ',color=GREEN,
               fontsize=11,ha='center',fontweight='bold',
               alpha=sc(fi(t,0.4,0.2)))
    items = [
        (0.4,'Vx, Vy  ∈  ℝ  (continuous)',    GOLD),
        (0.5,'Sampled from Gaussian N(μθ, σθ)', WHITE),
        (0.6,'Pband  ∈  {0,1,2,3}  (discrete)', PURPLE),
        (0.7,'Sampled from Categorical dist.',   WHITE),
        (0.8,'Log prob stored: log πθ(aⱼ|sⱼ)',  TEAL),
    ]
    y = 0.78
    for thresh, txt, col in items:
        a = sc(fi(t,thresh,0.15))
        ax_eq.text(0.05, y, txt, color=col, fontsize=9, alpha=a)
        y -= 0.14

# ═══════════════════════════════════════════════════════════
#  SCENE 8 — REWARD FUNCTION  (Phase 4, step 39)
# ═══════════════════════════════════════════════════════════
def draw_reward(t, lf):
    title_strip(fig,'PHASE 4 — REWARD FUNCTION  Rⱼ',
                'Multi-term reward drives λ₂ reduction, band matching, proximity & energy efficiency','PHASE 4')

    ax = fig.add_axes([0.03,0.08,0.62,0.82])
    grid_bg(ax); ax.axis('off'); ax.set_xlim(0,10); ax.set_ylim(0,8)

    # Full equation
    eq_a = sc(fi(t,0.0,0.25))
    ax.text(5.0,7.3,'Rⱼ  =  ω₁·T₁  +  ω₂·T₂  +  ω₃·T₃  −  ω₄·T₄',
            color=WHITE,fontsize=14,ha='center',fontweight='bold',alpha=eq_a)

    terms = [
        (0.15,'T₁  =  1 − λ₂(t)/λ₂(0)',
              'Connectivity Reduction',
              'Reward for reducing swarm\nalgebraic connectivity.',
              GOLD,   0.8, 1.4, '▲ Main term'),
        (0.30,'T₂  =  𝟙[Pband = Penemy]',
              'Band Match Reward',
              'Bonus when jammer selects\nthe same band as enemy.',
              GREEN,  0.15, 3.8, '▲ Binary'),
        (0.45,'T₃  =  e^(−k·d(μₖ,xⱼ))',
              'Proximity Reward',
              'Reward for staying close\nto the cluster centroid.',
              BLUE,   0.4, 6.2, '▲ Exponential'),
        (0.60,'T₄  =  Cenergy,j',
              'Energy Penalty',
              'Penalty for high power\nconsumption.',
              RED,    0.2, 8.6, '▼ Penalty'),
    ]
    for thresh, eq, name, desc, col, w, x, note in terms:
        a = sc(fi(t, thresh, 0.20))
        box_y = 1.2 + terms.index((thresh,eq,name,desc,col,w,x,note))*1.4
        ax.add_patch(FancyBboxPatch((0.3,box_y-0.05),9.4,1.1,
                     boxstyle='round,pad=0.05',
                     facecolor=col+'18',edgecolor=col,lw=1.5,alpha=a*0.9))
        ax.text(0.6, box_y+0.75, eq,
                color=col, fontsize=11, fontweight='bold', alpha=a)
        ax.text(0.6, box_y+0.45, name,
                color=WHITE, fontsize=9, alpha=a)
        ax.text(0.6, box_y+0.10, desc,
                color=WHITE+'99', fontsize=8, alpha=a*0.8)
        ax.text(9.5, box_y+0.45, f'ω={w}',
                color=col, fontsize=9, ha='right', alpha=a, fontweight='bold')

    # Right: bar chart of term contributions
    ax_r = fig.add_axes([0.68,0.35,0.29,0.55])
    grid_bg(ax_r)
    ax_r.set_title('Reward Term Contributions\n(example episode)',
                   color=WHITE, fontsize=9, pad=4)
    bar_a = sc(fi(t,0.65,0.25))
    term_vals = [0.42, 0.15, 0.31, -0.08]
    term_colors = [GOLD, GREEN, BLUE, RED]
    term_labels = ['T₁·ω₁\nλ₂ red.','T₂·ω₂\nband','T₃·ω₃\nprox.','−T₄·ω₄\nenergy']
    bars = ax_r.bar(range(4), term_vals,
                    color=term_colors, alpha=bar_a*0.85,
                    edgecolor=WHITE, lw=0.7)
    ax_r.axhline(0, color=WHITE, lw=1, alpha=0.5)
    ax_r.set_xticks(range(4))
    ax_r.set_xticklabels(term_labels, fontsize=7, color=WHITE)
    ax_r.set_ylabel('Value', color=WHITE, fontsize=8)
    ax_r.set_facecolor(BG)
    ax_r.tick_params(colors=WHITE, labelsize=7)
    for sp in ax_r.spines.values(): sp.set_edgecolor(DIM)
    for bar,v in zip(bars, term_vals):
        ax_r.text(bar.get_x()+bar.get_width()/2,
                  v + (0.02 if v>0 else -0.04),
                  f'{v:+.2f}', ha='center', color=WHITE,
                  fontsize=8, alpha=bar_a)

    total_r = sum(term_vals)
    ax_b = fig.add_axes([0.68,0.08,0.29,0.22])
    grid_bg(ax_b); ax_b.axis('off')
    ax_b.set_xlim(0,1); ax_b.set_ylim(0,1)
    ra = sc(fi(t,0.8,0.15))
    ax_b.add_patch(FancyBboxPatch((0.05,0.35),0.90,0.55,
                   boxstyle='round,pad=0.03',
                   facecolor=DIM,edgecolor=GOLD,lw=2,alpha=ra))
    ax_b.text(0.5,0.74,'Total Reward  Rⱼ',
              color=GOLD,fontsize=10,ha='center',fontweight='bold',alpha=ra)
    ax_b.text(0.5,0.52,f'Rⱼ  =  {total_r:+.2f}',
              color=GREEN,fontsize=16,ha='center',fontweight='bold',alpha=ra)
    ax_b.text(0.5,0.16,'Stored in rollout buffer',
              color=TEAL,fontsize=8,ha='center',alpha=ra*0.9)

# ═══════════════════════════════════════════════════════════
#  SCENE 9 — LIVE JAMMING SIMULATION  (Phase 4, steps 28-43)
# ═══════════════════════════════════════════════════════════
def draw_jamming(t, lf):
    lam2_now = max(0.0, 0.82 * (1 - sc(max(0,(t-0.25)/0.65))))
    title_strip(fig,'PHASE 4 — LIVE JAMMING SIMULATION',
                f'Jammers disrupt links  ·  λ₂ = {lam2_now:.3f}  →  0.000','PHASE 4')

    ax = fig.add_axes([0.04,0.08,0.52,0.82])
    grid_bg(ax); ax.set_xlim(0,ARENA); ax.set_ylim(0,ARENA)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for x in np.arange(0,ARENA+1,2):
        ax.axvline(x,color=GRIDC,lw=0.4)
        ax.axhline(x,color=GRIDC,lw=0.4)

    # Jammers orbit targets
    angle = t * 5.0
    jammer_now = jammer_tgt.copy()
    for i in range(4):
        jammer_now[i] += np.array([
            0.4*np.cos(angle + i*np.pi/2),
            0.3*np.sin(angle + i*np.pi/2)])

    # Determine broken links based on progress
    jam_progress = sc(max(0, (t-0.2)/0.75))
    broken = set()
    all_l = get_links(enemy_pos)
    n_break = int(jam_progress * len(all_l))
    for k, link in enumerate(all_l):
        if k < n_break:
            broken.add(link)

    # Draw remaining links
    for (i,j) in all_l:
        if (i,j) in broken:
            ax.plot([enemy_pos[i,0],enemy_pos[j,0]],
                    [enemy_pos[i,1],enemy_pos[j,1]],
                    '--', color=DIM, lw=0.6, alpha=0.3, zorder=2)
        else:
            strength = 1 - np.linalg.norm(enemy_pos[i]-enemy_pos[j])/COMM_R
            ax.plot([enemy_pos[i,0],enemy_pos[j,0]],
                    [enemy_pos[i,1],enemy_pos[j,1]],
                    '-', color=BLUE, lw=1.0+strength*0.8,
                    alpha=0.7*strength+0.1, zorder=2)

    # Enemy drones — scatter when fully jammed
    scatter_scale = sc(max(0,(t-0.85)/0.15))
    for i, pos in enumerate(enemy_pos):
        scatter_dir = (pos - np.array([5,6.5])) * scatter_scale * 1.5
        dp = pos + scatter_dir
        dp = np.clip(dp, 0.5, ARENA-0.5)
        col = RED if i<5 else ORANGE
        # Flicker when breaking
        flicker = 1.0 if (i,min(i+1,9)) not in broken else (0.5+0.5*np.sin(lf*0.8))
        ax.plot(*dp,'o',color=col,ms=12,mec='#ff8888',mew=0.8,
                alpha=flicker,zorder=5)

    # Jammer rings
    jam_active = sc(fi(t,0.1,0.2))
    for i, jp in enumerate(jammer_now):
        if jam_active > 0.1:
            pulse_r = 0.8 + 0.3*np.sin(lf*0.3+i)
            c = Circle(jp, pulse_r, fill=True,
                       facecolor=ORANGE+'22', edgecolor=ORANGE,
                       lw=1.5, alpha=jam_active*0.7, zorder=3)
            ax.add_patch(c)
        ax.plot(*jp,'D',color=GREEN,ms=14,mec=WHITE,mew=1.0,zorder=7)
        ax.text(jp[0]+0.2,jp[1]+0.2,f'J{i+1}',
                color=GREEN,fontsize=8,fontweight='bold')

    # λ₂ counter on main axis
    lam_col = GREEN if lam2_now>0.55 else (ORANGE if lam2_now>0.2 else RED)
    ax.text(5.0, 9.6, f'λ₂ = {lam2_now:.3f}',
            color=lam_col, fontsize=14, ha='center', fontweight='bold',
            bbox=dict(facecolor=BG,edgecolor=lam_col,boxstyle='round,pad=0.3'))

    # Right panel
    ax_r = fig.add_axes([0.59,0.08,0.39,0.82])
    grid_bg(ax_r)
    ax_r.set_title('λ₂ Live Reduction',color=WHITE,fontsize=10,pad=6)

    # λ₂ time series
    t_hist = np.linspace(0, t, 200)
    l2_hist = np.maximum(0, 0.82*(1 - np.array([sc(max(0,(tt-0.25)/0.65)) for tt in t_hist])))
    ax_r.plot(t_hist, l2_hist, color=TEAL, lw=2.5)
    ax_r.fill_between(t_hist, l2_hist, alpha=0.15, color=TEAL)
    ax_r.axhline(0.82, color=DIM, lw=1, ls=':', alpha=0.5)
    ax_r.text(0.02, 0.84, 'λ₂(0) = 0.82', color=DIM, fontsize=8,
              transform=ax_r.transAxes)
    ax_r.set_xlabel('Time', color=WHITE, fontsize=9)
    ax_r.set_ylabel('λ₂', color=WHITE, fontsize=9)
    ax_r.set_xlim(0,1); ax_r.set_ylim(-0.05,0.95)
    ax_r.tick_params(colors=WHITE,labelsize=8)
    ax_r.set_facecolor(BG)
    for sp in ax_r.spines.values(): sp.set_edgecolor(DIM)

    # Stats below graph
    n_active = len(all_l) - len(broken)
    stats_y = 0.34
    stats = [
        (f'Active links:  {n_active} / {len(all_l)}', BLUE),
        (f'Broken links:  {len(broken)}',              RED),
        (f'λ₂ reduction:  {(1-lam2_now/0.82)*100:.1f}%', GOLD),
        (f'Jammer band:  2.4 GHz ✓',                  GREEN),
        (f'Timestep:     {int(t*2048)}  / 2048',       WHITE),
    ]
    for st, col in stats:
        a = sc(fi(t,0.05,0.2))
        ax_r.text(0.05, stats_y, st, color=col, fontsize=9,
                  transform=ax_r.transAxes, alpha=a)
        stats_y -= 0.065

    if t > 0.9:
        fa = sc(fi(t,0.9,0.1))
        ax_r.text(0.5, 0.04, '✓  ROLLOUT BUFFER FULL\n    2048 / 2048 timesteps',
                  color=GREEN, fontsize=10, ha='center', va='bottom',
                  fontweight='bold', transform=ax_r.transAxes, alpha=fa)

# ═══════════════════════════════════════════════════════════
#  SCENE 10 — GAE  (Phase 5, steps 44-48)
# ═══════════════════════════════════════════════════════════
def draw_gae(t, lf):
    title_strip(fig,'PHASE 5 — ADVANTAGE COMPUTATION  (GAE)',
                'Generalized Advantage Estimation: Aₜ = Σ (γλ)ˡ δₜ₊ₗ','PHASE 5')

    ax = fig.add_axes([0.04,0.58,0.92,0.32])
    grid_bg(ax)
    ax.set_title('Rollout Buffer Timeline  (2048 timesteps)',color=WHITE,fontsize=9,pad=4)
    T_show = 60
    tt = np.arange(T_show)
    rewards = np.random.randn(T_show)*0.8 + 0.3
    values  = np.cumsum(rewards[::-1])[::-1]*0.1 + np.random.randn(T_show)*0.2
    values  = (values - values.min())/(values.max()-values.min()+1e-6)*3

    tl_a = sc(fi(t,0.0,0.3))
    ax.bar(tt, rewards, color=GOLD, alpha=tl_a*0.7, width=0.7, label='Reward rₜ')
    ax.plot(tt, values, color=BLUE, lw=2, alpha=tl_a*0.9, label='Value V(sₜ)')

    # TD errors
    td_a = sc(fi(t,0.25,0.25))
    td_errors = rewards + 0.99*np.roll(values,-1) - values
    td_errors[-1] = rewards[-1] - values[-1]
    ax.bar(tt, td_errors*0.5, bottom=values,
           color=TEAL, alpha=td_a*0.6, width=0.5, label='TD error δₜ')

    ax.axhline(0, color=WHITE, lw=0.8, alpha=0.4)
    ax.set_xlabel('Timestep t', color=WHITE, fontsize=8)
    ax.set_facecolor(BG)
    ax.tick_params(colors=WHITE, labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor(DIM)
    ax.legend(fontsize=7, labelcolor=WHITE, facecolor=DIM,
              loc='upper right', framealpha=0.7)

    # GAE wave
    ax2 = fig.add_axes([0.04,0.08,0.45,0.44])
    grid_bg(ax2)
    ax2.set_title('GAE Advantage Aₜ (raw)',color=WHITE,fontsize=9,pad=4)
    gae_a = sc(fi(t,0.45,0.25))
    advantages = np.array([sum((0.99*0.95)**l * td_errors[(k+l)%T_show]
                               for l in range(T_show-k))
                            for k in range(T_show)])
    ax2.plot(tt, advantages, color=PURPLE, lw=2.5, alpha=gae_a)
    ax2.fill_between(tt, advantages, alpha=gae_a*0.25, color=PURPLE)
    ax2.axhline(0, color=WHITE, lw=0.8, alpha=0.4)
    ax2.set_xlabel('Timestep', color=WHITE, fontsize=8)
    ax2.set_ylabel('Advantage Aₜ', color=WHITE, fontsize=8)
    ax2.set_facecolor(BG)
    ax2.tick_params(colors=WHITE, labelsize=7)
    for sp in ax2.spines.values(): sp.set_edgecolor(DIM)

    ax3 = fig.add_axes([0.53,0.08,0.44,0.44])
    grid_bg(ax3)
    ax3.set_title('Normalized Advantage Aₜ (mean=0, std=1)',color=WHITE,fontsize=9,pad=4)
    norm_a = sc(fi(t,0.65,0.25))
    adv_norm = (advantages - advantages.mean())/(advantages.std()+1e-8)
    ax3.plot(tt, adv_norm, color=GREEN, lw=2.5, alpha=norm_a)
    ax3.fill_between(tt, adv_norm, alpha=norm_a*0.25, color=GREEN)
    ax3.axhline(0, color=WHITE, lw=1, alpha=0.5)
    ax3.axhline(1, color=DIM, lw=1, ls='--', alpha=0.4, label='±1σ')
    ax3.axhline(-1, color=DIM, lw=1, ls='--', alpha=0.4)
    ax3.set_xlabel('Timestep', color=WHITE, fontsize=8)
    ax3.set_ylabel('Norm. Advantage', color=WHITE, fontsize=8)
    ax3.set_facecolor(BG)
    ax3.tick_params(colors=WHITE, labelsize=7)
    for sp in ax3.spines.values(): sp.set_edgecolor(DIM)
    ax3.legend(fontsize=7, labelcolor=WHITE, facecolor=DIM, framealpha=0.7)

    if t > 0.85:
        a = sc(fi(t,0.85,0.12))
        ax3.text(0.5,-0.18,'→ Normalized Aₜ passed to PPO update',
                 color=TEAL,fontsize=9,ha='center',
                 transform=ax3.transAxes,alpha=a,fontweight='bold')

# ═══════════════════════════════════════════════════════════
#  SCENE 11 — PPO CLIPPING  (Phase 6, steps 49-68)
# ═══════════════════════════════════════════════════════════
def draw_ppo(t, lf):
    title_strip(fig,'PHASE 6 — PPO POLICY UPDATE  (Clipped Surrogate Objective)',
                'Lclip = min(rₜ·Aₜ, clip(rₜ, 1−ε, 1+ε)·Aₜ)    ε = 0.2','PHASE 6')

    ax = fig.add_axes([0.04,0.42,0.46,0.48])
    grid_bg(ax)
    ax.set_title('PPO Clipping Mechanism',color=WHITE,fontsize=10,pad=4)
    r_vals = np.linspace(0.3, 2.0, 200)
    At_pos = 1.0
    clip_a = sc(fi(t,0.0,0.3))
    obj_unclipped = r_vals * At_pos
    obj_clipped   = np.clip(r_vals, 0.8, 1.2) * At_pos
    obj_final     = np.minimum(obj_unclipped, obj_clipped)
    ax.plot(r_vals, obj_unclipped, color=BLUE, lw=2, ls='--',
            alpha=clip_a*0.7, label='rₜ · Aₜ (unclipped)')
    ax.plot(r_vals, obj_clipped, color=ORANGE, lw=2, ls='-.',
            alpha=clip_a*0.7, label='clip(rₜ)·Aₜ')
    ax.plot(r_vals, obj_final, color=GREEN, lw=3,
            alpha=clip_a, label='min → Lclip (actual)')
    ax.axvline(0.8, color=RED, lw=1.5, ls=':', alpha=clip_a*0.8)
    ax.axvline(1.2, color=RED, lw=1.5, ls=':', alpha=clip_a*0.8)
    ax.axvline(1.0, color=WHITE, lw=1, ls='-', alpha=clip_a*0.4)
    ax.text(0.8,  -0.08,'1−ε',color=RED,fontsize=9,ha='center',alpha=clip_a)
    ax.text(1.2,  -0.08,'1+ε',color=RED,fontsize=9,ha='center',alpha=clip_a)
    ax.text(0.55, 0.6,'CLIPPED\nREGION',color=RED,fontsize=8,alpha=clip_a*0.7)
    ax.text(1.4,  0.6,'CLIPPED\nREGION',color=RED,fontsize=8,alpha=clip_a*0.7)
    ax.set_xlabel('Probability Ratio rₜ', color=WHITE, fontsize=9)
    ax.set_ylabel('Objective Value', color=WHITE, fontsize=9)
    ax.set_facecolor(BG)
    ax.tick_params(colors=WHITE,labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor(DIM)
    ax.legend(fontsize=7.5, labelcolor=WHITE, facecolor=DIM,
              framealpha=0.8, loc='upper left')

    ax2 = fig.add_axes([0.54,0.42,0.43,0.48])
    grid_bg(ax2)
    ax2.set_title('Total PPO Loss',color=WHITE,fontsize=10,pad=4)
    loss_a = sc(fi(t,0.35,0.3))
    ep_l = np.linspace(0,1,200)
    lclip = -0.5*(1-np.exp(-ep_l*3)) + np.random.randn(200)*0.02
    lval  = 2.0*np.exp(-ep_l*4) + 0.1 + np.abs(np.random.randn(200)*0.05)
    lent  = -0.3*np.exp(-ep_l*2) + np.random.randn(200)*0.01
    ltot  = lclip + 0.5*lval + lent
    ax2.plot(ep_l, lclip, color=GREEN, lw=2, alpha=loss_a*0.9,
             label='−Lclip')
    ax2.plot(ep_l, lval,  color=ORANGE, lw=2, alpha=loss_a*0.9,
             label='c₁·Lvalue')
    ax2.plot(ep_l, lent,  color=PURPLE, lw=1.5, alpha=loss_a*0.8,
             label='−c₂·H(π)')
    ax2.plot(ep_l, ltot,  color=WHITE, lw=2.5, alpha=loss_a,
             label='Total L')
    ax2.axhline(0, color=WHITE, lw=0.8, alpha=0.3)
    ax2.set_xlabel('Training Progress', color=WHITE, fontsize=9)
    ax2.set_ylabel('Loss', color=WHITE, fontsize=9)
    ax2.set_facecolor(BG)
    ax2.tick_params(colors=WHITE,labelsize=8)
    for sp in ax2.spines.values(): sp.set_edgecolor(DIM)
    ax2.legend(fontsize=7.5, labelcolor=WHITE, facecolor=DIM,
               framealpha=0.8, loc='upper right')

    ax_eq = fig.add_axes([0.04,0.08,0.92,0.28])
    grid_bg(ax_eq); ax_eq.axis('off')
    ax_eq.set_xlim(0,10); ax_eq.set_ylim(0,3)
    eqs = [
        (0.00, 0, 2.5, 'Total Loss:',
               'L  =  Lclip  −  c₁·Lvalue  +  c₂·H(π)', WHITE, GOLD),
        (0.30, 0, 1.5, 'Actor update (Adam):',
               'θ  ←  θ  +  α · m̂ / (√v̂ + ε)', BLUE, TEAL),
        (0.55, 5, 1.5, 'Critic update (Adam):',
               'φ  ←  φ  −  α · Adam(∇φ Lvalue)', PURPLE, GREEN),
        (0.75, 0, 0.4, 'Repeat K=4..10 epochs, then:',
               'πθ_old  ←  πθ    |    Clear rollout buffer', WHITE, WHITE),
    ]
    for thresh, x, y, label, eq, col1, col2 in eqs:
        a = sc(fi(t, thresh+0.1, 0.18))
        ax_eq.text(x+0.1, y+0.3, label, color=col1, fontsize=9,
                   alpha=a, fontweight='bold')
        ax_eq.text(x+0.1, y, eq, color=col2, fontsize=10, alpha=a)

# ═══════════════════════════════════════════════════════════
#  SCENE 12 — TRAINING RESULTS  (Phases 5-6 over many episodes)
# ═══════════════════════════════════════════════════════════
def draw_training(t, lf):
    title_strip(fig,'PHASES 5-6 — TRAINING PROGRESS OVER 847 EPISODES',
                'MARL-PPO converges: λ₂ reduction ≥ 70% · Reward plateaus · Entropy drops','PHASES 5-6')

    n_show = max(2, int(sc(fi(t,0.0,0.85)) * EP))

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           left=0.07, right=0.97,
                           top=0.88, bottom=0.08,
                           hspace=0.40, wspace=0.35)
    axes = [fig.add_subplot(gs[i,j]) for i in range(2) for j in range(2)]
    plots = [
        (rew_c, 'Episode Reward',    'Reward',       GREEN,  True),
        (l2r_c, 'λ₂ Reduction (%)', 'Reduction %',  GOLD,   False),
        (ent_c, 'Policy Entropy',    'Entropy H(π)', PURPLE, True),
        (vl_c,  'Value Loss',        'Loss',         ORANGE, True),
    ]
    smoothing_window = 20
    for ax_p, (data, title, ylabel, col, descend) in zip(axes, plots):
        grid_bg(ax_p)
        ax_p.set_title(title, color=WHITE, fontsize=9, pad=4)
        xs = ep_x[:n_show]; ys = data[:n_show]
        ax_p.plot(xs, ys, color=col, lw=0.8, alpha=0.25)
        if n_show > smoothing_window:
            smooth = np.convolve(ys, np.ones(smoothing_window)/smoothing_window,
                                 mode='valid')
            ax_p.plot(xs[smoothing_window-1:], smooth,
                      color=col, lw=2.5, alpha=0.95)
        ax_p.set_xlabel('Episode', color=WHITE, fontsize=8)
        ax_p.set_ylabel(ylabel, color=WHITE, fontsize=8)
        ax_p.set_facecolor(BG)
        ax_p.tick_params(colors=WHITE, labelsize=7)
        for sp in ax_p.spines.values(): sp.set_edgecolor(DIM)
        # Convergence line
        if t > 0.75 and title == 'λ₂ Reduction (%)':
            ax_p.axhline(70, color=RED, lw=1.5, ls='--', alpha=0.7,
                         label='Target 70%')
            ax_p.legend(fontsize=7, labelcolor=WHITE, facecolor=DIM, framealpha=0.7)
        if t > 0.88:
            final_val = data[n_show-1]
            conv_a = sc(fi(t,0.88,0.12))
            ax_p.text(0.97, 0.06, f'Final: {final_val:.1f}',
                      color=col, fontsize=8, ha='right', va='bottom',
                      transform=ax_p.transAxes, alpha=conv_a, fontweight='bold')

    if t > 0.92:
        ca = sc(fi(t,0.92,0.08))
        fig.text(0.5, 0.02,
                 '✓  CONVERGENCE ACHIEVED  —  λ₂ reduction = 78.4% consistently',
                 color=GREEN, fontsize=11, ha='center',
                 fontweight='bold', alpha=ca)

# ═══════════════════════════════════════════════════════════
#  SCENE 13 — REAL-TIME DEPLOYMENT  (Phase 8)
# ═══════════════════════════════════════════════════════════
def draw_deploy2(t, lf):
    lam2_rt = max(0.0, 0.82*(1-sc(t*1.1)))
    title_strip(fig,'PHASE 8 — REAL-TIME DEPLOYMENT',
                f'Trained πθ* running deterministically  ·  λ₂ = {lam2_rt:.3f}','PHASE 8')

    ax = fig.add_axes([0.04,0.08,0.52,0.82])
    grid_bg(ax); ax.set_xlim(0,ARENA); ax.set_ylim(0,ARENA)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for x in np.arange(0,ARENA+1,2):
        ax.axvline(x,color=GRIDC,lw=0.4)
        ax.axhline(x,color=GRIDC,lw=0.4)

    # Scatter enemy drones as mission progresses
    scatter_t = sc(max(0,(t-0.6)/0.4))
    scatter_dirs = [pos-np.array([5,6.5]) for pos in enemy_pos]
    for i, pos in enumerate(enemy_pos):
        sdp = pos + scatter_dirs[i]*scatter_t*2.5
        sdp = np.clip(sdp, 0.3, ARENA-0.3)
        col = RED if i<5 else ORANGE
        alpha_d = max(0.3, 1.0-scatter_t*0.5)
        ax.plot(*sdp,'o',color=col,ms=11,mec='#ff8888',mew=0.6,alpha=alpha_d)

    # Links disappear
    all_l = get_links(enemy_pos)
    n_keep = max(0, int((1-sc(t*1.3))*len(all_l)))
    for i_link, (i,j) in enumerate(all_l[:n_keep]):
        ax.plot([enemy_pos[i,0],enemy_pos[j,0]],
                [enemy_pos[i,1],enemy_pos[j,1]],
                '-',color=BLUE,lw=1.0,alpha=0.6,zorder=2)

    # Jammers active
    angle2 = t*6.0
    for i in range(4):
        jp = jammer_tgt[i]+np.array([0.35*np.cos(angle2+i*1.57),
                                      0.28*np.sin(angle2+i*1.57)])
        pr = 0.7+0.2*np.sin(lf*0.4+i)
        ax.add_patch(Circle(jp,pr,fill=True,
                           facecolor=ORANGE+'1a',edgecolor=ORANGE,
                           lw=1.5,alpha=0.8,zorder=3))
        ax.plot(*jp,'D',color=GREEN,ms=14,mec=WHITE,mew=1.0,zorder=7)
        ax.text(jp[0]+0.2,jp[1]+0.25,f'J{i+1}',
                color=GREEN,fontsize=8,fontweight='bold')

    lam_col2 = GREEN if lam2_rt>0.55 else (ORANGE if lam2_rt>0.2 else RED)
    ax.text(5,9.6,f'λ₂ = {lam2_rt:.3f}',
            color=lam_col2,fontsize=14,ha='center',fontweight='bold',
            bbox=dict(facecolor=BG,edgecolor=lam_col2,
                      boxstyle='round,pad=0.3'))

    ax_r = fig.add_axes([0.59,0.08,0.39,0.82])
    grid_bg(ax_r)
    ax_r.set_title('Real-Time Control Loop',color=WHITE,fontsize=10,pad=5)
    t_rt = np.linspace(0,t,300)
    l2_rt_hist = np.maximum(0,0.82*(1-np.array([sc(tt*1.1) for tt in t_rt])))
    ax_r.plot(t_rt, l2_rt_hist, color=TEAL, lw=2.5)
    ax_r.fill_between(t_rt, l2_rt_hist, alpha=0.15, color=TEAL)
    ax_r.axhline(0.82,color=DIM,lw=1,ls=':',alpha=0.4)
    ax_r.axhline(0.0, color=GREEN,lw=1.5,ls='--',alpha=0.5,label='Target: 0')
    ax_r.set_xlabel('Real-Time Step', color=WHITE, fontsize=9)
    ax_r.set_ylabel('λ₂', color=WHITE, fontsize=9)
    ax_r.set_xlim(0,1); ax_r.set_ylim(-0.05,0.9)
    ax_r.set_facecolor(BG)
    ax_r.tick_params(colors=WHITE,labelsize=8)
    for sp in ax_r.spines.values(): sp.set_edgecolor(DIM)
    ax_r.legend(fontsize=8, labelcolor=WHITE, facecolor=DIM, framealpha=0.7)

    loop_items = [
        'OBSERVE  →  sⱼ from sensors',
        'NORMALIZE  →  (s − mean) / std',
        'ACTOR πθ*  →  deterministic action',
        'EXTRACT  →  Vx, Vy, Pband=argmax',
        'EXECUTE  →  move + transmit',
        'MEASURE  →  λ₂(t) from new L',
        'REPEAT  →  every timestep',
    ]
    y_pos = 0.39
    for item in loop_items:
        a = sc(fi(t,0.05,0.3))
        ax_r.text(0.03, y_pos, item, color=TEAL, fontsize=8.5,
                  transform=ax_r.transAxes, alpha=a)
        y_pos -= 0.048

    if t > 0.8 and lam2_rt < 0.15:
        fa = sc(fi(t,0.82,0.12))
        ax_r.text(0.5, 0.05, '✓  λ₂ < THRESHOLD\n   MISSION SUCCESS',
                  color=GREEN, fontsize=10, ha='center', va='bottom',
                  fontweight='bold', transform=ax_r.transAxes, alpha=fa)

# ═══════════════════════════════════════════════════════════
#  SCENE 14 — MISSION COMPLETE  (Phase 9)
# ═══════════════════════════════════════════════════════════
def draw_done(t, lf):
    title_strip(fig,'PHASE 9 — MISSION COMPLETE',
                'Swarm fragmented  ·  λ₂ = 0.000  ·  Connectivity disruption: 100%','PHASE 9')

    ax = fig.add_axes([0.04,0.08,0.52,0.82])
    grid_bg(ax); ax.set_xlim(0,ARENA); ax.set_ylim(0,ARENA)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for x in np.arange(0,ARENA+1,2):
        ax.axvline(x,color=GRIDC,lw=0.4)
        ax.axhline(x,color=GRIDC,lw=0.4)

    scatter_dirs = [(pos-np.array([5,6.5]))*2.5 for pos in enemy_pos]
    for i,(pos,sd) in enumerate(zip(enemy_pos,scatter_dirs)):
        dp = np.clip(pos+sd*sc(t),0.3,ARENA-0.3)
        spin_a = 0.3+0.3*np.sin(lf*0.5+i)
        ax.plot(*dp,'o',color=RED,ms=10,mec='#ff5555',mew=0.6,alpha=spin_a)

    for i in range(4):
        ax.plot(*jammer_tgt[i],'D',color=GREEN,ms=15,mec=WHITE,mew=1.2,alpha=0.9)

    ax.text(5,9.5,'λ₂  =  0.000',color=RED,fontsize=16,ha='center',
            fontweight='bold',
            bbox=dict(facecolor=BG,edgecolor=RED,boxstyle='round,pad=0.4'))
    ax.text(5,0.5,'SWARM  FRAGMENTED',color=GOLD,fontsize=12,ha='center',
            fontweight='bold',alpha=sc(fi(t,0.2,0.3)))

    ax_r = fig.add_axes([0.59,0.08,0.39,0.82])
    grid_bg(ax_r); ax_r.axis('off')
    ax_r.set_xlim(0,1); ax_r.set_ylim(0,1)
    ax_r.text(0.5,0.96,'MISSION REPORT',color=GOLD,fontsize=14,
              ha='center',fontweight='bold',alpha=sc(fi(t,0.0,0.2)))
    metrics = [
        ('λ₂ Reduction',          '100.0 %',  GREEN),
        ('Initial λ₂',             '0.820',    BLUE),
        ('Final λ₂',               '0.000',    RED),
        ('Comm. Links Broken',     '21 / 21',  ORANGE),
        ('Band Match Rate',        '100 %',    GREEN),
        ('Jammers Deployed',       '4 / 4',    TEAL),
        ('Mission Duration',       '47 steps', WHITE),
        ('Energy Consumed',        '2.3 W',    WHITE),
    ]
    y = 0.85
    for i,(label,val,col) in enumerate(metrics):
        a = sc(fi(t, i*0.08, 0.15))
        ax_r.add_patch(FancyBboxPatch((0.04,y-0.05),0.92,0.065,
                       boxstyle='round,pad=0.01',
                       facecolor=DIM,edgecolor=col,lw=1,alpha=a*0.5))
        ax_r.text(0.09, y-0.015, f'✓  {label}',
                  color=WHITE, fontsize=9, alpha=a)
        ax_r.text(0.88, y-0.015, val,
                  color=col, fontsize=9, ha='right', fontweight='bold', alpha=a)
        y -= 0.085

    if t > 0.82:
        fa = sc(fi(t,0.82,0.15))
        ax_r.add_patch(FancyBboxPatch((0.04,0.03),0.92,0.12,
                       boxstyle='round,pad=0.03',
                       facecolor=DIM,edgecolor=GOLD,lw=2.5,alpha=fa))
        ax_r.text(0.5,0.11,'AI DEFENSE SUCCESSFUL',
                  color=GOLD,fontsize=12,ha='center',
                  fontweight='bold',alpha=fa)
        ax_r.text(0.5,0.05,'λ₂  →  0   ≡   Complete Fragmentation',
                  color=TEAL,fontsize=8.5,ha='center',alpha=fa*0.9)

# ═══════════════════════════════════════════════════════════
#  SCENE ROUTER
# ═══════════════════════════════════════════════════════════
SCENE_FUNCS = {
    'title':   draw_title,
    'swarm':   draw_swarm,
    'dbscan':  draw_dbscan,
    'graph':   draw_graph,
    'lambda2': draw_lambda2,
    'deploy':  draw_deploy,
    'state':   draw_state,
    'action':  draw_action,
    'reward':  draw_reward,
    'jamming': draw_jamming,
    'gae':     draw_gae,
    'ppo':     draw_ppo,
    'training':draw_training,
    'deploy2': draw_deploy2,
    'done':    draw_done,
}

def update(frame):
    scene_name, t, lf = get_scene(frame)
    if scene_name != _prev_scene[0]:
        fig.clear()
        fig.set_facecolor(BG)
        _prev_scene[0] = scene_name
        # Print progress
        idx = next(i for i,s in enumerate(SCENES) if s[0]==scene_name)
        print(f'\r  Scene {idx+1:02d}/15: {scene_name:<12} {"|"*(idx+1)}{" "*(14-idx)}',
              end='', flush=True)
    SCENE_FUNCS[scene_name](t, lf)
    return []

# ═══════════════════════════════════════════════════════════
#  RUN
# ═══════════════════════════════════════════════════════════
print('='*58)
print('  Multi-Agent Jammer Drone System — 2D Visualization')
print('  15 Scenes | 3 min 6s | 20 FPS | All Phases')
print('='*58)
print(f'  Total frames: {TOTAL}  |  Duration: {TOTAL//FPS}s\n')

anim = animation.FuncAnimation(fig, update, frames=TOTAL,
                                interval=1000//FPS, blit=False,
                                repeat=False)

# Save
print('  Saving as MP4...')
try:
    writer = animation.FFMpegWriter(
        fps=FPS, bitrate=3000,
        extra_args=['-vcodec','libx264','-pix_fmt','yuv420p',
                    '-preset','fast', '-crf','22'])
    anim.save('drone_system_visualization.mp4', writer=writer,
              dpi=110, savefig_kwargs={'facecolor': BG})
    print('\n\n  ✅  Saved: drone_system_visualization.mp4')
    print('  📁  File is in your MARL JAMMER folder')

except Exception as e:
    print(f'\n  ⚠️  FFmpeg error: {e}')
    print('  Trying GIF fallback...')
    anim.save('drone_system_visualization.gif',
              writer=animation.PillowWriter(fps=FPS),
              dpi=80, savefig_kwargs={'facecolor': BG})
    print('  ✅  Saved: drone_system_visualization.gif')

print('\n  Showing live preview (close window to exit)')
plt.show()