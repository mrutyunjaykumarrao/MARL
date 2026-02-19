"""
Visualization Module
====================

Plotting and animation utilities.
    - plots: Training curves, bar charts, scalability plots
    - heatmaps: Coverage heatmaps before/after training
    - animation: Trajectory GIF generation
"""

from .plots import (
    plot_training_curves,
    plot_baseline_comparison,
    plot_lambda2_evolution,
)

from .heatmaps import (
    plot_coverage_heatmap,
)

from .animation import (
    create_trajectory_animation,
)
