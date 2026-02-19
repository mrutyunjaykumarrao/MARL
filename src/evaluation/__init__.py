"""
Evaluation Module
=================

Metrics, baselines, and ablation study configurations.
    - metrics: Lambda-2 reduction, energy efficiency, band match rate
    - baselines: Random, Greedy, Single-Agent, Independent PPO
    - ablation: Ablation study configurations
"""

from .metrics import (
    compute_lambda2_reduction,
    compute_energy_efficiency,
    compute_band_match_rate,
    compute_fragmentation_time,
)

from .baselines import (
    RandomPolicy,
    GreedyDistancePolicy,
    evaluate_baseline,
)
