"""
Physics Module
==============

Contains all RF propagation and signal processing calculations:
    - fspl: Free-Space Path Loss calculations
    - communication_graph: Adjacency matrix, Laplacian, Lambda-2
    - jamming: FSPL-based jamming disruption logic
"""

from .fspl import (
    fspl_db,
    fspl_linear,
    received_power_watts,
    received_power_dbm,
    db_to_watts,
    watts_to_db,
    compute_comm_range,
    compute_jam_range,
)

from .communication_graph import (
    compute_adjacency_matrix,
    compute_degree_matrix,
    compute_laplacian,
    compute_lambda2,
    is_graph_connected,
)

from .jamming import (
    compute_midpoints,
    compute_jamming_power,
    compute_disrupted_links,
    apply_jamming_to_adjacency,
)
