"""
Unit Tests for Communication Graph Module
==========================================

Tests the adjacency matrix, Laplacian, and lambda-2 (Fiedler value)
calculations as specified in PROJECT_MASTER_GUIDE_v2.md Sections 3.1-3.5.

Key tests:
    - Adjacency matrix matches FSPL-based link determination
    - Laplacian has correct mathematical properties
    - Lambda-2 correctly identifies connectivity/disconnection
    - Proposition 1: lambda-2 = 0 iff graph is disconnected

Run with: python -m pytest tests/test_laplacian.py -v
Or standalone: python tests/test_laplacian.py

Author: MARL Jammer Team
"""

import sys
import os
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from physics.communication_graph import (
    compute_adjacency_matrix,
    compute_adjacency_from_distances,
    compute_degree_matrix,
    compute_laplacian,
    compute_lambda2,
    compute_all_eigenvalues,
    compute_lambda2_from_positions,
    is_graph_connected,
    count_connected_components,
    get_edge_count,
    compute_graph_density,
    verify_laplacian_properties,
)

from physics.fspl import compute_comm_range


class TestAdjacencyMatrix:
    """Test adjacency matrix construction."""
    
    def test_empty_graph(self):
        """Empty positions should return empty adjacency."""
        positions = np.array([]).reshape(0, 2)
        A = compute_adjacency_matrix(positions, 0.1, 1e-12, 2.4e9)
        assert A.shape == (0, 0)
    
    def test_single_node(self):
        """Single node should have 1x1 zero adjacency."""
        positions = np.array([[0, 0]])
        A = compute_adjacency_matrix(positions, 0.1, 1e-12, 2.4e9)
        assert A.shape == (1, 1)
        assert A[0, 0] == 0  # No self-loops
    
    def test_two_close_nodes(self):
        """Two close nodes should be connected."""
        # Use a distance well within comm range
        # At 2.4 GHz with FSPL, comm range is ~3145m (pure free space)
        positions = np.array([[0, 0], [100, 0]])  # 100m apart - definitely connected
        A = compute_adjacency_matrix(positions, 0.1, 1e-12, 2.4e9)
        
        assert A.shape == (2, 2)
        assert A[0, 0] == 0  # No self-loop
        assert A[1, 1] == 0  # No self-loop
        assert A[0, 1] == 1  # Connected
        assert A[1, 0] == 1  # Symmetric
    
    def test_two_far_nodes(self):
        """Two far nodes should be disconnected."""
        # Need to be beyond FSPL comm range (~3145m at 2.4 GHz)
        positions = np.array([[0, 0], [5000, 0]])  # 5000m apart
        A = compute_adjacency_matrix(positions, 0.1, 1e-12, 2.4e9)
        
        assert A[0, 1] == 0  # Not connected
        assert A[1, 0] == 0  # Symmetric
    
    def test_adjacency_symmetric(self):
        """Adjacency matrix should be symmetric."""
        positions = np.random.rand(10, 2) * 100
        A = compute_adjacency_matrix(positions, 0.1, 1e-12, 2.4e9)
        
        assert np.allclose(A, A.T), "Adjacency should be symmetric"
    
    def test_adjacency_no_self_loops(self):
        """Diagonal should be zero (no self-loops)."""
        positions = np.random.rand(5, 2) * 100
        A = compute_adjacency_matrix(positions, 0.1, 1e-12, 2.4e9)
        
        assert np.all(np.diag(A) == 0), "Diagonal should be all zeros"
    
    def test_adjacency_binary(self):
        """Adjacency values should be 0 or 1."""
        positions = np.random.rand(10, 2) * 100
        A = compute_adjacency_matrix(positions, 0.1, 1e-12, 2.4e9)
        
        assert np.all((A == 0) | (A == 1)), "Values should be 0 or 1"
    
    def test_complete_graph_close_nodes(self):
        """All close nodes should form complete graph."""
        # 5 nodes in 10x10 area - all within 86m comm range
        positions = np.array([
            [0, 0], [5, 0], [10, 0], [5, 5], [5, 10]
        ])
        A = compute_adjacency_matrix(positions, 0.1, 1e-12, 2.4e9)
        
        # Complete graph K_5 has 5*4/2 = 10 edges
        num_edges = get_edge_count(A)
        assert num_edges == 10, f"Expected 10 edges, got {num_edges}"
    
    def test_frequency_affects_connectivity(self):
        """Higher frequency should have shorter range."""
        # At a distance where 433 MHz connects but 5.8 GHz might not
        # FSPL gives much longer ranges than expected, so we need very far distances
        # Instead, test that connectivity changes appropriately
        positions = np.array([[0, 0], [2000, 0]])  # 2000m apart
        
        A_433 = compute_adjacency_matrix(positions, 0.1, 1e-12, 4.33e8)  # 433 MHz
        A_5800 = compute_adjacency_matrix(positions, 0.1, 1e-12, 5.8e9)  # 5.8 GHz
        
        # 433 MHz has longer range than 5.8 GHz
        # At 2000m, 433 MHz should likely connect, 5.8 GHz less likely
        # Test that range behavior is frequency-dependent
        R_433 = compute_comm_range(0.1, 1e-12, 4.33e8)
        R_5800 = compute_comm_range(0.1, 1e-12, 5.8e9)
        
        assert R_433 > R_5800, "433 MHz should have longer range"
        # At any distance < R_5800, both should connect
        # At R_5800 < d < R_433, only 433 MHz connects
        mid_distance = (R_433 + R_5800) / 2
        positions_mid = np.array([[0, 0], [mid_distance, 0]])
        
        A_433_mid = compute_adjacency_matrix(positions_mid, 0.1, 1e-12, 4.33e8)
        A_5800_mid = compute_adjacency_matrix(positions_mid, 0.1, 1e-12, 5.8e9)
        
        assert A_433_mid[0, 1] == 1, "433 MHz should connect at mid-distance"
        assert A_5800_mid[0, 1] == 0, "5.8 GHz should NOT connect at mid-distance"
    
    def test_jamming_mask(self):
        """Jamming mask should remove edges."""
        positions = np.array([[0, 0], [50, 0], [100, 0]])
        
        # Without jamming
        A_clean = compute_adjacency_matrix(positions, 0.1, 1e-12, 2.4e9)
        
        # With jamming on link (0, 1)
        jammed = np.zeros((3, 3), dtype=bool)
        jammed[0, 1] = True
        jammed[1, 0] = True
        
        A_jammed = compute_adjacency_matrix(positions, 0.1, 1e-12, 2.4e9, jammed)
        
        assert A_clean[0, 1] == 1, "Original should have edge (0,1)"
        assert A_jammed[0, 1] == 0, "Jammed should NOT have edge (0,1)"


class TestDegreeMatrix:
    """Test degree matrix computation."""
    
    def test_degree_diagonal(self):
        """Degree matrix should be diagonal."""
        A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])  # Triangle
        D = compute_degree_matrix(A)
        
        # Should be diagonal
        assert np.all(D - np.diag(np.diag(D)) == 0), "D should be diagonal"
    
    def test_degree_values(self):
        """Degree should equal number of neighbors."""
        A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])  # Triangle
        D = compute_degree_matrix(A)
        
        # Each node has degree 2
        assert np.all(np.diag(D) == 2), "Each node should have degree 2"
    
    def test_degree_star_graph(self):
        """Star graph: center has degree N-1, others have degree 1."""
        # Star: node 0 connected to all others
        N = 5
        A = np.zeros((N, N))
        for i in range(1, N):
            A[0, i] = 1
            A[i, 0] = 1
        
        D = compute_degree_matrix(A)
        
        assert D[0, 0] == N - 1, f"Center should have degree {N-1}"
        for i in range(1, N):
            assert D[i, i] == 1, f"Leaf {i} should have degree 1"


class TestLaplacian:
    """Test Laplacian matrix computation and properties."""
    
    def test_laplacian_formula(self):
        """L = D - A."""
        A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])  # Path
        D = compute_degree_matrix(A)
        L = compute_laplacian(A)
        
        assert np.allclose(L, D - A), "L should equal D - A"
    
    def test_laplacian_symmetric(self):
        """Laplacian should be symmetric."""
        A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        L = compute_laplacian(A)
        
        assert np.allclose(L, L.T), "Laplacian should be symmetric"
    
    def test_laplacian_row_sum_zero(self):
        """Row sums of Laplacian should be zero."""
        A = np.array([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0], [0, 1, 0, 0]])
        L = compute_laplacian(A)
        
        row_sums = np.sum(L, axis=1)
        assert np.allclose(row_sums, 0), "Row sums should be zero"
    
    def test_laplacian_positive_semidefinite(self):
        """All eigenvalues should be >= 0."""
        A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        L = compute_laplacian(A)
        
        eigenvalues = compute_all_eigenvalues(L)
        assert np.all(eigenvalues >= -1e-10), "Eigenvalues should be non-negative"
    
    def test_laplacian_smallest_eigenvalue_zero(self):
        """Smallest eigenvalue should be 0."""
        A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        L = compute_laplacian(A)
        
        eigenvalues = compute_all_eigenvalues(L)
        assert abs(eigenvalues[0]) < 1e-10, "Smallest eigenvalue should be 0"
    
    def test_verify_properties_passes(self):
        """verify_laplacian_properties should pass for valid Laplacian."""
        A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        L = compute_laplacian(A)
        
        result = verify_laplacian_properties(L)
        assert result["valid"], f"Properties check failed: {result}"


class TestLambda2:
    """Test lambda-2 (Fiedler value) computation - THE KEY METRIC."""
    
    def test_lambda2_empty_graph(self):
        """Empty graph should have lambda-2 = 0."""
        L = np.array([]).reshape(0, 0)
        assert compute_lambda2(L) == 0.0
    
    def test_lambda2_single_node(self):
        """Single node should have lambda-2 = 0."""
        L = np.array([[0]])
        assert compute_lambda2(L) == 0.0
    
    def test_lambda2_two_connected_nodes(self):
        """Two connected nodes: L has eigenvalues [0, 2], so lambda-2 = 2."""
        A = np.array([[0, 1], [1, 0]])
        L = compute_laplacian(A)
        
        lambda2 = compute_lambda2(L)
        assert abs(lambda2 - 2.0) < 1e-10, f"Expected 2.0, got {lambda2}"
    
    def test_lambda2_two_disconnected_nodes(self):
        """Two disconnected nodes should have lambda-2 = 0."""
        A = np.array([[0, 0], [0, 0]])
        L = compute_laplacian(A)
        
        lambda2 = compute_lambda2(L)
        assert abs(lambda2) < 1e-10, f"Expected 0, got {lambda2}"
    
    def test_lambda2_complete_graph(self):
        """Complete graph K_n has lambda-2 = n (all eigenvalues except 0 are n)."""
        for n in [3, 4, 5, 6]:
            A = np.ones((n, n)) - np.eye(n)
            L = compute_laplacian(A)
            
            lambda2 = compute_lambda2(L)
            assert abs(lambda2 - n) < 1e-10, f"K_{n}: expected {n}, got {lambda2}"
    
    def test_lambda2_path_graph(self):
        """Path graph: lambda-2 = 2(1 - cos(pi/n))."""
        n = 5
        A = np.zeros((n, n))
        for i in range(n - 1):
            A[i, i+1] = 1
            A[i+1, i] = 1
        L = compute_laplacian(A)
        
        lambda2 = compute_lambda2(L)
        expected = 2 * (1 - np.cos(np.pi / n))
        assert abs(lambda2 - expected) < 1e-6, f"Expected {expected}, got {lambda2}"
    
    def test_lambda2_cycle_graph(self):
        """Cycle graph: lambda-2 = 2(1 - cos(2*pi/n))."""
        n = 6
        A = np.zeros((n, n))
        for i in range(n):
            A[i, (i+1) % n] = 1
            A[(i+1) % n, i] = 1
        L = compute_laplacian(A)
        
        lambda2 = compute_lambda2(L)
        expected = 2 * (1 - np.cos(2 * np.pi / n))
        assert abs(lambda2 - expected) < 1e-6, f"Expected {expected}, got {lambda2}"


class TestProposition1:
    """
    Test Proposition 1 from the project guide:
    Lambda-2 = 0 if and only if the graph is disconnected.
    
    This is the theoretical foundation of the entire project!
    """
    
    def test_connected_implies_lambda2_positive(self):
        """If graph is connected, then lambda-2 > 0."""
        # Connected graphs of various types
        
        # Complete graph
        A = np.ones((5, 5)) - np.eye(5)
        assert is_graph_connected(A), "Complete graph should be connected"
        L = compute_laplacian(A)
        assert compute_lambda2(L) > 0, "Connected graph should have lambda-2 > 0"
        
        # Path graph
        A = np.zeros((5, 5))
        for i in range(4):
            A[i, i+1] = 1
            A[i+1, i] = 1
        assert is_graph_connected(A), "Path graph should be connected"
        L = compute_laplacian(A)
        assert compute_lambda2(L) > 0, "Connected graph should have lambda-2 > 0"
        
        # Star graph
        A = np.zeros((5, 5))
        for i in range(1, 5):
            A[0, i] = 1
            A[i, 0] = 1
        assert is_graph_connected(A), "Star graph should be connected"
        L = compute_laplacian(A)
        assert compute_lambda2(L) > 0, "Connected graph should have lambda-2 > 0"
    
    def test_disconnected_implies_lambda2_zero(self):
        """If graph is disconnected, then lambda-2 = 0."""
        # Two isolated nodes
        A = np.zeros((2, 2))
        assert not is_graph_connected(A), "No edges = disconnected"
        L = compute_laplacian(A)
        assert abs(compute_lambda2(L)) < 1e-10, "Disconnected should have lambda-2 = 0"
        
        # Two separate components
        A = np.array([
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ])
        assert not is_graph_connected(A), "Two components = disconnected"
        L = compute_laplacian(A)
        assert abs(compute_lambda2(L)) < 1e-10, "Disconnected should have lambda-2 = 0"
    
    def test_lambda2_decreases_as_edges_removed(self):
        """As edges are removed, lambda-2 should decrease or stay same."""
        # Start with complete graph
        A = np.ones((6, 6)) - np.eye(6)
        L = compute_laplacian(A)
        prev_lambda2 = compute_lambda2(L)
        
        # Remove edges one by one
        edges_to_remove = [(0, 1), (0, 2), (1, 2), (3, 4)]
        for i, j in edges_to_remove:
            A[i, j] = 0
            A[j, i] = 0
            L = compute_laplacian(A)
            curr_lambda2 = compute_lambda2(L)
            assert curr_lambda2 <= prev_lambda2 + 1e-10, \
                f"Lambda-2 should not increase when removing edge ({i},{j})"
            prev_lambda2 = curr_lambda2
    
    def test_swarm_fragmentation_scenario(self):
        """
        Simulate a realistic scenario:
        - Start with connected swarm
        - Progressively jam links
        - Verify lambda-2 decreases to 0 when fully fragmented
        """
        # 10 drones in 100x100 area, most connected at 2.4GHz
        np.random.seed(42)
        positions = np.random.rand(10, 2) * 80  # 80x80 to ensure connectivity
        
        # Initial graph
        A = compute_adjacency_matrix(positions, 0.1, 1e-12, 2.4e9)
        L = compute_laplacian(A)
        lambda2_initial = compute_lambda2(L)
        
        assert lambda2_initial > 0, "Initial swarm should be connected"
        
        # Simulate progressive jamming by removing edges
        A_jammed = A.copy()
        edges = []
        for i in range(10):
            for j in range(i+1, 10):
                if A[i, j] == 1:
                    edges.append((i, j))
        
        # Remove half the edges
        np.random.shuffle(edges)
        for i, j in edges[:len(edges)//2]:
            A_jammed[i, j] = 0
            A_jammed[j, i] = 0
        
        L_jammed = compute_laplacian(A_jammed)
        lambda2_jammed = compute_lambda2(L_jammed)
        
        assert lambda2_jammed < lambda2_initial, \
            "Jamming should reduce lambda-2"
        
        # Remove all edges
        A_fragmented = np.zeros((10, 10))
        L_fragmented = compute_laplacian(A_fragmented)
        lambda2_fragmented = compute_lambda2(L_fragmented)
        
        assert abs(lambda2_fragmented) < 1e-10, \
            "Fully fragmented swarm should have lambda-2 = 0"


class TestConnectivityUtilities:
    """Test connectivity analysis utilities."""
    
    def test_is_connected_true(self):
        """is_graph_connected should return True for connected graphs."""
        A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])  # Triangle
        assert is_graph_connected(A) == True
    
    def test_is_connected_false(self):
        """is_graph_connected should return False for disconnected graphs."""
        A = np.array([[0, 0], [0, 0]])  # Two isolated nodes
        assert is_graph_connected(A) == False
    
    def test_count_components_one(self):
        """count_connected_components should return 1 for connected graph."""
        A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        assert count_connected_components(A) == 1
    
    def test_count_components_two(self):
        """count_connected_components should return 2 for two components."""
        A = np.array([
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ])
        assert count_connected_components(A) == 2
    
    def test_count_components_all_isolated(self):
        """All isolated nodes = N components."""
        A = np.zeros((5, 5))
        assert count_connected_components(A) == 5
    
    def test_edge_count(self):
        """get_edge_count should return correct number."""
        A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])  # Triangle = 3 edges
        assert get_edge_count(A) == 3
    
    def test_graph_density(self):
        """compute_graph_density should return correct value."""
        # Complete graph K_4 has density 1.0
        A = np.ones((4, 4)) - np.eye(4)
        assert abs(compute_graph_density(A) - 1.0) < 1e-10
        
        # Empty graph has density 0.0
        A = np.zeros((4, 4))
        assert abs(compute_graph_density(A)) < 1e-10


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_lambda2_from_positions(self):
        """compute_lambda2_from_positions should work correctly."""
        positions = np.array([[0, 0], [50, 0], [25, 25]])  # Close triangle
        
        lambda2, A, L = compute_lambda2_from_positions(
            positions, 0.1, 1e-12, 2.4e9
        )
        
        assert lambda2 > 0, "Close drones should form connected graph"
        assert A.shape == (3, 3)
        assert L.shape == (3, 3)


def run_all_tests():
    """Run all tests and report results."""
    test_classes = [
        TestAdjacencyMatrix,
        TestDegreeMatrix,
        TestLaplacian,
        TestLambda2,
        TestProposition1,
        TestConnectivityUtilities,
        TestConvenienceFunctions,
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        print(f"\n{'='*60}")
        print(f"Running {test_class.__name__}")
        print('='*60)
        
        instance = test_class()
        methods = [m for m in dir(instance) if m.startswith('test_')]
        
        for method_name in methods:
            total_tests += 1
            method = getattr(instance, method_name)
            try:
                method()
                print(f"  [PASS] {method_name}")
                passed_tests += 1
            except AssertionError as e:
                print(f"  [FAIL] {method_name}: {e}")
                failed_tests.append(f"{test_class.__name__}.{method_name}")
            except Exception as e:
                print(f"  [ERROR] {method_name}: {type(e).__name__}: {e}")
                failed_tests.append(f"{test_class.__name__}.{method_name}")
    
    print(f"\n{'='*60}")
    print(f"Communication Graph Tests: {passed_tests}/{total_tests} passed")
    print('='*60)
    
    if failed_tests:
        print("\nFailed tests:")
        for t in failed_tests:
            print(f"  - {t}")
        return False
    else:
        print("\nAll tests PASSED!")
        return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
