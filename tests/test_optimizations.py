
import numpy as np
import pandas as pd
import pytest
from pymarxan.models.problem import ConservationProblem
from pymarxan.solvers.cache import ProblemCache

def test_conservation_problem_validation():
    """Test that duplicate IDs are rejected."""
    df = pd.DataFrame({"id": [1, 1], "cost": [10, 20], "status": [0, 0]})
    feats = pd.DataFrame({"id": [1], "target": [10], "spf": [1], "name": ["f1"]})
    puvs = pd.DataFrame({"species": [1], "pu": [1], "amount": [5]})
    
    with pytest.raises(ValueError, match="planning_units\\['id'\\] contains duplicate values"):
        ConservationProblem(df, feats, puvs)

def test_cache_csr_structure():
    """Test that ProblemCache correctly builds CSR adjacency."""
    # simple 3-node line graph: 1--2--3
    pus = pd.DataFrame({"id": [1, 2, 3], "cost": [1, 1, 1], "status": [0, 0, 0]})
    feats = pd.DataFrame({"id": [1], "target": [1], "spf": [1], "name": ["f1"]})
    puvs = pd.DataFrame({"species": [1, 1, 1], "pu": [1, 2, 3], "amount": [1, 1, 1]})
    
    # Boundary: (1,2) weight 10, (2,3) weight 20
    bnd = pd.DataFrame({
        "id1": [1, 2],
        "id2": [2, 3],
        "boundary": [10.0, 20.0]
    })
    
    prob = ConservationProblem(pus, feats, puvs, boundary=bnd)
    cache = ProblemCache.from_problem(prob)
    
    # Neighbors of 1 (index 0): 2 (index 1) with weight 10
    # Neighbors of 2 (index 1): 1 (index 0) w 10, 3 (index 2) w 20
    # Neighbors of 3 (index 2): 2 (index 1) w 20
    
    # Check adj_start
    # [start(0), start(1), start(2), start(3)]
    # 0 -> 0
    # 1 -> 1 (0+1)
    # 2 -> 3 (1+2)
    # 3 -> 4 (3+1)
    assert np.array_equal(cache.adj_start, [0, 1, 3, 4])
    
    # Check indices and weights
    # Node 0 neighbors: [1]
    assert cache.adj_indices[0] == 1
    assert cache.adj_weights[0] == 10.0
    
    # Node 1 neighbors: [0, 2] (sorted order from code)
    assert np.all(np.isin(cache.adj_indices[1:3], [0, 2]))
    w_sum = np.sum(cache.adj_weights[1:3])
    assert w_sum == 30.0 # 10 + 20
    
    # Check boundary computation
    # Select 1 and 2. Shared boundary is 10. External is 0 (no self-boundary).
    # Total boundary should be 0 because they share the edge and both are selected?
    # No, standard Marxan boundary definition:
    # boundary = sum(boundary_cost(i, j) for edges (i,j) where selected[i] != selected[j])
    # + sum(boundary_cost(i, i) if selected[i])
    
    # Select 1 only: Boundary = edge(1,2) = 10.
    sel = np.array([True, False, False])
    b = cache._compute_boundary(sel)
    assert b == 10.0
    
    # Select 1 and 2: Boundary = edge(2,3) = 20. (Edge 1-2 is internal now)
    sel = np.array([True, True, False])
    b = cache._compute_boundary(sel)
    assert b == 20.0
