import numpy as np

def displacement_metrics(order_generator, seniority_order, trials, rng):
    """
    Computes both absolute and signed displacement for each seniority rank.
    Returns: abs_dev, signed_dev
    """

    n = len(seniority_order)
    abs_dev = np.zeros(n)
    signed_dev = np.zeros(n)

    for _ in range(trials):
        order = np.array(order_generator(rng), dtype=int)

        # pos[crew] = position where that crew picks
        pos = np.empty(n, dtype=int)
        for pick_pos, crew in enumerate(order):
            pos[crew] = pick_pos

        diff = pos - np.arange(n)  # signed displacement
        abs_dev   += np.abs(diff)
        signed_dev += diff

    return abs_dev / trials, signed_dev / trials

def total_utility(matching, utilities):
    """
    matching: dict {crew: line}
    utilities: 2D array where utilities[i][j] is the utility of crew i for line j

    Returns the total utility of the matching
    """
    total_util = 0
    for crew, line in matching.items():
        total_util += utilities[crew][line]
    return total_util


def utilities_per_crew(matching, U):
    """
    matching: can be a dict {crew: line} or a numpy array of shape (NUM_CREWS,)
    U: utility matrix of shape (NUM_CREWS, NUM_LINES)

    returns: utilities[i] = U[i, matching[i]]
    """
    if isinstance(matching, dict):
        # Convert dict to array
        crews = sorted(matching.keys())
        assigned_lines = np.array([matching[c] for c in crews])
    else:
        # Already an array
        assigned_lines = matching
    
    utilities = U[np.arange(len(assigned_lines)), assigned_lines]
    return utilities

def gini(values):
    """
    Compute Gini coefficient of a 1D numpy array.
    Works for utilities or ranks (although Gini on ranks is especially meaningful).
    """
    x = np.array(values).flatten()
    n = len(x)
    if n == 0:
        return 0.0
    mean = np.mean(x)
    if mean == 0:
        return 0.0
    
    # Sum of absolute pairwise differences
    diff_sum = np.abs(x[:, None] - x[None, :]).sum()
    
    return diff_sum / (2 * n * n * mean)

def justified_envy(seniority_order, U, matches):
    justified_envy_by_rank = []
    for p_i, crew_i in enumerate(seniority_order):
        justified_envy = 0
        own_line = matches[crew_i]
        own_utilities = U[crew_i, :]
        for crew_j in seniority_order[p_i:]:
            other_line = matches[crew_j]
            if own_utilities[other_line] > own_utilities[own_line]:
                justified_envy += 1
        justified_envy_by_rank.append(justified_envy)
    return np.array(justified_envy_by_rank)
