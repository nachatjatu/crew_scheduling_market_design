import numpy as np

def serial_dictatorship(preferences, order, lines):
    """
    preferences: 2D array where preferences[i] is the preference list of crew i
    order: list of crew indices indicating the order in which they pick
    lines: list of available lines

    Returns a matching dict {crew: line}
    """
    available_lines = set(lines.keys())
    matching = {}

    for crew in order:
        pref_list = preferences[crew]
        for line in pref_list:
            if line in available_lines:
                matching[crew] = line
                available_lines.remove(line)
                break

    return matching


def random_serial_dictatorship(preferences, lines, rng=None):
    """
    rng: np.random.Generator
    """
    n_crews = preferences.shape[0]
    
    if rng is None:
        rng = np.random.default_rng()
        
    order = rng.permutation(n_crews)
    return serial_dictatorship(preferences, order, lines)


def banded_permutation(seniority_order, k, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    
    n = len(seniority_order)
    permuted = []

    for start in range(0, n, k):
        band = np.array(seniority_order[start:start+k])  # copy, not view
        rng.shuffle(band)
        permuted.extend(band)
        
    return np.array(permuted)



def k_band_serial_dictatorship(preferences, seniority_order, lines, k, rng=None):
    permuted_order = banded_permutation(seniority_order, k, rng)
    return serial_dictatorship(preferences, permuted_order, lines), permuted_order


def epsilon_shuffle_order(seniority_order, eps, rng):
    n = len(seniority_order)
    
    ranks = np.arange(n)                    # 0,1,2,...,n-1
    noise = rng.normal(0, 1, size=n)        # independent Gaussian noise
    
    perturbed = ranks + eps * noise         # add noise to ranks
    order = np.argsort(perturbed)           # re-sort crews by perturbed rank
    return seniority_order[order]



def epsilon_serial_dictatorship(preferences, seniority_order, lines, eps, rng):
    n = len(seniority_order)

    # correct: apply noise to ranks, not crew numbers
    ranks = np.arange(n)
    noise = rng.normal(0, 1, size=n)

    perturbed = ranks + eps * noise
    new_order = seniority_order[np.argsort(perturbed)]
    
    return serial_dictatorship(preferences, new_order, lines), new_order



