import numpy as np
import gurobipy as gp
from gurobipy import GRB
from demand_pool import DemandPool

gp.setParam('LogToConsole', 0)

class Crew:
    def __init__(self, id, utilities, budget0, conflicts=None):
        """
        utilities[j]: utility of item j
        budget0: initial budget b_i^0
        conflicts: list of (j,k) item pairs that cannot be taken together
        """
        self.id = id
        self.utilities = np.array(utilities)
        self.n_items = len(utilities)
        self.conflicts = conflicts if conflicts is not None else []

        # -------------------------------
        # Build a persistent knapsack MIP
        # -------------------------------
        m = gp.Model(f"agent_{id}")
        m.Params.OutputFlag = 0

        # binary variables x_j
        self.x = m.addVars(self.n_items, vtype=GRB.BINARY)

        # objective = maximize sum_j u_ij x_j
        m.setObjective(gp.quicksum(self.utilities[j] * self.x[j]
                                   for j in range(self.n_items)),
                       GRB.MAXIMIZE)

        # price*quantity <= budget   (will be updated each solve)
        self.price_expr = gp.LinExpr(0.0)  # we will dynamically adjust coefficients
        for j in range(self.n_items):
            self.price_expr += 0.0 * self.x[j]

        self.budget_constr = m.addConstr(self.price_expr <= budget0)

        # add conflict constraints once
        for (j, k) in self.conflicts:
            m.addConstr(self.x[j] + self.x[k] <= 1)

        m.update()
        self.model = m


    # ============================================================
    #   FAST UTILITIES (for EF-TB speedups)
    # ============================================================

    def utility_fast(self, bundle):
        """Just dot product; bundle is binary array."""
        return float(np.dot(self.utilities, bundle))

    def valid(self, bundle):
        """Check if bundle violates conflicts."""
        for j, k in self.conflicts:
            if bundle[j] == 1 and bundle[k] == 1:
                return False
        return True


    # ============================================================
    #   Subregion computation (unchanged logic, faster demand())
    # ============================================================

    def compute_budget_subregions(self, prices, delta, epsilon, budget0):

        b_min = budget0 - epsilon
        b_max = budget0 + epsilon
        budgets = np.arange(b_min, b_max + 1e-12, delta)

        # Parallel demand using persistent workers
        if not hasattr(self, "demand_pool"):
            self.demand_pool = DemandPool(
                utilities=self.utilities,
                conflicts=self.conflicts,
                processes=8   # choose appropriate number
            )

        bundles_list = self.demand_pool.solve_many(prices, budgets)

        # Build subregions as before
        regions = []
        prev_bundle = None
        region_start = None

        for b, bundle in zip(budgets, bundles_list):
            bundle_tuple = tuple(bundle)

            if prev_bundle is None:
                prev_bundle = bundle_tuple
                region_start = b
                continue

            if bundle_tuple != prev_bundle:
                regions.append((np.array(prev_bundle, dtype=int),
                                region_start, b - delta))
                prev_bundle = bundle_tuple
                region_start = b

        regions.append((np.array(prev_bundle, dtype=int),
                        region_start, budgets[-1]))

        return regions