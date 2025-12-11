import numpy as np
import gurobipy as gp
from gurobipy import GRB
from multiprocessing import Pool

# ------------------------------------------------------------
# Global objects inside each worker
# ------------------------------------------------------------
_worker_model = None
_worker_x = None
_worker_budget_constr = None
_worker_n = None
_worker_conflicts = None
_worker_utilities = None


# ------------------------------------------------------------
# 1. INITIALIZER — runs ONCE per worker
# ------------------------------------------------------------
def demand_initializer(utilities, conflicts):
    """
    Build a persistent knapsack Gurobi model inside each worker.
    """
    global _worker_model, _worker_x, _worker_budget_constr
    global _worker_n, _worker_conflicts, _worker_utilities

    _worker_utilities = np.array(utilities)
    _worker_conflicts = conflicts
    _worker_n = len(utilities)

    m = gp.Model()
    m.Params.OutputFlag = 0

    x = m.addVars(_worker_n, vtype=GRB.BINARY, name="x")

    # Objective (fixed)
    m.setObjective(gp.quicksum(_worker_utilities[j] * x[j] for j in range(_worker_n)),
                   GRB.MAXIMIZE)

    # Budget constraint placeholder with price coefficients = 0 initially
    price_expr = gp.LinExpr(0.0)
    for j in range(_worker_n):
        price_expr += 0.0 * x[j]

    budget_constr = m.addConstr(price_expr <= 0.0)

    # Add conflicts once
    for (j, k) in conflicts:
        m.addConstr(x[j] + x[k] <= 1)

    m.update()

    _worker_model = m
    _worker_x = x
    _worker_budget_constr = budget_constr


# ------------------------------------------------------------
# 2. DEMAND FUNCTION — reuses persistent model per call
# ------------------------------------------------------------
def demand_solve(args):
    """
    Solve knapsack demand using persistent worker model.
    args = (prices, budget)
    """
    global _worker_model, _worker_x, _worker_budget_constr
    global _worker_n

    prices, budget = args

    # Update price coefficients
    for j in range(_worker_n):
        _worker_model.chgCoeff(_worker_budget_constr, _worker_x[j], float(prices[j]))

    # Update RHS
    _worker_budget_constr.setAttr(GRB.Attr.RHS, float(budget))

    _worker_model.update()
    _worker_model.optimize()

    bundle = np.zeros(_worker_n, dtype=int)
    for j in range(_worker_n):
        bundle[j] = int(round(_worker_x[j].X))

    return bundle


# ------------------------------------------------------------
# 3. DemandPool manager
# ------------------------------------------------------------
class DemandPool:
    def __init__(self, utilities, conflicts, processes=4):
        self.processes = processes

        # Build worker processes, each with its own persistent model
        self.pool = Pool(
            processes=processes,
            initializer=demand_initializer,
            initargs=(utilities, conflicts)
        )

    def solve_many(self, prices, budgets):
        """
        prices: 1D price vector
        budgets: list or array of budgets
        """
        args_list = [(prices, float(b)) for b in budgets]
        return self.pool.map(demand_solve, args_list)

    def close(self):
        self.pool.close()
        self.pool.join()
