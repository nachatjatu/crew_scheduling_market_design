import numpy as np
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import pickle

class ACEEI:
    def __init__(self, agents, capacities, budgets0, delta=0.01, epsilon=0.1, t=2, tol=1, max_iter=1000):
        self.agents = agents
        self.capacities = capacities
        self.budgets0 = budgets0
        self.delta = delta
        self.epsilon = epsilon
        self.t = t
        self.max_iter = max_iter
        self.prices = np.zeros_like(self.capacities)
        self.tol = tol

        # Will hold previously discovered constraint pairs
        self.active_constraints = set()     # set of (i, li, j, lj)
        self.full_rescreen_period = 10      # recompute full EF-TB every N iterations



    def clip(self, z):
        # \tilde z_j = z_j if p_j > 0, = max(0, z_j) if p_j = 0
        clipped = np.zeros_like(z)
        for j in range(len(self.capacities)):
            if self.prices[j] > 0:
                clipped[j] = z[j]
            else:
                clipped[j] = max(0, z[j])
        return clipped    

    def run(self):
        best_error = float("inf")
        best_prices = None
        best_budgets = None
        best_bundles = None

        self.prices = np.zeros_like(self.capacities)
        
        for iter in range(self.max_iter):

            # -------------------------------------------------------
            # 1. Enumerate budget subregions
            # -------------------------------------------------------
            all_regions = {}
            for agent in self.agents:
                agent_regions = agent.compute_budget_subregions(
                    self.prices, self.delta, self.epsilon, self.budgets0[agent.id]
                )
                all_regions[agent.id] = agent_regions

            # -------------------------------------------------------
            # 2. Build ILP
            # -------------------------------------------------------
            m = gp.Model("ACEEI_Budget_Perturbation")
            m.Params.OutputFlag = 0

            # integral allocation decision variables
            x = {}
            for agent in self.agents:
                i = agent.id
                for l, subregion in enumerate(all_regions[i]):
                    x[(i, l)] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{l}")

            # clearing variables with pos/neg splits for abs values
            z = {}
            z_pos, z_neg = {}, {}
            for j in range(len(self.capacities)):
                z[j] = m.addVar(lb=-GRB.INFINITY, name=f"z_{j}")
                z_pos[j] = m.addVar(name=f"z+_{j}")
                z_neg[j] = m.addVar(name=f"z-_{j}")
                m.addConstr(z[j] == z_pos[j] - z_neg[j])

            # -------------------------------------------------------
            # 2a. One subregion per agent
            # -------------------------------------------------------
            for agent in self.agents:
                i = agent.id
                m.addConstr(
                    gp.quicksum(x[(i, l)] for l in range(len(all_regions[i]))) == 1
                )

            # -------------------------------------------------------
            # 2b. Market clearing
            # -------------------------------------------------------
            for j in range(len(self.capacities)):
                lhs = gp.quicksum(
                    x[(agent.id, l)] * all_regions[agent.id][l][0][j]
                    for agent in self.agents 
                    for l in range(len(all_regions[agent.id]))
                )

                if self.prices[j] > 0:
                    m.addConstr(lhs == self.capacities[j] + z[j])
                else:
                    m.addConstr(lhs <= self.capacities[j] + z[j])

            # -------------------------------------------------------
            # 2c. EF-TB Constraints (FAST VERSION)
            # -------------------------------------------------------
            free_mask = (self.prices < 1e-6).astype(int)

            if iter % self.full_rescreen_period == 0:
                # Full expensive build
                constrained_pairs = self.screen_eftb_constraints(all_regions, free_mask)
            else:
                # Only re-check constraints that were previously binding
                constrained_pairs = self.rescreen_active_pairs(all_regions, free_mask)

            # Replace active set
            self.active_constraints = constrained_pairs

            # Add to ILP
            for (i, li, j, lj) in constrained_pairs:
                m.addConstr(x[(i, li)] + x[(j, lj)] <= 1)


            # -------------------------------------------------------
            # 2e. Objective
            # -------------------------------------------------------
            m.setObjective(
                gp.quicksum(z_pos[j] + z_neg[j] for j in range(len(self.capacities))),
                GRB.MINIMIZE
            )

            m.optimize()

            # -------------------------------------------------------
            # 2f. Solve
            # -------------------------------------------------------
            x_sol = {}
            for agent in self.agents:
                i = agent.id
                for l in range(len(all_regions[i])):
                    x_sol[(i, l)] = x[(i, l)].X

            z_sol = np.array([z[j].X for j in range(len(self.capacities))])

            # -------------------------------------------------------
            # 2g. Re-solve with minimum norm budget
            # -------------------------------------------------------
        
            z_star = m.objVal

            m.addConstr(
                gp.quicksum(z_pos[j] + z_neg[j] for j in range(len(self.capacities))) == z_star
            )


            m.setObjective(
                gp.quicksum(x[(agent.id,l)] * all_regions[agent.id][l][2]   # region_end = chosen budget
                            for agent in self.agents
                            for l in range(len(all_regions[agent.id]))),
                GRB.MINIMIZE
            )

            m.optimize()
            
            # -------------------------------------------------------
            # 2g. Extract chosen budgets and bundles 
            # -------------------------------------------------------
            perturbed_budgets = {}
            bundles = {}

            for agent in self.agents:
                i = agent.id
                chosen_l = None
                for l in range(len(all_regions[i])):
                    if x[(i, l)].X > 0.5:     # chosen subregion
                        chosen_l = l
                        break

                assert chosen_l is not None, f"No subregion chosen for agent {i}"

                prev_bundle, region_start, region_end = all_regions[i][chosen_l]

                bundles[i] = np.array(prev_bundle)     # the actual bundle
                perturbed_budgets[i] = region_end      # chosen budget in that subregion

            # -------------------------------------------------------
            # 3. Termination check
            # -------------------------------------------------------
            # if |\tilde z|_2 = 0, terminate with p* = p, b* = b
            clipped = self.clip(z_sol)
            clearing_error = np.linalg.norm(clipped)

            # -------------------------
            # Save best iterate so far
            # -------------------------
            if clearing_error < best_error:
                best_error = clearing_error
                best_prices = self.prices.copy()
                best_budgets = perturbed_budgets.copy()
                best_bundles = {i: bundles[i].copy() for i in bundles}

            if iter % 10 == 0:
                print(f'== Iteration {iter} ==')
                print(f'Prices: {self.prices}')
                print(f'Excess Demand: {z_sol}')
                print(f'Clipped Excess Demand: {clipped}')
                print(f'Clearing Error: {clearing_error}\n')

            if clearing_error <= self.tol:
                print(f"== A-CEEI FOUND at iter {iter} ==")
                print(f'Excess Demand: {z_sol}')
                print(f'Clipped Excess Demand: {clipped}')
                print(f'Clearing Error: {clearing_error}')
                return self.prices, perturbed_budgets, bundles

            # -------------------------------------------------------
            # 4. Tatonnement
            # -------------------------------------------------------
            # p <- p + delta * \tilde z
            self.prices = self.prices + self.delta * clipped

        print("Reached max iterations. Returning best solution found.\n")
        return best_prices, best_budgets, best_bundles


    def violates_eftb_contested_fast(self, agent_i, bundle_i, superbundle_i_j):
        """
        Check u_i(superbundle_j) > u_i(bundle_i)
        Uses additive utilities only. Assumes bundle validity already checked.
        """
        ui_bundle = agent_i.utility_fast(bundle_i)
        ui_super = agent_i.utility_fast(superbundle_i_j)

        return (ui_super > ui_bundle + 1e-12)
    

    def screen_eftb_constraints(self, all_regions, free_mask):
        """
        Returns a set of constraint pairs (i, li, j, lj) that violate contested EF-TB.
        Runs full O(n^2 * k^2) check but with skipping optimizations.
        """
        constrained_pairs = set()
        prices = self.prices
        priority = self.budgets0

        for agent_i in self.agents:
            i = agent_i.id

            for agent_j in self.agents:
                j = agent_j.id
                if i == j:
                    continue

                # Only enforce when i has higher priority
                if priority[i] < priority[j]:
                    continue

                # loops over subregions
                for li, (bundle_i, bi_lo, bi_hi) in enumerate(all_regions[i]):
                    ui = agent_i.utility_fast(bundle_i)
                    bi_budget = bi_hi    # max budget allowed for region

                    for lj, (bundle_j, bj_lo, bj_hi) in enumerate(all_regions[j]):
                        
                        # Build contested superbundle
                        superbundle = np.maximum(bundle_j, free_mask)

                        # --- Skip Condition A: invalid bundle ---
                        if not agent_i.valid(superbundle):
                            continue

                        # --- Skip Condition B: too expensive to afford ---
                        if prices.dot(superbundle) > bi_budget + 1e-9:
                            continue

                        # --- Actual contested EF-TB check ---
                        if self.violates_eftb_contested_fast(agent_i, bundle_i, superbundle):
                            constrained_pairs.add((i, li, j, lj))

        return constrained_pairs
    
    def rescreen_active_pairs(self, all_regions, free_mask):
        """
        Only re-check constraints that were active previously.
        Much cheaper than full screening.
        """
        new_pairs = set()
        prices = self.prices
        priority = self.budgets0

        for (i, li, j, lj) in self.active_constraints:

            # Recompute superbundle
            bundle_i, bi_lo, bi_hi = all_regions[i][li]
            bundle_j, bj_lo, bj_hi = all_regions[j][lj]
            agent_i = self.agents[i]
            superbundle = np.maximum(bundle_j, free_mask)

            # Check skip rules again
            if not agent_i.valid(superbundle):
                continue

            if prices.dot(superbundle) > bi_hi + 1e-9:
                continue

            # Check actual violation again
            if self.violates_eftb_contested_fast(agent_i, bundle_i, superbundle):
                new_pairs.add((i, li, j, lj))

        return new_pairs
