class BidlineSimulator:
    def __init__(self, crew_list, pairings, rule):
        self.crew_list = crew_list
        self.pairings = pairings
        self.rule = rule
        self.allocations = None

    def run(self):
        self.allocations = self.rule.allocate(self.crew_list, self.pairings)

    def evaluate_fairness(self, metric="envy"):
        pass

    def evaluate_efficiency(self):
        pass