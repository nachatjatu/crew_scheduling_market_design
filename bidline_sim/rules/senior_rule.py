from rules.allocation_rule import AllocationRule

class SeniorRule(AllocationRule):
    def allocate(self, crew_list, pairings):
        crew_sorted = sorted(crew_list, key=lambda c: c.seniority)

        available = {p.id: p for p in pairings}
        allocations = {}

        for crew in crew_sorted:
            bids = crew.bid(available.values())
            for pairing in bids:
                if pairing.id in available:
                    allocations[crew.id] = pairing.id
                    del available[pairing.id]
                    break
        
        return allocations