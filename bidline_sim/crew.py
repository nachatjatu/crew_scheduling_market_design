class CrewMember:
    def __init__(self, id, base, seniority, preferences):
        self.id = id
        # self.base = base
        self.seniority = seniority
        self.preferences = preferences

    def bid(self, available_pairings):
        return sorted(
            available_pairings,
            key=lambda p: self.preferences.get(p.id, 0),
            reverse=True
        )