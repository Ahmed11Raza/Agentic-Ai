class DietPlan:
    def __init__(self, goal: str = "maintain"):
        self.goal = goal

    def generate_plan(self):
        plans = {
            "lose": [
                {"meal": "Oats & Banana", "calories": 300},
                {"meal": "Grilled Chicken & Veggies", "calories": 500},
                {"meal": "Fruit Salad", "calories": 200},
            ],
            "gain": [
                {"meal": "Eggs & Toast", "calories": 400},
                {"meal": "Beef Wrap", "calories": 600},
                {"meal": "Protein Shake", "calories": 300},
            ],
            "maintain": [
                {"meal": "Paratha & Omelette", "calories": 350},
                {"meal": "Rice & Daal", "calories": 500},
                {"meal": "Yogurt & Fruits", "calories": 250},
            ]
        }
        return plans.get(self.goal, [])
