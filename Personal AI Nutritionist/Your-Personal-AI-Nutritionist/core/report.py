from db.database import SessionLocal
from db.models import User, Meal

class HealthReport:
    def __init__(self, username: str):
        self.db = SessionLocal()
        self.user = self.db.query(User).filter_by(username=username).first()

    def generate(self):
        meals = self.db.query(Meal).filter_by(user_id=self.user.id).all()
        total_calories = sum(m.calories for m in meals)
        total_protein = sum(m.protein for m in meals)
        total_carbs = sum(m.carbs for m in meals)
        total_fats = sum(m.fats for m in meals)

        return {
            "Calories": total_calories,
            "Protein (g)": total_protein,
            "Carbs (g)": total_carbs,
            "Fats (g)": total_fats,
            "Advice": "You're on track!" if total_calories < 2000 else "Consider reducing intake."
        }
