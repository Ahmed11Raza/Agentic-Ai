from db.database import SessionLocal
from db.models import Meal, User

class MealLogger:
    def __init__(self, username: str):
        self.db = SessionLocal()
        self.user = self.db.query(User).filter_by(username=username).first()

    def log_meal(self, name: str, calories: float, protein: float, carbs: float, fats: float):
        if not self.user:
            raise Exception("User not found")
        meal = Meal(name=name, calories=calories, protein=protein, carbs=carbs, fats=fats, user_id=self.user.id)
        self.db.add(meal)
        self.db.commit()

    def get_all_meals(self):
        return self.db.query(Meal).filter_by(user_id=self.user.id).all()
