import stripe

stripe.api_key = "sk_test_your_key"

class Payment:
    def __init__(self, amount, email):
        self.amount = amount
        self.email = email

    def create_checkout(self):
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            customer_email=self.email,
            line_items=[{
                "price_data": {
                    "currency": "usd",
                    "unit_amount": int(self.amount * 100),
                    "product_data": {
                        "name": "Premium Diet Plan",
                    },
                },
                "quantity": 1,
            }],
            mode="payment",
            success_url="http://localhost:8501?success=true",
            cancel_url="http://localhost:8501?canceled=true",
        )
        return session.url
