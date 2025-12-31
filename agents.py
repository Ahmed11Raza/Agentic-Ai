from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import AsyncGenerator, Literal, Optional

from openai import AsyncOpenAI
from config import OPENAI_API_KEY, OPENAI_MODEL
from database import get_loan_info, verify_employee, Feedback, Session
from sqlalchemy.exc import SQLAlchemyError

Mode = Literal["text", "summary", "detail"]
Personality = Literal["professional", "friendly", "funny"]

HELP_TEXT = (
    "### Commands\n"
    "- `/help` – Show this help\n"
    "- `/reset` – Clear conversation memory\n"
    "- `/about` – About this bot\n"
    "- `/mode [text|summary|detail]` – Change reply depth\n"
    "- `/persona [professional|friendly|funny]` – Switch tone/personality\n"
    "- `/short` or `/long` – Quick depth toggle\n"
    "- `/joke`, `/quote`, `/fact` – Fun extras\n"
    "- `/hr login <employee_id> <password>` – Log in for HR access\n"
    "- `/hr details` – Get loan details\n"
    "- `/hr balance` – Get loan balance\n"
    "- `/hr feedback <message>` – Submit HR feedback\n"
    "- `/hr logout` – Log out from HR session\n"
)

ABOUT_TEXT = (
    "🤖 **Atlas AI** — Atlas Honda’s assistant with integrated HR management.\n"
    "Features: streaming, short-term memory, commands, modes, personalities, quick KB lookup, HR loan inquiries, authentication, feedback, and graceful fallbacks.\n"
    "No database required for general use, but HR features use SQLite.\n"
)

FUN_JOKES = [
    "Why did the bike stay upright? It was two-tired of falling over! 😄",
    "I told my motorcycle a joke; it couldn’t handlebar it.",
]
FUN_QUOTES = [
    "“The only impossible journey is the one you never begin.” — Tony Robbins",
    "“Simplicity is the soul of efficiency.” — Austin Freeman",
]
FUN_FACTS = [
    "A well-maintained chain can noticeably improve fuel efficiency and ride smoothness.",
    "Tyre pressure affects handling, braking distance, and tyre life—check it regularly.",
]

logger = logging.getLogger(__name__)

def _read_kb() -> list[dict]:
    kb_path = Path("kb.json")
    if kb_path.exists():
        try:
            return json.loads(kb_path.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []

class Agent:
    def __init__(self, name: str, description: str, max_memory: int = 8):
        self.name = name
        self.description = description
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)

        self.memory: list[dict[str, str]] = []
        self.max_memory = max_memory

        self.mode: Mode = "text"
        self.persona: Personality = "professional"

        self._kb = _read_kb()
        self.last_reply_text: str = ""  # populated after each stream

        # HR integration (per-agent, now per-user)
        self.hr_session = Session()
        self.hr_logged_in = False
        self.hr_employee_id: Optional[str] = None
        self.hr_state: Optional[str] = None  # e.g., 'awaiting_login', 'awaiting_feedback'

    # ---------- Memory ----------
    def _add_to_memory(self, role: Literal["user", "assistant"], content: str) -> None:
        if not content:
            return
        self.memory.append({"role": role, "content": content})
        if len(self.memory) > self.max_memory:
            self.memory = self.memory[-self.max_memory :]

    # ---------- Persona/Mode ----------
    def _persona_instructions(self) -> str:
        if self.persona == "friendly":
            return "Tone: warm, friendly, and supportive."
        if self.persona == "funny":
            return "Tone: light, witty, and sparingly humorous while staying helpful."
        return "Tone: professional, concise, and accurate."

    def _mode_instructions(self) -> str:
        if self.mode == "summary":
            return "Respond concisely in 1–3 sentences."
        if self.mode == "detail":
            return "Respond thoroughly with structured, step-by-step clarity."
        return "Respond naturally with balanced detail."

    # ---------- KB quick lookup ----------
    def _kb_lookup(self, text: str) -> Optional[str]:
        if not self._kb:
            return None
        low = text.lower()
        # simple tag/keyword match (fast & local)
        best: Optional[str] = None
        score_best = 0
        for item in self._kb:
            tags = [t.lower() for t in item.get("tags", [])]
            score = sum(1 for t in tags if t in low)
            if score > score_best and score > 0:
                score_best = score
                best = item.get("a")
        return best

    # ---------- Natural Language HR Handling ----------
    def _handle_hr_natural_language(self, text: str) -> Optional[str]:
        low = text.lower().strip()
        # Check for login intent
        if any(word in low for word in ["login", "log in", "sign in", "authenticate"]):
            if self.hr_logged_in:
                return "You're already logged in."
            if "emp001" in low and "pass123" in low:
                return self._hr_login("emp001", "pass123")
            elif "emp002" in low and "pass456" in low:
                return self._hr_login("emp002", "pass456")
            else:
                self.hr_state = "awaiting_login"
                return "Sure, I can help you log in. Please provide your employee ID and password in the format: emp001 pass123"

        # Check for logout
        if any(word in low for word in ["logout", "log out", "sign out"]):
            if not self.hr_logged_in:
                return "You're not logged in."
            return self._hr_logout()

        # Check for details
        if any(word in low for word in ["details", "loan details", "loan info", "my loan"]):
            if not self.hr_logged_in:
                return "To view loan details, please log in first."
            return self._hr_get_loan_details()

        # Check for balance
        if any(word in low for word in ["balance", "loan balance", "remaining balance", "outstanding"]):
            if not self.hr_logged_in:
                return "To view loan balance, please log in first."
            return self._hr_get_loan_balance()

        # Check for feedback
        if any(word in low for word in ["feedback", "comment", "suggestion"]):
            if not self.hr_logged_in:
                return "To submit feedback, please log in first."
            message = text.strip()
            if len(message.split()) < 3:  # Simple check for actual message
                self.hr_state = "awaiting_feedback"
                return "What feedback would you like to submit?"
            return self._hr_submit_feedback(message)

        # Check for loan calculator
        if any(word in low for word in ["calculate loan", "loan calculator", "calculate", "emi"]):
            self.hr_state = "awaiting_loan_calc"
            return "Sure, I can help with a basic loan calculation. Please provide: principal amount, interest rate (%), and tenure (years) separated by spaces. E.g., 100000 10 5"

        # Handle awaiting states
        if self.hr_state == "awaiting_login":
            parts = text.split()
            if len(parts) == 2:
                emp_id, pwd = parts
                self.hr_state = None
                return self._hr_login(emp_id, pwd)
            else:
                return "Please provide employee ID and password separated by space."

        if self.hr_state == "awaiting_feedback":
            if text.strip():
                self.hr_state = None
                return self._hr_submit_feedback(text)
            else:
                return "Feedback cannot be empty."

        return None

    # ---------- HR Integration ----------
    def _hr_login(self, employee_id: str, password: str) -> str:
        employee_id = employee_id.strip()
        password = password.strip()
        if not employee_id or not password:
            return "❌ Employee ID and password cannot be empty."
        if verify_employee(employee_id, password):
            self.hr_logged_in = True
            self.hr_employee_id = employee_id
            return "✅ HR login successful for employee ID: " + employee_id
        return "❌ HR login failed. Invalid employee ID or password."

    def _hr_logout(self) -> str:
        self.hr_logged_in = False
        self.hr_employee_id = None
        return "✅ HR session logged out."

    def _hr_get_loan_details(self) -> str:
        if not self.hr_logged_in:
            return "❌ Please log in with /hr login <employee_id> <password> to access HR features."
        assert self.hr_employee_id is not None
        return get_loan_info(self.hr_employee_id, "details")

    def _hr_get_loan_balance(self) -> str:
        if not self.hr_logged_in:
            return "❌ Please log in with /hr login <employee_id> <password> to access HR features."
        assert self.hr_employee_id is not None
        return get_loan_info(self.hr_employee_id, "balance")

    def _hr_submit_feedback(self, message: str) -> str:
        message = message.strip()
        if not self.hr_logged_in:
            return "❌ Please log in with /hr login <employee_id> <password> to access HR features."
        if not message:
            return "❌ Feedback message cannot be empty."
        try:
            session = self.hr_session
            session.add(Feedback(employee_id=self.hr_employee_id, message=message))
            session.commit()
            return "✅ Feedback submitted successfully."
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Error submitting feedback: {e}")
            return "❌ Error submitting feedback. Please try again."

    def _calculate_loan_emi(self, principal: float, rate: float, tenure: float) -> str:
        # Simple EMI calculation: EMI = [P x R x (1+R)^N] / [(1+R)^N-1]
        # Where P = Principal, R = Monthly interest rate, N = Number of months
        monthly_rate = rate / 100 / 12
        months = tenure * 12
        if monthly_rate == 0:
            emi = principal / months
        else:
            emi = (principal * monthly_rate * (1 + monthly_rate) ** months) / ((1 + monthly_rate) ** months - 1)
        total_payment = emi * months
        total_interest = total_payment - principal
        return (
            f"📊 **Loan Calculation Summary**\n"
            f"Principal: ₹{principal:,.0f}\n"
            f"Interest Rate: {rate}%\n"
            f"Tenure: {tenure} years ({months:.0f} months)\n"
            f"Monthly EMI: ₹{emi:,.0f}\n"
            f"Total Interest: ₹{total_interest:,.0f}\n"
            f"Total Payment: ₹{total_payment:,.0f}\n\n"
            f"💡 This is a general calculation. For personalized loan details, please log in and check your HR account."
        )

    # ---------- Commands ----------
    def _handle_commands(self, text: str) -> Optional[str]:
        t = text.strip()
        low = t.lower()

        if low == "/help":
            return HELP_TEXT
        if low == "/reset":
            self.memory.clear()
            return "✅ Memory cleared."
        if low == "/about":
            return ABOUT_TEXT

        if low.startswith("/mode"):
            parts = t.split(maxsplit=1)
            if len(parts) == 1:
                return f"Current mode: **{self.mode}**.\nUse `/mode text`, `/mode summary`, or `/mode detail`."
            new_mode = parts[1].strip().lower()
            if new_mode in ("text", "summary", "detail"):
                self.mode = new_mode  # type: ignore[assignment]
                return f"✅ Mode changed to **{self.mode}**."
            return "❌ Invalid mode. Use: `/mode text`, `/mode summary`, or `/mode detail`."

        if low.startswith("/persona"):
            parts = t.split(maxsplit=1)
            if len(parts) == 1:
                return f"Current persona: **{self.persona}**.\nUse `/persona professional|friendly|funny`."
            new_p = parts[1].strip().lower()
            if new_p in ("professional", "friendly", "funny"):
                self.persona = new_p  # type: ignore[assignment]
                return f"✅ Persona changed to **{self.persona}**."
            return "❌ Invalid persona. Use: `/persona professional|friendly|funny`."

        if low in ("/short", "/summary"):
            self.mode = "summary"  # type: ignore[assignment]
            return "✅ Mode changed to **summary**."
        if low in ("/long", "/detail"):
            self.mode = "detail"  # type: ignore[assignment]
            return "✅ Mode changed to **detail**."

        if low == "/joke":
            return f"🎭 {random.choice(FUN_JOKES)}"
        if low == "/quote":
            return f"📜 {random.choice(FUN_QUOTES)}"
        if low == "/fact":
            return f"📘 {random.choice(FUN_FACTS)}"

        # HR Commands
        if low.startswith("/hr login"):
            parts = t.split(maxsplit=3)
            if len(parts) < 4:
                return "❌ Usage: /hr login <employee_id> <password>"
            employee_id = parts[2]
            password = parts[3]
            return self._hr_login(employee_id, password)

        if low == "/hr logout":
            return self._hr_logout()

        if low == "/hr details":
            return self._hr_get_loan_details()

        if low == "/hr balance":
            return self._hr_get_loan_balance()

        if low.startswith("/hr feedback"):
            parts = t.split(maxsplit=2)  # Capture full message
            if len(parts) < 3:
                return "❌ Usage: /hr feedback <message>"
            message = parts[2]
            return self._hr_submit_feedback(message)

        return None

    # ---------- Prompt build ----------
    def _build_messages(self) -> list[dict[str, str]]:
        system_text = (
            f"{self.description}\n"
            f"{self._persona_instructions()}\n"
            f"Mode: {self.mode}. {self._mode_instructions()}\n"
            f"Always reply in the same language as the user’s latest message.\n"
            f"If a question matches the knowledge base you know, answer directly and succinctly.\n"
            f"If unclear, ask a brief, specific clarifying question.\n"
            f"For HR queries, you can handle them in natural language (e.g., 'log me in', 'show my loan details').\n"
            f"If the user asks for help with loans or calculations, provide general advice but direct to HR for specifics."
        )
        return [{"role": "system", "content": system_text}, *self.memory]

    # ---------- Summarizer (optional TL;DR) ----------
    async def summarize(self, text: str) -> str:
        if not text.strip():
            return ""
        try:
            comp = await self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "Summarize the user's content to one concise line."},
                    {"role": "user", "content": text},
                ],
            )
            content = comp.choices[0].message.content
            return content.strip() if content else ""
        except Exception:
            return ""

    # ---------- Streaming main ----------
    async def stream_response(self, user_input: str) -> AsyncGenerator[str, None]:
        # Commands (quick, no API call)
        cmd = self._handle_commands(user_input)
        if cmd is not None:
            self.last_reply_text = cmd
            yield cmd
            return

        # Natural language HR handling
        hr_response = self._handle_hr_natural_language(user_input)
        if hr_response is not None:
            self.last_reply_text = hr_response
            yield hr_response
            return

        # KB instant answer (no API call)
        kb_hit = self._kb_lookup(user_input)
        if kb_hit:
            self._add_to_memory("user", user_input)
            self._add_to_memory("assistant", kb_hit)
            self.last_reply_text = kb_hit
            yield kb_hit
            return

        # Add user to memory and call model
        self._add_to_memory("user", user_input)
        self.last_reply_text = ""

        try:
            stream = await self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=self._build_messages(),
                stream=True,
            )

            chunks: list[str] = []
            async for chunk in stream:
                token: Optional[str] = None
                if chunk.choices:
                    # handle dict vs object SDK delta
                    delta_obj = chunk.choices[0].delta
                    token = getattr(delta_obj, "content", None)
                    if token is None and isinstance(delta_obj, dict):
                        token = delta_obj.get("content")
                if token:
                    chunks.append(token)
                    yield token

            final_text = "".join(chunks).strip() if chunks else ""
            if final_text:
                self._add_to_memory("assistant", final_text)
                self.last_reply_text = final_text
            else:
                fallback = "I couldn’t generate a response. Could you rephrase your question?"
                self._add_to_memory("assistant", fallback)
                self.last_reply_text = fallback
                yield fallback

        except Exception:
            fallback = "⚠️ I couldn’t process that right now. Please rephrase or try again."
            self.last_reply_text = fallback
            yield fallback

    def __del__(self):
        if hasattr(self, 'hr_session'):
            try:
                self.hr_session.close()
            except Exception as e:
                logger.error(f"Error closing HR session: {e}")