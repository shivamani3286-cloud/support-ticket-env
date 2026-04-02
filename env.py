"""
Customer Support Ticket Triage - OpenEnv Environment
=====================================================
An AI agent reads customer support tickets and must classify, route,
and respond to them appropriately.
"""

import random
from typing import Any, Optional
from pydantic import BaseModel, Field
from enum import Enum


# ─── Pydantic Models ────────────────────────────────────────────────────────

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Department(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    GENERAL = "general"
    RETURNS = "returns"
    ACCOUNT = "account"


class Observation(BaseModel):
    ticket_id: str
    subject: str
    body: str
    customer_tier: str          # "free", "pro", "enterprise"
    previous_contacts: int
    task: str                   # "classify" | "route" | "respond"
    task_description: str


class Action(BaseModel):
    priority: Optional[str] = None        # for task 1
    department: Optional[str] = None      # for task 2
    response_text: Optional[str] = None   # for task 3


class Reward(BaseModel):
    value: float = Field(ge=0.0, le=1.0)
    reason: str


# ─── Ticket Dataset ──────────────────────────────────────────────────────────

TICKETS = [
    # Easy tickets (clear priority signals)
    {
        "subject": "Can't log in",
        "body": "I forgot my password. How do I reset it?",
        "customer_tier": "free",
        "previous_contacts": 0,
        "true_priority": Priority.LOW,
        "true_department": Department.ACCOUNT,
        "keywords": ["password", "reset", "login"],
    },
    {
        "subject": "Invoice question",
        "body": "Hi, I'd like to understand the charges on my last invoice. Can you break it down?",
        "customer_tier": "pro",
        "previous_contacts": 1,
        "true_priority": Priority.MEDIUM,
        "true_department": Department.BILLING,
        "keywords": ["invoice", "charges", "billing"],
    },
    {
        "subject": "URGENT: Production system is DOWN",
        "body": (
            "Our entire production environment is down! All users are affected. "
            "We are losing $10,000 per minute. This is a P0 emergency! "
            "We need someone immediately!"
        ),
        "customer_tier": "enterprise",
        "previous_contacts": 0,
        "true_priority": Priority.CRITICAL,
        "true_department": Department.TECHNICAL,
        "keywords": ["urgent", "down", "emergency", "production"],
    },
    {
        "subject": "Refund request",
        "body": "I accidentally purchased the wrong plan. I'd like a refund for this month.",
        "customer_tier": "pro",
        "previous_contacts": 2,
        "true_priority": Priority.MEDIUM,
        "true_department": Department.RETURNS,
        "keywords": ["refund", "purchase", "plan"],
    },
    {
        "subject": "API rate limits",
        "body": (
            "We keep hitting rate limits on the API during peak hours. "
            "This is causing failures in our automated pipeline."
        ),
        "customer_tier": "enterprise",
        "previous_contacts": 3,
        "true_priority": Priority.HIGH,
        "true_department": Department.TECHNICAL,
        "keywords": ["api", "rate limit", "failure"],
    },
    {
        "subject": "How do I export my data?",
        "body": "I want to export all my account data. Is there a way to do this?",
        "customer_tier": "free",
        "previous_contacts": 0,
        "true_priority": Priority.LOW,
        "true_department": Department.GENERAL,
        "keywords": ["export", "data"],
    },
    {
        "subject": "Billing discrepancy",
        "body": (
            "I was charged twice this month for the same subscription. "
            "Please fix this immediately and issue a refund for the duplicate charge."
        ),
        "customer_tier": "pro",
        "previous_contacts": 1,
        "true_priority": Priority.HIGH,
        "true_department": Department.BILLING,
        "keywords": ["charged twice", "duplicate", "refund"],
    },
    {
        "subject": "Feature not working",
        "body": "The dark mode toggle doesn't seem to save my preference. Every time I reload it resets.",
        "customer_tier": "free",
        "previous_contacts": 0,
        "true_priority": Priority.LOW,
        "true_department": Department.TECHNICAL,
        "keywords": ["dark mode", "not working", "resets"],
    },
]

GOOD_RESPONSE_KEYWORDS = {
    "greeting": ["hello", "hi", "dear", "greetings", "thank you for"],
    "empathy": ["understand", "sorry", "apologize", "appreciate"],
    "action": ["will", "team", "help", "assist", "resolve", "investigate", "look into"],
    "closing": ["please", "let us know", "contact", "feel free", "reach out"],
}


# ─── Environment ─────────────────────────────────────────────────────────────

class SupportTicketEnv:
    """
    OpenEnv-compatible Customer Support Ticket Triage environment.

    Tasks:
      Task 1 (easy)   - Classify ticket priority: low / medium / high / critical
      Task 2 (medium) - Route ticket to correct department
      Task 3 (hard)   - Draft an appropriate customer response
    """

    TASKS = [
        {
            "id": "classify_priority",
            "name": "Classify Ticket Priority",
            "difficulty": "easy",
            "description": (
                "Read the support ticket and classify its priority level. "
                "Valid values: low, medium, high, critical."
            ),
        },
        {
            "id": "route_department",
            "name": "Route to Department",
            "difficulty": "medium",
            "description": (
                "Read the support ticket and route it to the correct department. "
                "Valid values: billing, technical, general, returns, account."
            ),
        },
        {
            "id": "draft_response",
            "name": "Draft Customer Response",
            "difficulty": "hard",
            "description": (
                "Read the support ticket and write a professional, empathetic, "
                "helpful response to the customer. Must include: greeting, "
                "acknowledgement, action plan, and closing."
            ),
        },
    ]

    def __init__(self, task_id: str = "classify_priority", seed: Optional[int] = None):
        assert task_id in [t["id"] for t in self.TASKS], f"Unknown task: {task_id}"
        self.task_id = task_id
        self.task_info = next(t for t in self.TASKS if t["id"] == task_id)
        self._rng = random.Random(seed)
        self._current_ticket = None
        self._done = False
        self._step_count = 0

    # ── OpenEnv Interface ────────────────────────────────────────────────────

    def reset(self) -> Observation:
        """Reset environment and return initial observation."""
        self._current_ticket = self._rng.choice(TICKETS)
        self._done = False
        self._step_count = 0
        return self._make_observation()

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        """
        Execute action and return (observation, reward, done, info).
        Each episode is a single step (one ticket, one action).
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")

        reward = self._grade(action)
        self._done = True
        self._step_count += 1

        obs = self._make_observation()
        info = {
            "task_id": self.task_id,
            "ticket_id": obs.ticket_id,
            "step": self._step_count,
        }
        return obs, reward, self._done, info

    def state(self) -> dict:
        """Return current environment state."""
        return {
            "task_id": self.task_id,
            "done": self._done,
            "step_count": self._step_count,
            "current_ticket": self._current_ticket,
        }

    # ── Graders ──────────────────────────────────────────────────────────────

    def _grade(self, action: Action) -> Reward:
        if self.task_id == "classify_priority":
            return self._grade_classify(action)
        elif self.task_id == "route_department":
            return self._grade_route(action)
        elif self.task_id == "draft_response":
            return self._grade_response(action)

    def _grade_classify(self, action: Action) -> Reward:
        """Grade: priority classification (easy)."""
        if not action.priority:
            return Reward(value=0.0, reason="No priority provided.")

        predicted = action.priority.strip().lower()
        true_val = self._current_ticket["true_priority"].value

        priority_order = ["low", "medium", "high", "critical"]

        if predicted not in priority_order:
            return Reward(value=0.0, reason=f"Invalid priority value: '{predicted}'.")

        if predicted == true_val:
            return Reward(value=1.0, reason="Correct priority classification!")

        # Partial credit for adjacent priority levels
        diff = abs(priority_order.index(predicted) - priority_order.index(true_val))
        if diff == 1:
            return Reward(value=0.5, reason=f"Off by one level. Expected '{true_val}', got '{predicted}'.")
        return Reward(value=0.0, reason=f"Wrong priority. Expected '{true_val}', got '{predicted}'.")

    def _grade_route(self, action: Action) -> Reward:
        """Grade: department routing (medium)."""
        if not action.department:
            return Reward(value=0.0, reason="No department provided.")

        predicted = action.department.strip().lower()
        true_val = self._current_ticket["true_department"].value

        valid_depts = [d.value for d in Department]
        if predicted not in valid_depts:
            return Reward(value=0.0, reason=f"Invalid department: '{predicted}'.")

        if predicted == true_val:
            return Reward(value=1.0, reason="Correct department routing!")

        # Partial credit: billing & returns are related
        related = {("billing", "returns"), ("returns", "billing")}
        if (predicted, true_val) in related:
            return Reward(value=0.4, reason=f"Close but wrong. Expected '{true_val}', got '{predicted}'.")

        return Reward(value=0.0, reason=f"Wrong department. Expected '{true_val}', got '{predicted}'.")

    def _grade_response(self, action: Action) -> Reward:
        """Grade: response drafting (hard) — rubric-based partial credit."""
        if not action.response_text:
            return Reward(value=0.0, reason="No response text provided.")

        text = action.response_text.lower()
        score = 0.0
        reasons = []

        # 1. Greeting (0.2)
        if any(kw in text for kw in GOOD_RESPONSE_KEYWORDS["greeting"]):
            score += 0.2
            reasons.append("Has greeting (+0.2)")
        else:
            reasons.append("Missing greeting (-0.2)")

        # 2. Empathy (0.2)
        if any(kw in text for kw in GOOD_RESPONSE_KEYWORDS["empathy"]):
            score += 0.2
            reasons.append("Shows empathy (+0.2)")
        else:
            reasons.append("Missing empathy (-0.2)")

        # 3. Clear action / next steps (0.3)
        if any(kw in text for kw in GOOD_RESPONSE_KEYWORDS["action"]):
            score += 0.3
            reasons.append("Includes action/next steps (+0.3)")
        else:
            reasons.append("Missing action/next steps (-0.3)")

        # 4. Professional closing (0.15)
        if any(kw in text for kw in GOOD_RESPONSE_KEYWORDS["closing"]):
            score += 0.15
            reasons.append("Has professional closing (+0.15)")
        else:
            reasons.append("Missing professional closing (-0.15)")

        # 5. Mentions ticket subject / is relevant (0.15)
        ticket_keywords = self._current_ticket.get("keywords", [])
        if any(kw in text for kw in ticket_keywords):
            score += 0.15
            reasons.append("Response is relevant to ticket (+0.15)")
        else:
            reasons.append("Response seems generic / irrelevant (-0.15)")

        score = round(min(max(score, 0.0), 1.0), 2)
        return Reward(value=score, reason=" | ".join(reasons))

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _make_observation(self) -> Observation:
        t = self._current_ticket
        ticket_id = f"TKT-{self._rng.randint(1000, 9999)}"
        return Observation(
            ticket_id=ticket_id,
            subject=t["subject"],
            body=t["body"],
            customer_tier=t["customer_tier"],
            previous_contacts=t["previous_contacts"],
            task=self.task_id,
            task_description=self.task_info["description"],
        )


# ─── Task Registry (for openenv validate) ────────────────────────────────────

def get_tasks():
    return SupportTicketEnv.TASKS


def make_env(task_id: str, seed: Optional[int] = None) -> SupportTicketEnv:
    return SupportTicketEnv(task_id=task_id, seed=seed)


# ─── Quick smoke test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    for task in ["classify_priority", "route_department", "draft_response"]:
        print(f"\n{'='*60}")
        print(f"Task: {task}")
        env = make_env(task, seed=42)
        obs = env.reset()
        print(f"Ticket: [{obs.subject}] — {obs.body[:80]}...")
        print(f"Customer tier: {obs.customer_tier} | Prior contacts: {obs.previous_contacts}")

        if task == "classify_priority":
            action = Action(priority="high")
        elif task == "route_department":
            action = Action(department="technical")
        else:
            action = Action(response_text=(
                "Hello! Thank you for reaching out to our support team. "
                "We understand your concern and we sincerely apologize for the inconvenience. "
                "Our team will investigate this issue and resolve it as quickly as possible. "
                "Please let us know if you have any additional questions."
            ))

        obs2, reward, done, info = env.step(action)
        print(f"Reward: {reward.value:.2f} — {reward.reason}")
        print(f"Done: {done}")
