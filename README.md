# 🎫 Support Ticket Triage — OpenEnv Environment

> **Meta x Scaler OpenEnv Hackathon — Round 1 Submission**
> Author: Krishnakanthula Shivamani

---

## 📋 Environment Description

**Support Ticket Triage** is a real-world OpenEnv environment where an AI agent acts as a customer support specialist. Given customer support tickets, the agent must:

1. **Classify** the ticket's urgency/priority
2. **Route** it to the correct department
3. **Draft** a professional, empathetic response

This simulates work performed millions of times daily by support teams at software companies — making it a genuine, high-value real-world task (not a game or toy).

---

## 🎯 Tasks

| # | Task ID | Difficulty | Description |
|---|---------|------------|-------------|
| 1 | `classify_priority` | Easy | Classify ticket priority: `low` / `medium` / `high` / `critical` |
| 2 | `route_department` | Medium | Route to correct dept: `billing` / `technical` / `general` / `returns` / `account` |
| 3 | `draft_response` | Hard | Write a professional, empathetic response to the customer |

---

## 📐 Observation Space

```python
class Observation(BaseModel):
    ticket_id: str           # Unique ticket identifier
    subject: str             # Email subject line
    body: str                # Full ticket body text
    customer_tier: str       # "free" | "pro" | "enterprise"
    previous_contacts: int   # How many times customer has contacted before
    task: str                # Which task the agent should perform
    task_description: str    # Natural language description of the task
```

---

## ⚡ Action Space

```python
class Action(BaseModel):
    priority: Optional[str] = None        # Task 1: "low"|"medium"|"high"|"critical"
    department: Optional[str] = None      # Task 2: "billing"|"technical"|"general"|"returns"|"account"
    response_text: Optional[str] = None   # Task 3: free-form response text
```

---

## 🏆 Reward / Grading Logic

### Task 1 — Classify Priority (Easy)
| Outcome | Score |
|---------|-------|
| Exact match | 1.0 |
| Off by one level | 0.5 |
| Wrong / invalid | 0.0 |

### Task 2 — Route Department (Medium)
| Outcome | Score |
|---------|-------|
| Correct department | 1.0 |
| Closely related (billing ↔ returns) | 0.4 |
| Wrong / invalid | 0.0 |

### Task 3 — Draft Response (Hard)
Rubric-based partial credit across 5 criteria:

| Criterion | Weight |
|-----------|--------|
| Greeting present | 0.20 |
| Empathy shown | 0.20 |
| Action / next steps | 0.30 |
| Professional closing | 0.15 |
| Ticket-relevant content | 0.15 |

---

## 🌐 API Endpoints (HTTP)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check — returns 200 |
| `POST` | `/reset` | Reset environment, get initial observation |
| `POST` | `/step` | Execute action, get reward |
| `GET` | `/state` | Get current environment state |
| `GET` | `/tasks` | List all tasks |

### Example: Reset
```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "classify_priority", "seed": 42}'
```

### Example: Step
```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"task_id": "classify_priority", "priority": "high"}'
```

---

## 🐍 Python Usage

```python
from env import make_env, Action

# Task 1: Classify Priority
env = make_env("classify_priority", seed=42)
obs = env.reset()
print(obs.subject, obs.body)

action = Action(priority="high")
obs, reward, done, info = env.step(action)
print(reward.value, reward.reason)

# Task 3: Draft Response
env = make_env("draft_response", seed=0)
obs = env.reset()
action = Action(response_text="Hello! Thank you for contacting us...")
obs, reward, done, info = env.step(action)
```

---

## 🚀 Setup & Installation

### Local

```bash
git clone <your-repo-url>
cd support-ticket-env
pip install -r requirements.txt

# Run smoke test
python env.py

# Start server
python server.py
```

### Docker

```bash
docker build -t support-ticket-env .

docker run -p 7860:7860 \
  -e API_BASE_URL=https://api.openai.com/v1 \
  -e MODEL_NAME=gpt-4o-mini \
  -e HF_TOKEN=your_key_here \
  support-ticket-env
```

---

## 📊 Baseline Scores

Run the baseline inference script:

```bash
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=your_key_here

python inference.py
```

Expected baseline scores (GPT-4o-mini, 5 seeds each):

| Task | Avg Score |
|------|-----------|
| classify_priority | ~0.80 |
| route_department | ~0.75 |
| draft_response | ~0.72 |
| **Overall** | **~0.76** |

---

## 🔧 Environment Variables

| Variable | Description |
|----------|-------------|
| `API_BASE_URL` | LLM API endpoint (e.g. `https://api.openai.com/v1`) |
| `MODEL_NAME` | Model identifier (e.g. `gpt-4o-mini`) |
| `HF_TOKEN` | Your Hugging Face / API key |

---

## 📁 File Structure

```
support-ticket-env/
├── env.py              # Core OpenEnv environment
├── server.py           # FastAPI HTTP server (HF Spaces)
├── inference.py        # Baseline inference script
├── openenv.yaml        # Environment metadata
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container definition
└── README.md           # This file
```

---

## 💡 Design Decisions

- **Real-world relevance**: Customer support triage is a high-value, high-volume task performed by humans at every tech company.
- **Deterministic grading**: Priority and routing tasks have clear correct answers; response grading uses a fixed rubric with no LLM judge.
- **Partial credit**: All graders provide meaningful signal across the full trajectory, not just binary pass/fail.
- **Scalable dataset**: Ticket dataset can be extended with more examples without changing the grading logic.
-
