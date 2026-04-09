"""
inference.py — Baseline inference script for Support Ticket Triage OpenEnv
===========================================================================
Uses the OpenAI-compatible client to run an LLM agent against all 3 tasks.

Environment variables required:
  API_BASE_URL  — The LLM API endpoint (e.g. https://api.openai.com/v1)
  MODEL_NAME    — Model identifier (e.g. gpt-4o-mini)
  HF_TOKEN      — Your Hugging Face / API key (used as the API key)

Run:
  python inference.py
"""

import os
import json
import time
from openai import OpenAI
from env import make_env, Action

# ─── Config ──────────────────────────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3")
HF_TOKEN     = os.environ.get("HF_TOKEN", "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")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

SEEDS    = [0, 1, 2, 3, 4]   # 5 episodes per task → average score
MAX_TIME = 18 * 60            # 18 min safety limit (< 20 min requirement)

# ─── Prompt Builders ─────────────────────────────────────────────────────────

def build_prompt_classify(obs) -> str:
    return f"""You are a customer support specialist. Read the following ticket and classify its priority.

Ticket ID: {obs.ticket_id}
Customer Tier: {obs.customer_tier}
Previous Contacts: {obs.previous_contacts}
Subject: {obs.subject}
Body: {obs.body}

Classify the priority as EXACTLY one of: low, medium, high, critical
Consider: urgency of language, customer tier, business impact, and previous contacts.

Respond with ONLY a JSON object:
{{"priority": "<your answer>"}}"""


def build_prompt_route(obs) -> str:
    return f"""You are a customer support specialist. Read the following ticket and route it to the correct department.

Ticket ID: {obs.ticket_id}
Customer Tier: {obs.customer_tier}
Previous Contacts: {obs.previous_contacts}
Subject: {obs.subject}
Body: {obs.body}

Route to EXACTLY one of: billing, technical, general, returns, account
- billing: payment, invoice, subscription charges
- technical: bugs, errors, API issues, outages
- general: general questions, how-to, features
- returns: refunds, cancellations, wrong purchases
- account: login, password, profile settings

Respond with ONLY a JSON object:
{{"department": "<your answer>"}}"""


def build_prompt_respond(obs) -> str:
    return f"""You are a professional customer support agent. Write a helpful, empathetic response to the following ticket.

Ticket ID: {obs.ticket_id}
Customer Tier: {obs.customer_tier}
Previous Contacts: {obs.previous_contacts}
Subject: {obs.subject}
Body: {obs.body}

Your response MUST include:
1. A warm greeting
2. Acknowledgement / empathy for their situation
3. Clear action or next steps you will take
4. A professional closing inviting them to contact you again

Respond with ONLY a JSON object:
{{"response_text": "<your full response>"}}"""


PROMPT_BUILDERS = {
    "classify_priority": build_prompt_classify,
    "route_department":  build_prompt_route,
    "draft_response":    build_prompt_respond,
}

# ─── LLM Call ────────────────────────────────────────────────────────────────

def call_llm(prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"LLM call failed: {e}")
        if "priority" in prompt.lower():
            return '{"priority": "medium"}'
        elif "department" in prompt.lower():
            return '{"department": "general"}'
        else:
            return '{"response_text": "Thank you for contacting us. We will look into your issue shortly."}'


def parse_action(task_id: str, raw: str) -> Action:
    """Parse LLM JSON output into an Action object."""
    # Strip markdown fences if present
    clean = raw.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(clean)
    except json.JSONDecodeError:
        # Fallback: try to extract value heuristically
        data = {}

    if task_id == "classify_priority":
        return Action(priority=data.get("priority", "").strip().lower())
    elif task_id == "route_department":
        return Action(department=data.get("department", "").strip().lower())
    elif task_id == "draft_response":
        return Action(response_text=data.get("response_text", ""))
    return Action()


# ─── Run Evaluation ──────────────────────────────────────────────────────────

def evaluate_task(task_id: str) -> dict:
    print(f"\n{'─'*60}")
    print(f"  Task: {task_id}")
    print(f"{'─'*60}")

    scores = []
    prompt_builder = PROMPT_BUILDERS[task_id]

    for seed in SEEDS:
        env = make_env(task_id, seed=seed)
        obs = env.reset()

        prompt = prompt_builder(obs)
        raw = call_llm(prompt)

        action = parse_action(task_id, raw)
        _, reward, _, info = env.step(action)

        scores.append(reward.value)
        print(f"  Seed {seed}: score={reward.value:.2f} | {reward.reason[:60]}")

    avg = round(sum(scores) / len(scores), 4)
    print(f"  ► Average: {avg:.4f}")
    return {"task_id": task_id, "scores": scores, "average": avg}


def main():
    start = time.time()
    print("=" * 60)
    print("  Support Ticket Triage — Baseline Inference")
    print(f"  Model: {MODEL_NAME}")
    print(f"  API:   {API_BASE_URL}")
    print("=" * 60)

    results = []
    for task_id in ["classify_priority", "route_department", "draft_response"]:
        if time.time() - start > MAX_TIME:
            print("⚠ Time limit approaching — stopping early.")
            break
        result = evaluate_task(task_id)
        results.append(result)

    overall = round(sum(r["average"] for r in results) / len(results), 4)

    print(f"\n{'='*60}")
    print("  FINAL BASELINE SCORES")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['task_id']:<25} {r['average']:.4f}")
    print(f"  {'OVERALL AVERAGE':<25} {overall:.4f}")
    print(f"  Elapsed: {time.time()-start:.1f}s")
    print("=" * 60)

    # Write scores to file for CI/CD checks
    with open("baseline_scores.json", "w") as f:
        json.dump({"results": results, "overall": overall}, f, indent=2)
    print("\n  Scores saved to baseline_scores.json")


if __name__ == "__main__":
    main()
