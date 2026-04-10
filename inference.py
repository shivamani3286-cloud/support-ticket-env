import os
import json
import time
from typing import List, Optional
from openai import OpenAI
from env import make_env, Action

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "dummy-key")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

SEEDS = [0, 1, 2, 3, 4]
BENCHMARK = "support-ticket-env"

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error=None):
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

def call_llm(prompt):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[DEBUG] LLM call failed: {e}", flush=True)
        return "{}"

def build_prompt_classify(obs):
    return f"""Classify ticket priority as exactly one of: low, medium, high, critical
Ticket: {obs.subject} - {obs.body}
Customer Tier: {obs.customer_tier}
Respond with ONLY: {{"priority": "<answer>"}}"""

def build_prompt_route(obs):
    return f"""Route ticket to exactly one of: billing, technical, general, returns, account
Ticket: {obs.subject} - {obs.body}
Respond with ONLY: {{"department": "<answer>"}}"""

def build_prompt_respond(obs):
    return f"""Write a helpful response to this support ticket.
Ticket: {obs.subject} - {obs.body}
Respond with ONLY: {{"response_text": "<your response>"}}"""

PROMPT_BUILDERS = {
    "classify_priority": build_prompt_classify,
    "route_department":  build_prompt_route,
    "draft_response":    build_prompt_respond,
}

def parse_action(task_id, raw):
    clean = raw.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(clean)
    except:
        data = {}
    if task_id == "classify_priority":
        return Action(priority=data.get("priority", "low").strip().lower())
    elif task_id == "route_department":
        return Action(department=data.get("department", "general").strip().lower())
    elif task_id == "draft_response":
        return Action(response_text=data.get("response_text", "Thank you for contacting us."))
    return Action()

def run_task(task_id):
    all_rewards = []
    total_steps = 0

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    for seed in SEEDS:
        try:
            env = make_env(task_id, seed=seed)
            obs = env.reset()
            prompt = PROMPT_BUILDERS[task_id](obs)
            raw = call_llm(prompt)
            action = parse_action(task_id, raw)
            _, reward, done, info = env.step(action)
            r = reward.value
            all_rewards.append(r)
            total_steps += 1
            log_step(step=total_steps, action=str(action), reward=r, done=done)
        except Exception as e:
            all_rewards.append(0.0)
            total_steps += 1
            log_step(step=total_steps, action="error", reward=0.0, done=True, error=str(e))

    score = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
    success = score > 0.0
    log_end(success=success, steps=total_steps, score=score, rewards=all_rewards)
    return score

def main():
    for task_id in ["classify_priority", "route_department", "draft_response"]:
        run_task(task_id)

if __name__ == "__main__":
    main()
