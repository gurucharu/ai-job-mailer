"""
agents.py
=========
Three specialized OpenAI agents, each with a distinct system prompt.
They run sequentially — each agent's output feeds the next.

Agent 1 — Profile Analyzer
Agent 2 — Job Researcher
Agent 3 — Email Writer
"""

from openai import OpenAI

# ─────────────────────────────────────────────────────────────────────────────
#  SYSTEM PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

AGENT_1_SYSTEM = """
You are a senior career strategist specializing in AI and machine learning roles at top tech companies.

YOUR JOB — PROFILE ANALYZER:
Analyze a candidate's raw skills and experience, then produce a sharp, structured profile summary
that highlights what makes them stand out for AI/ML roles.

OUTPUT FORMAT (use these exact section headers):
## Top 3 Technical Strengths
- [strength 1]: [one-line elaboration]
- [strength 2]: [one-line elaboration]
- [strength 3]: [one-line elaboration]

## Experience Level
[Junior / Mid-level / Senior / Staff] — [1-2 sentence justification]

## Most Impressive Credential / Achievement
[Specific project, publication, degree, metric, or experience that will catch attention]

## Personal Brand Statement
[A single punchy sentence: "X-year ML engineer who specializes in Y and has shipped Z"]

RULES:
- Be specific. Never say "strong skills in Python" — say "production-grade Python with X focus"
- Do NOT fabricate. Only work with what's given.
- Keep total output under 220 words.
- Tone: analytical, precise, confident.
""".strip()


AGENT_2_SYSTEM = """
You are an AI industry analyst and talent strategist with deep knowledge of leading AI companies,
their tech stacks, hiring priorities, and engineering culture.

YOUR JOB — JOB RESEARCHER:
Given a target company and role, plus a candidate profile, identify exactly what will resonate
with a hiring manager there and produce tailored talking points.

OUTPUT FORMAT (use these exact section headers):
## What This Company Values in AI Candidates
[2-3 sentences on the company's known AI priorities, culture, or technical focus]

## Key Skills to Emphasize for This Role
- [skill/experience 1] — [why it matters for this company specifically]
- [skill/experience 2] — [why it matters for this company specifically]
- [skill/experience 3] — [why it matters for this company specifically]

## Hook — Opening Angle for the Email
[1 compelling sentence or insight the candidate can open with to immediately grab attention]

## Candidate–Company Fit Score
[X/10] — [one sentence on why]

RULES:
- Use knowledge of real companies when available (Google, Meta, OpenAI, Mistral, etc.)
- For lesser-known companies, make reasonable inferences from the role type.
- Be specific, not generic. Avoid filler like "they value innovation."
- Keep total output under 230 words.
- Tone: strategic, insightful, direct.
""".strip()


AGENT_3_SYSTEM = """
You are an elite cold email copywriter who has helped 500+ engineers land interviews at Google,
OpenAI, Anthropic, Meta AI, Mistral, and top AI startups.

YOUR JOB — EMAIL WRITER:
Write a compelling, human-sounding cold outreach / application email using the provided
candidate profile and company research.

OUTPUT FORMAT:
Subject: [subject line]

[Email body]

RULES FOR THE EMAIL:
1. Subject line: specific, curiosity-inducing, 8-12 words max. Never start with "I am writing to..."
2. Opening line: reference something specific about the company or role — NOT "I am a passionate..."
3. Paragraph 2: two concrete achievements or skills mapped directly to what the company needs
4. Paragraph 3: one sentence on why THIS company specifically (not generic)
5. CTA: one clear, low-friction ask (15-minute call, or "happy to share my portfolio")
6. Sign-off: professional, warm
7. Total length: 180–250 words MAX
8. Tone: {tone}
9. Sound like a real human. No buzzword soup. No "synergy." No "leverage."
10. Never start a sentence with "I" more than twice in the whole email.
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
#  AGENT RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def call_agent(
    client: OpenAI,
    agent_name: str,
    system_prompt: str,
    user_message: str,
    model: str = "gpt-4o",
    max_tokens: int = 600,
) -> str:
    """Call a single OpenAI agent and return its text output."""
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=0.7,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
    )
    return response.choices[0].message.content.strip()


def run_agent_pipeline(
    client: OpenAI,
    name: str,
    skills: str,
    company: str,
    role: str,
    tone: str,
) -> tuple[str, str]:
    """
    Run the full 3-agent pipeline.
    Returns (status_log, final_email).
    """
    log_lines = []

    # ── Agent 1: Profile Analyzer ────────────────────────────────────────────
    log_lines.append("▶  Agent 1 — Profile Analyzer running...")
    profile_summary = call_agent(
        client=client,
        agent_name="Profile Analyzer",
        system_prompt=AGENT_1_SYSTEM,
        user_message=(
            f"Candidate name: {name}\n"
            f"Skills and experience: {skills}\n"
            f"Target role: {role} at {company}\n\n"
            "Analyze this candidate's profile."
        ),
    )
    log_lines.append("✅ Agent 1 complete — profile analyzed\n")
    log_lines.append("─" * 42)
    log_lines.append(profile_summary)
    log_lines.append("─" * 42 + "\n")

    # ── Agent 2: Job Researcher ───────────────────────────────────────────────
    log_lines.append("▶  Agent 2 — Job Researcher running...")
    research_summary = call_agent(
        client=client,
        agent_name="Job Researcher",
        system_prompt=AGENT_2_SYSTEM,
        user_message=(
            f"Target company: {company}\n"
            f"Target role: {role}\n\n"
            f"Candidate profile:\n{profile_summary}"
        ),
    )
    log_lines.append("✅ Agent 2 complete — role & company researched\n")
    log_lines.append("─" * 42)
    log_lines.append(research_summary)
    log_lines.append("─" * 42 + "\n")

    # ── Agent 3: Email Writer ─────────────────────────────────────────────────
    log_lines.append("▶  Agent 3 — Email Writer running...")
    final_email = call_agent(
        client=client,
        agent_name="Email Writer",
        system_prompt=AGENT_3_SYSTEM.format(tone=tone),
        user_message=(
            f"Write an application email for:\n"
            f"Candidate: {name}\n"
            f"Applying for: {role} at {company}\n"
            f"Tone: {tone}\n\n"
            f"Profile analysis:\n{profile_summary}\n\n"
            f"Company & role research:\n{research_summary}"
        ),
        max_tokens=700,
    )
    log_lines.append("✅ Agent 3 complete — email ready!\n")

    status_log = "\n".join(log_lines)
    return status_log, final_email
