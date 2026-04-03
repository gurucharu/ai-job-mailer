"""
AI Job Email Generator
======================
3-Agent pipeline using OpenAI API:
  Agent 1 — Profile Analyzer
  Agent 2 — Job Researcher
  Agent 3 — Email Writer

Email delivery via SendGrid.
Built for Hugging Face Spaces with Gradio.
"""

import os
import gradio as gr
from openai import OpenAI
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from agents import run_agent_pipeline

# ── env vars (set in HF Spaces Secrets) ─────────────────────────────────────
OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY", "")
SENDGRID_API_KEY = os.environ.get("SENDGRID_API_KEY", "")
SENDER_EMAIL     = os.environ.get("SENDER_EMAIL", "")   # your verified sender

# ── CSS theme ────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

* { box-sizing: border-box; }

body, .gradio-container {
    font-family: 'DM Sans', sans-serif !important;
    background: #0a0a0f !important;
    color: #e8e8f0 !important;
}

.gradio-container {
    max-width: 860px !important;
    margin: 0 auto !important;
}

/* Header */
.app-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    border-bottom: 1px solid #1e1e2e;
    margin-bottom: 1.5rem;
}
.app-header h1 {
    font-size: 2rem;
    font-weight: 600;
    letter-spacing: -0.5px;
    background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 6px;
}
.app-header p {
    color: #6b7280;
    font-size: 0.95rem;
    margin: 0;
}

/* Agent badges */
.agent-pills {
    display: flex;
    gap: 10px;
    justify-content: center;
    flex-wrap: wrap;
    margin-bottom: 1.5rem;
}
.pill {
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 500;
    border: 1px solid;
}
.pill-1 { background: #1e1033; color: #a78bfa; border-color: #4c2d80; }
.pill-2 { background: #0d2218; color: #34d399; border-color: #1a5e3f; }
.pill-3 { background: #0d1a2e; color: #60a5fa; border-color: #1a3a6e; }

/* Cards / panels */
.panel {
    background: #12121c !important;
    border: 1px solid #1e1e2e !important;
    border-radius: 12px !important;
}

/* Inputs */
input[type=text], textarea, .gr-input {
    background: #0f0f1a !important;
    border: 1px solid #2a2a3e !important;
    border-radius: 8px !important;
    color: #e8e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
}
input[type=text]:focus, textarea:focus {
    border-color: #a78bfa !important;
    box-shadow: 0 0 0 3px rgba(167,139,250,0.12) !important;
}

/* Buttons */
.gr-button-primary {
    background: linear-gradient(135deg, #7c3aed, #2563eb) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    letter-spacing: 0.2px !important;
    transition: opacity 0.2s !important;
}
.gr-button-primary:hover { opacity: 0.88 !important; }
.gr-button-secondary {
    background: #1e1e2e !important;
    border: 1px solid #2a2a3e !important;
    border-radius: 8px !important;
    color: #e8e8f0 !important;
}

/* Status box */
.status-box {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
    background: #080810 !important;
    border: 1px solid #1e1e2e !important;
    border-radius: 8px !important;
    color: #6b7280 !important;
}

/* Output email */
.email-output {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.83rem !important;
    line-height: 1.75 !important;
    background: #080810 !important;
    border: 1px solid #1e1e2e !important;
    color: #d1d5db !important;
    border-radius: 8px !important;
}

/* Labels */
label { color: #9ca3af !important; font-size: 0.82rem !important; font-weight: 500 !important; }

/* Footer */
.footer {
    text-align: center;
    padding: 1.5rem 0 1rem;
    color: #374151;
    font-size: 0.78rem;
}
"""

HEADER_HTML = """
<div class="app-header">
  <h1>✦ AI Job Email Generator</h1>
  <p>Three specialized AI agents craft your perfect job application email</p>
</div>
<div class="agent-pills">
  <span class="pill pill-1">◆ Agent 1 — Profile Analyzer</span>
  <span class="pill pill-2">◆ Agent 2 — Job Researcher</span>
  <span class="pill pill-3">◆ Agent 3 — Email Writer</span>
</div>
"""

FOOTER_HTML = """
<div class="footer">
  Built with OpenAI · SendGrid · Gradio &nbsp;|&nbsp; Deploy on Hugging Face Spaces
</div>
"""


# ── Core generate function ────────────────────────────────────────────────────
def generate_email(
    your_name,
    your_skills,
    target_company,
    target_role,
    recipient_email,
    tone,
    send_email,
    openai_key_input,
    sendgrid_key_input,
    sender_email_input,
):
    # Allow keys from UI if not set in env (useful for HF Spaces demo mode)
    oai_key  = OPENAI_API_KEY  or openai_key_input.strip()
    sg_key   = SENDGRID_API_KEY or sendgrid_key_input.strip()
    from_email = SENDER_EMAIL   or sender_email_input.strip()

    # Validation
    if not oai_key:
        return "❌ OpenAI API key missing.", ""
    if not your_name.strip():
        return "❌ Please enter your name.", ""
    if not your_skills.strip():
        return "❌ Please describe your skills.", ""
    if not target_company.strip() or not target_role.strip():
        return "❌ Please fill in target company and role.", ""
    if send_email and (not sg_key or not from_email or not recipient_email.strip()):
        return "❌ To send email, provide SendGrid key, sender email, and recipient email.", ""

    try:
        client = OpenAI(api_key=oai_key)

        # ── Run the 3-agent pipeline ──────────────────────────────────────────
        status_log, final_email = run_agent_pipeline(
            client=client,
            name=your_name.strip(),
            skills=your_skills.strip(),
            company=target_company.strip(),
            role=target_role.strip(),
            tone=tone,
        )

        # ── Optionally send via SendGrid ──────────────────────────────────────
        if send_email:
            subject_line = ""
            body_lines = []
            for line in final_email.splitlines():
                if line.lower().startswith("subject:"):
                    subject_line = line[8:].strip()
                else:
                    body_lines.append(line)
            email_body = "\n".join(body_lines).strip()

            if not subject_line:
                subject_line = f"Application for {target_role} at {target_company}"

            message = Mail(
                from_email=from_email,
                to_emails=recipient_email.strip(),
                subject=subject_line,
                plain_text_content=email_body,
            )
            sg = SendGridAPIClient(sg_key)
            response = sg.send(message)

            if response.status_code in (200, 202):
                status_log += f"\n\n✅ Email sent to {recipient_email} via SendGrid!"
            else:
                status_log += f"\n\n⚠️ SendGrid returned status {response.status_code}."

        return status_log, final_email

    except Exception as e:
        return f"❌ Error: {str(e)}", ""


# ── Gradio UI ─────────────────────────────────────────────────────────────────
def build_ui():
    with gr.Blocks(title="AI Job Email Generator", css=CSS) as demo:   # ← Add css=CSS here

        gr.HTML(HEADER_HTML)

        with gr.Row():
            with gr.Column(scale=1, elem_classes="panel"):
                gr.Markdown("### 👤 Your profile")
                your_name = gr.Textbox(
                    label="Your full name",
                    placeholder="e.g. Priya Sharma",
                )
                your_skills = gr.Textbox(
                    label="Skills & experience",
                    placeholder=(
                        "e.g. 3 years Python, PyTorch, LLM fine-tuning, RAG pipelines, "
                        "deployed 2 production ML models, B.Tech CS from BITS Pilani..."
                    ),
                    lines=5,
                )

                gr.Markdown("### 🎯 Target job")
                target_company = gr.Textbox(
                    label="Company",
                    placeholder="e.g. Google DeepMind",
                )
                target_role = gr.Textbox(
                    label="Role / position",
                    placeholder="e.g. ML Engineer",
                )
                tone = gr.Dropdown(
                    label="Email tone",
                    choices=[
                        "Professional and confident",
                        "Enthusiastic and warm",
                        "Concise and direct",
                        "Formal and respectful",
                    ],
                    value="Professional and confident",
                )

                gr.Markdown("### 📧 Send options")
                recipient_email = gr.Textbox(
                    label="Recipient email (optional)",
                    placeholder="recruiter@company.com",
                )
                send_email = gr.Checkbox(
                    label="Send email via SendGrid",
                    value=False,
                )
                gr.Markdown("### 🔑 API keys")
                gr.Markdown(
                    "<small style='color:#6b7280'>Set these as Secrets in HF Spaces, "
                    "or enter below for local testing.</small>"
                )
                openai_key_input = gr.Textbox(
                    label="OpenAI API key",
                    placeholder="sk-...",
                    type="password",
                )
                sendgrid_key_input = gr.Textbox(
                    label="SendGrid API key",
                    placeholder="SG...",
                    type="password",
                )
                sender_email_input = gr.Textbox(
                    label="Your verified sender email",
                    placeholder="you@yourdomain.com",
                )

                generate_btn = gr.Button(
                    "✦ Generate Email",
                    variant="primary",
                    size="lg",
                )

            with gr.Column(scale=1, elem_classes="panel"):
                gr.Markdown("### 🤖 Agent pipeline log")
                status_output = gr.Textbox(
                    label="",
                    lines=8,
                    interactive=False,
                    elem_classes="status-box",
                    placeholder="Agent logs will appear here...",
                )

                gr.Markdown("### ✉️ Generated email")
                email_output = gr.Textbox(
                    label="",
                    lines=18,
                    interactive=True,
                    elem_classes="email-output",
                    placeholder="Your crafted email will appear here...",
                    buttons=["copy"],          # Fixed for Gradio 4.44+
                    show_label=False,
                )
          
          
            # ... (your entire UI code stays exactly the same)

        generate_btn.click(
            fn=generate_email,
            inputs=[...],   # your inputs
            outputs=[status_output, email_output],
        )

        gr.HTML(FOOTER_HTML)

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False
        # Do NOT put css= here
    )

    
