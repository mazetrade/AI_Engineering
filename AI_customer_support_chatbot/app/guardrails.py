from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
client = Anthropic()

# ------------------------------------------------------------------ #
#  ESCALATION TRIGGERS                                                 #
#  Keywords that should immediately flag a conversation for           #
#  human review — no AI should handle these alone                     #
# ------------------------------------------------------------------ #

ESCALATION_KEYWORDS = [
    # Legal / financial risk
    "lawyer", "attorney", "legal action", "sue", "lawsuit", "court",
    "fraud", "scam", "chargeback", "dispute", "unauthorized charge",
    # Strong negative emotions
    "furious", "outraged", "unacceptable", "disgusting", "terrible",
    "worst", "horrible", "never shopping", "report you",
    # Safety
    "injured", "hurt", "allergic reaction", "damaged product", "recalled",
    # High-value issues
    "stolen", "missing package", "never arrived", "wrong address",
    "data breach", "hacked", "account compromised",
]

# ------------------------------------------------------------------ #
#  OUT-OF-SCOPE TOPICS                                                 #
#  Things the bot should politely refuse to discuss                   #
# ------------------------------------------------------------------ #

OUT_OF_SCOPE_TOPICS = [
    "politics", "religion", "dating", "medical advice",
    "investment advice", "cryptocurrency", "gambling",
    "competitor products", "personal information",
]

# Prompt to detect complex issues that keywords might miss
ESCALATION_DETECTION_PROMPT = """
You are a support quality classifier for a retail store.
Given a customer message and the conversation history, decide if this 
needs a human agent.

Return ONLY one of:
- "escalate" — if the issue is complex, emotional, legal, or unresolvable by AI
- "continue"  — if the AI can handle it

Escalate when:
- The customer is very angry or distressed
- There is a legal threat or fraud claim
- The issue requires accessing real account/order data
- The customer explicitly asks for a human
- The problem has not been resolved after 3+ exchanges

Return only the label.
"""


def check_keyword_escalation(message: str) -> tuple[bool, str]:
    """
    Fast keyword-based escalation check.
    Runs BEFORE the LLM call to catch obvious cases cheaply.

    Returns:
        (should_escalate: bool, reason: str)
    """
    message_lower = message.lower()
    for keyword in ESCALATION_KEYWORDS:
        if keyword in message_lower:
            return True, f"Sensitive keyword detected: '{keyword}'"
    return False, ""


def check_out_of_scope(message: str) -> bool:
    """
    Checks if the message is clearly outside retail/e-commerce scope.
    Uses simple keyword matching — fast and no API cost.
    """
    message_lower = message.lower()
    for topic in OUT_OF_SCOPE_TOPICS:
        if topic in message_lower:
            return True
    return False


def check_llm_escalation(
    message: str,
    conversation_history: list[dict]
) -> tuple[bool, str]:
    """
    Deeper escalation check using Claude Haiku.
    Only called when keyword check passes — catches emotional tone,
    repeated frustration, and complex multi-turn issues.

    Returns:
        (should_escalate: bool, reason: str)
    """
    # Build a summary of the conversation for the classifier
    history_text = ""
    for msg in conversation_history[-6:]:  # Only last 6 messages for efficiency
        role = "Customer" if msg["role"] == "user" else "Bot"
        history_text += f"{role}: {msg['content']}\n"

    classification_message = (
        f"Conversation so far:\n{history_text}\n"
        f"Latest message: {message}"
    )

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=10,
        system=ESCALATION_DETECTION_PROMPT,
        messages=[{"role": "user", "content": classification_message}]
    )

    result = response.content[0].text.strip().lower()
    if result == "escalate":
        return True, "Conversation complexity requires human agent"
    return False, ""


def run_guardrails(
    message: str,
    conversation_history: list[dict]
) -> dict:
    """
    Master guardrail function — runs all checks in order.
    Returns a structured result the endpoint can act on.

    Check order (cheapest first):
    1. Out-of-scope keyword check  → free, instant
    2. Escalation keyword check    → free, instant
    3. LLM escalation check        → small API call (Haiku)

    Returns a dict:
    {
        "action":  "continue" | "escalate" | "out_of_scope",
        "reason":  str,
        "message": str   ← the reply to send to the user
    }
    """
    # --- Check 1: Out of scope ---
    if check_out_of_scope(message):
        return {
            "action": "out_of_scope",
            "reason": "Topic outside retail support scope",
            "message": (
                "I'm your retail support assistant, so I can only help with "
                "orders, products, shipping, returns, and payments. "
                "For anything else, I'd suggest reaching out to the right "
                "specialist. Is there anything store-related I can help you with?"
            )
        }

    # --- Check 2: Keyword escalation ---
    should_escalate, reason = check_keyword_escalation(message)
    if should_escalate:
        return {
            "action": "escalate",
            "reason": reason,
            "message": (
                "I completely understand your concern and I want to make sure "
                "you get the best possible help. Let me connect you with one of "
                "our specialist agents who can resolve this for you directly. "
                "\n\nPlease contact us at:\n"
                "- Email: support@store.com\n"
                "- Phone: 1-800-STORE-01 (Mon-Fri 9am-6pm EST)\n"
                "- Live chat: Available on our website\n\n"
                "Your reference number for this conversation will be provided "
                "by the agent. We apologize for the inconvenience."
            )
        }

    # --- Check 3: LLM escalation (only if conversation has some history) ---
    if len(conversation_history) >= 4:  # At least 2 exchanges before checking
        should_escalate, reason = check_llm_escalation(message, conversation_history)
        if should_escalate:
            return {
                "action": "escalate",
                "reason": reason,
                "message": (
                    "It looks like your issue needs some extra attention from "
                    "our team. I'd like to connect you with a human specialist "
                    "who can access your account and resolve this properly.\n\n"
                    "Please contact us at:\n"
                    "- Email: support@store.com\n"
                    "- Phone: 1-800-STORE-01 (Mon-Fri 9am-6pm EST)\n\n"
                    "Thank you for your patience!"
                )
            }

    # --- All checks passed — safe to continue ---
    return {
        "action": "continue",
        "reason": "",
        "message": ""
    }