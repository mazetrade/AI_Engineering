import os
from anthropic import Anthropic
from dotenv import load_dotenv
from app.knowledge_base import knowledge_base

load_dotenv()
client = Anthropic()

SYSTEM_PROMPT = """
You are a helpful customer support assistant for a retail and e-commerce store.

You will be given CONTEXT from our knowledge base at the start of each message.
Always base your answers on this context when it is relevant.
If the context does not contain enough information to answer, say so honestly
and suggest the customer contact support directly.

You help customers with:
- Order status and tracking
- Product information and availability
- Returns and refund policies
- Shipping questions
- Payment issues
- General store inquiries

Rules:
- Be friendly, concise, and professional
- Never invent product details, prices, or policies
- If the issue is too complex, suggest escalating to a human agent
- Stay focused on retail/e-commerce topics only
"""

INTENT_PROMPT = """
You are a classifier. Given a customer message, return ONLY one of these labels:
- order_tracking
- product_inquiry
- returns_refunds
- shipping
- payment
- general
- out_of_scope

Return only the label, nothing else.
"""


def detect_intent(user_message: str) -> str:
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=20,
        system=INTENT_PROMPT,
        messages=[{"role": "user", "content": user_message}]
    )
    intent = response.content[0].text.strip().lower()
    valid_intents = {
        "order_tracking", "product_inquiry", "returns_refunds",
        "shipping", "payment", "general", "out_of_scope"
    }
    return intent if intent in valid_intents else "general"


def get_chat_response(conversation_history: list[dict]) -> str:
    """
    Retrieves relevant context from the knowledge base, injects it
    into the latest user message, then sends everything to Claude.
    """
    # 1. Get the last user message (it's always the last item in history)
    last_user_message = conversation_history[-1]["content"]

    # 2. Search the knowledge base for relevant documents
    relevant_docs = knowledge_base.search(last_user_message, top_k=3)

    # 3. Format those documents as readable context text
    context = knowledge_base.format_context(relevant_docs)

    # 4. Build an augmented version of the conversation history
    #    We inject the context into the FIRST user message turn only
    #    This avoids sending redundant context on every single message
    augmented_history = []
    for i, message in enumerate(conversation_history):
        if i == len(conversation_history) - 1 and message["role"] == "user":
            # Wrap the latest user message with retrieved context
            augmented_content = (
                f"CONTEXT FROM KNOWLEDGE BASE:\n{context}\n\n"
                f"CUSTOMER MESSAGE:\n{message['content']}"
            )
            augmented_history.append({
                "role": "user",
                "content": augmented_content
            })
        else:
            # All previous messages stay as-is
            augmented_history.append(message)

    # 5. Send to Claude
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=augmented_history
    )
    return response.content[0].text