import streamlit as st
import requests

# ------------------------------------------------------------------ #
#  CONFIG                                                              #
# ------------------------------------------------------------------ #

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Retail Support Chat",
    page_icon="🛍️",
    layout="centered"
)

# ------------------------------------------------------------------ #
#  CUSTOM CSS                                                          #
#  Styles the chat bubbles, badges, and escalation banner             #
# ------------------------------------------------------------------ #

st.markdown("""
<style>
    /* User bubble — right aligned */
    .user-bubble {
        background-color: #4F46E5;
        color: white;
        padding: 12px 16px;
        border-radius: 18px 18px 4px 18px;
        margin: 6px 0 6px 20%;
        text-align: right;
        font-size: 15px;
        line-height: 1.5;
    }

    /* Bot bubble — left aligned */
    .bot-bubble {
        background-color: #F3F4F6;
        color: #111827;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 4px;
        margin: 6px 20% 6px 0;
        font-size: 15px;
        line-height: 1.5;
    }

    /* Intent badge under each bot message */
    .intent-badge {
        font-size: 11px;
        color: #6B7280;
        margin: 2px 0 10px 6px;
        font-style: italic;
    }

    /* Escalation warning banner */
    .escalation-banner {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
        font-size: 14px;
        color: #92400E;
    }

    /* Out of scope banner */
    .oos-banner {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
        font-size: 14px;
        color: #1E40AF;
    }

    /* Hide the default Streamlit header */
    header { visibility: hidden; }

    /* Make the chat area scrollable */
    .chat-container {
        max-height: 520px;
        overflow-y: auto;
        padding: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ #
#  SESSION STATE INITIALISATION                                        #
#  st.session_state persists values across reruns (like React state)  #
# ------------------------------------------------------------------ #

if "session_id" not in st.session_state:
    st.session_state.session_id = None
    # None means no conversation started yet
    # Gets set on the first API response

if "messages" not in st.session_state:
    st.session_state.messages = []
    # List of dicts: {"role": "user"/"assistant", "content": str,
    #                 "intent": str, "action": str}

if "is_escalated" not in st.session_state:
    st.session_state.is_escalated = False
    # Once escalated, we lock the input so the user contacts support


# ------------------------------------------------------------------ #
#  HELPER FUNCTIONS                                                    #
# ------------------------------------------------------------------ #

def send_message(user_text: str) -> dict:
    """
    Calls POST /chat on the FastAPI backend.
    Returns the full response dict or raises on failure.
    """
    payload = {
        "message": user_text,
        "session_id": st.session_state.session_id
    }
    response = requests.post(f"{API_URL}/chat", json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def reset_conversation():
    """
    Clears the session both on the backend and in Streamlit state.
    """
    if st.session_state.session_id:
        try:
            requests.delete(
                f"{API_URL}/chat/reset/{st.session_state.session_id}",
                timeout=10
            )
        except Exception:
            pass  # If backend is down, still clear local state

    st.session_state.session_id = None
    st.session_state.messages = []
    st.session_state.is_escalated = False


def render_message(msg: dict):
    """
    Renders a single message as an HTML bubble.
    Handles user bubbles, bot bubbles, and action banners.
    """
    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-bubble">{msg["content"]}</div>',
            unsafe_allow_html=True
        )

    elif msg["role"] == "assistant":
        action = msg.get("action", "continue")

        # Choose the right bubble style based on action
        if action == "escalate":
            st.markdown(
                f'<div class="escalation-banner">🔔 <strong>Escalated to human agent</strong><br>{msg["content"]}</div>',
                unsafe_allow_html=True
            )
        elif action == "out_of_scope":
            st.markdown(
                f'<div class="oos-banner">ℹ️ {msg["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="bot-bubble">{msg["content"]}</div>',
                unsafe_allow_html=True
            )

        # Show intent badge under normal bot replies
        if action == "continue" and msg.get("intent"):
            intent_label = msg["intent"].replace("_", " ").title()
            st.markdown(
                f'<div class="intent-badge">Topic: {intent_label}</div>',
                unsafe_allow_html=True
            )


# ------------------------------------------------------------------ #
#  SIDEBAR                                                             #
# ------------------------------------------------------------------ #

with st.sidebar:
    st.title("🛍️ Retail Support")
    st.markdown("---")

    # Session info
    if st.session_state.session_id:
        st.markdown("**Session**")
        # Show a shortened version of the session ID
        short_id = st.session_state.session_id[:8] + "..."
        st.code(short_id, language=None)
        st.markdown(
            f"**Messages:** {len(st.session_state.messages)}"
        )
    else:
        st.info("No active session yet.")

    st.markdown("---")

    # Reset button
    if st.button("🔄 New Conversation", use_container_width=True):
        reset_conversation()
        st.rerun()

    st.markdown("---")

    # Quick topic buttons — clicking one pre-fills a question
    st.markdown("**Quick topics**")
    quick_topics = {
        "📦 Track my order": "How can I track my order?",
        "↩️ Return an item": "What is your return policy?",
        "🚚 Shipping info": "How long does shipping take?",
        "💳 Payment options": "What payment methods do you accept?",
        "👟 Nike Air Max": "Tell me about the Nike Air Max 270",
    }

    for label, question in quick_topics.items():
        if st.button(label, use_container_width=True):
            # Inject the question as if the user typed it
            if not st.session_state.is_escalated:
                with st.spinner("Getting answer..."):
                    try:
                        data = send_message(question)
                        st.session_state.session_id = data["session_id"]
                        st.session_state.messages.append({
                            "role": "user",
                            "content": question,
                            "intent": "",
                            "action": ""
                        })
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": data["reply"],
                            "intent": data["intent"],
                            "action": data["action"]
                        })
                        if data["action"] == "escalate":
                            st.session_state.is_escalated = True
                    except Exception as e:
                        st.error(f"Error: {e}")
                st.rerun()

    st.markdown("---")
    st.markdown(
        "<small style='color: #9CA3AF;'>Powered by Claude + FastAPI</small>",
        unsafe_allow_html=True
    )


# ------------------------------------------------------------------ #
#  MAIN CHAT AREA                                                      #
# ------------------------------------------------------------------ #

st.title("Customer Support Chat")
st.markdown(
    "<p style='color: #6B7280; margin-top: -12px;'>"
    "Ask me anything about orders, products, shipping, or returns.</p>",
    unsafe_allow_html=True
)

# Render all messages in the conversation
chat_placeholder = st.container()
with chat_placeholder:
    if not st.session_state.messages:
        # Welcome message when no conversation yet
        st.markdown(
            '<div class="bot-bubble">👋 Hi! I\'m your retail support assistant. '
            'How can I help you today? You can ask me about orders, products, '
            'shipping, returns, or payments.</div>',
            unsafe_allow_html=True
        )
    else:
        for msg in st.session_state.messages:
            render_message(msg)

# ------------------------------------------------------------------ #
#  INPUT AREA                                                          #
# ------------------------------------------------------------------ #

st.markdown("---")

# If escalated, lock the input and show contact info
if st.session_state.is_escalated:
    st.warning(
        "This conversation has been escalated to our support team. "
        "Please contact us directly:\n\n"
        "📧 support@store.com  |  📞 1-800-STORE-01"
    )
else:
    # Normal chat input
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])

        with col1:
            user_input = st.text_input(
                label="Message",
                placeholder="Type your message here...",
                label_visibility="collapsed"
            )
        with col2:
            submitted = st.form_submit_button(
                "Send",
                use_container_width=True
            )

    # Handle form submission
    if submitted and user_input.strip():
        with st.spinner("Thinking..."):
            try:
                data = send_message(user_input.strip())

                # Save session_id on first reply
                st.session_state.session_id = data["session_id"]

                # Append user message
                st.session_state.messages.append({
                    "role": "user",
                    "content": user_input.strip(),
                    "intent": "",
                    "action": ""
                })

                # Append bot reply
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": data["reply"],
                    "intent": data["intent"],
                    "action": data["action"]
                })

                # Lock input if escalated
                if data["action"] == "escalate":
                    st.session_state.is_escalated = True

            except requests.exceptions.ConnectionError:
                st.error(
                    "Cannot reach the API. Make sure FastAPI is running: "
                    "`uvicorn app.main:app --reload`"
                )
            except Exception as e:
                st.error(f"Something went wrong: {e}")

        st.rerun()