from pathlib import Path
import sys
import os

# --------------------------------------------------
# PROJECT ROOT
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# --------------------------------------------------
# IMPORTS
# --------------------------------------------------

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI


# --------------------------------------------------
# ENVIRONMENT
# --------------------------------------------------

load_dotenv(ROOT_DIR / ".env")

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL")


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="AI Assistant | AgroWeedGuard",
    page_icon="🤖",
    layout="wide"
)


# --------------------------------------------------
# API VALIDATION
# --------------------------------------------------

if not NVIDIA_API_KEY:

    st.error(
        "NVIDIA API key not found. "
        "Please check your .env file."
    )

    st.stop()


if not LLM_MODEL:

    st.error(
        "LLM model not configured. "
        "Please check your .env file."
    )

    st.stop()


# --------------------------------------------------
# NVIDIA CLIENT
# --------------------------------------------------

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)


# --------------------------------------------------
# SYSTEM PROMPT
# --------------------------------------------------

SYSTEM_PROMPT = """
You are AgroWeedGuard AI, an agricultural AI assistant.

AgroWeedGuard is an AI-powered weed detection and
classification system.

Your responsibilities are to:

- Explain detected weed species.
- Explain model predictions and confidence scores.
- Provide useful agricultural information.
- Explain potential effects of weeds on crops.
- Discuss general weed management and control approaches.
- Help users understand the AgroWeedGuard models.
- Answer follow-up questions while maintaining conversation context.

Be clear, practical, and concise.

When discussing a model prediction, distinguish between
the model's prediction and general agricultural knowledge.

Do not claim that a prediction is guaranteed to be correct.
"""


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "chat_messages" not in st.session_state:

    st.session_state.chat_messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]


# --------------------------------------------------
# PAGE TITLE
# --------------------------------------------------

st.title("🤖 AgroWeedGuard AI Assistant")

st.markdown(
    """
Ask questions about detected weeds, their characteristics,
potential crop impact, management strategies, or the
AgroWeedGuard models.
"""
)

st.divider()


# --------------------------------------------------
# DISPLAY CHAT HISTORY
# --------------------------------------------------

for message in st.session_state.chat_messages:

    if message["role"] == "system":
        continue

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# --------------------------------------------------
# CHAT INPUT
# --------------------------------------------------

user_prompt = st.chat_input(
    "Ask AgroWeedGuard AI..."
)


# --------------------------------------------------
# SEND MESSAGE
# --------------------------------------------------

if user_prompt:

    # ----------------------------------------------
    # DISPLAY USER MESSAGE
    # ----------------------------------------------

    st.session_state.chat_messages.append(
        {
            "role": "user",
            "content": user_prompt
        }
    )

    with st.chat_message("user"):

        st.markdown(user_prompt)


    # ----------------------------------------------
    # GENERATE RESPONSE
    # ----------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "AgroWeedGuard AI is thinking..."
        ):

            try:

                response = client.chat.completions.create(

                    model=LLM_MODEL,

                    messages=(
                        st.session_state.chat_messages
                    ),

                    temperature=0.6,

                    top_p=0.95,

                    max_tokens=2048,

                    stream=False
                )


                answer = (
                    response
                    .choices[0]
                    .message
                    .content
                )


                # ----------------------------------
                # DISPLAY RESPONSE
                # ----------------------------------

                st.markdown(answer)


                # ----------------------------------
                # SAVE RESPONSE
                # ----------------------------------

                st.session_state.chat_messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )


            except Exception as e:

                st.error(
                    f"LLM request failed: {e}"
                )


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.subheader("🤖 AgroWeedGuard AI")

    st.caption(
        f"Model: {LLM_MODEL}"
    )

    st.divider()

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True
    ):

        st.session_state.chat_messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        ]

        st.rerun()