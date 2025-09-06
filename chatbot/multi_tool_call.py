import os
import base64
from io import BytesIO
from PIL import Image
import streamlit as st
import urllib.parse

from langchain_groq import ChatGroq
import google.generativeai as genai

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState

# --- Step 1: API Keys ---
groq_api_key = os.getenv("GROQ_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key or not groq_api_key:
    st.error("Please set GOOGLE_API_KEY and GROQ_API_KEY in your environment.")
    st.stop()

# Configure Gemini
genai.configure(api_key=google_api_key)

# Gemini Models
gemini_text = genai.GenerativeModel("gemini-2.5-flash")
gemini_image = genai.GenerativeModel("gemini-2.5-flash-image-preview")

# Groq Models
llm_reasoning = ChatGroq(api_key=groq_api_key, model="llama-3.3-70b-versatile", temperature=0)
llm_fast = ChatGroq(api_key=groq_api_key, model="gemma2-9b-it", temperature=0)
llm_backup = ChatGroq(api_key=groq_api_key, model="llama-3.1-8b-instant", temperature=0)

# --- Step 2: Define Tools with proper docstrings ---
@tool
def llm_reasoning_tool(query: str) -> str:
    """Call the reasoning LLM for math, analysis, or reasoning queries."""
    return llm_reasoning.invoke(query).content

@tool
def llm_fast_tool(query: str) -> str:
    """Call the fast LLM for quick summaries or fast answers."""
    return llm_fast.invoke(query).content

@tool
def llm_backup_tool(query: str) -> str:
    """Call the backup LLM for general queries."""
    return llm_backup.invoke(query).content

@tool
def gemini_text_tool(query: str) -> str:
    """Call Gemini to generate text responses for general queries."""
    try:
        response = gemini_text.generate_content(query)
        return response.text or "[Gemini returned no text]"
    except Exception as e:
        return f"[Gemini Text Error] {e}"

@tool
def generate_image_gemini_tool(prompt: str) -> dict:
    """Generate an image using Gemini and return text + proper image bytes."""
    try:
        response = gemini_image.generate_content(prompt)
    except Exception as e:
        return {"text": f"[Gemini Image Error] {e}", "image": None}

    if not response.candidates:
        return {"text": "[Gemini Image] No candidates returned.", "image": None}

    candidate = response.candidates[0]
    description_text = ""
    generated_image = None

    for part in candidate.content.parts:
        if getattr(part, "text", None):
            description_text += part.text + "\n"

        if getattr(part, "inline_data", None):
            try:
                img_data_base64 = part.inline_data.data
                if isinstance(img_data_base64, str):
                    img_bytes = base64.b64decode(img_data_base64)
                else:
                    img_bytes = bytes(img_data_base64)

                # Verify the image
                img = Image.open(BytesIO(img_bytes))
                img.verify()  # ensure it's valid
                generated_image = img_bytes
            except Exception as e:
                description_text += f"\n[Gemini Inline Data Decode Error: {e}]"

    return {"text": description_text.strip(), "image": generated_image}

# --- Step 3: Tool Router ---
def tool_router(query: str):
    q = query.lower()
    if "math" in q or "analyze" in q or "reason" in q:
        return llm_reasoning_tool.invoke(query)
    elif "quick" in q or "fast" in q or "summarize" in q:
        return llm_fast_tool.invoke(query)
    elif any(word in q for word in ["image", "draw", "picture", "generate"]):
        return generate_image_gemini_tool.invoke(query)
    elif "gemini" in q:
        return gemini_text_tool.invoke(query)
    else:
        return llm_backup_tool.invoke(query)

# --- Step 4: Agent Logic ---
def call_tool(state: MessagesState):
    last_message = state["messages"][-1]
    if isinstance(last_message, HumanMessage):
        response = tool_router(last_message.content)
        return {"messages": [AIMessage(content=str(response))]}
    return {"messages": []}

# --- Step 5: Build LangGraph Workflow ---
workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_tool)
workflow.set_entry_point("agent")
app = workflow.compile()

# --- Step 6: Streamlit UI ---
import streamlit as st
from streamlit import session_state as state

# Helper to get tool name
def get_tool_name(query):
    q = query.lower()
    if "image" in q or "picture" in q or "draw" in q:
        return "generate_image_gemini_tool"
    elif "math" in q or "analyze" in q or "reason" in q:
        return "llm_reasoning_tool"
    elif "quick" in q or "fast" in q or "summarize" in q:
        return "llm_fast_tool"
    else:
        return "llm_backup_tool"

st.title("Multi-Tool Agent")

if 'history' not in state:
    state.history = []

# Display chat history
for q, a, tool_name in state.history:
    # Question right, then answer left (with image above text if present)
    st.markdown(f"<div style='text-align:right;margin-bottom:2px;width:70%;margin-left:auto;background-color:#f0f0f0;padding:8px;border-radius:6px;'><b>Question:</b> {q}</div>", unsafe_allow_html=True)
    if tool_name == "generate_image_gemini_tool" and isinstance(a, dict):
        answer_text = a.get("text", "")
        img_bytes = a.get("image")
        if img_bytes:
            st.image(img_bytes, caption="Generated Image", use_column_width=True)
        st.markdown(f"<div style='text-align:left;margin-bottom:16px;width:70%;background-color:#e0f7fa;padding:8px;border-radius:6px;'><b>Tool Used:</b> {tool_name}<br><b>Answer:</b> {answer_text}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='text-align:left;margin-bottom:16px;width:70%;background-color:#e0f7fa;padding:8px;border-radius:6px;'><b>Tool Used:</b> {tool_name}<br><b>Answer:</b> {a}</div>", unsafe_allow_html=True)

# Move textarea to bottom
st.markdown("<div style='height:30vh;'></div>", unsafe_allow_html=True)
user_input = st.text_area("Enter your query:", height=40, key="input", help="Type your question here",
                         placeholder="Ask something...",
                         )

if st.button("Send") and user_input.strip():
    tool_name = get_tool_name(user_input)
    with st.spinner(f"Calling {tool_name}..."):
        response = tool_router(user_input)
        # Clear input workaround: set a dummy key
        st.text_area("Enter your query:", value="", key="input_clear", height=40, help="Type your question here", placeholder="Ask something...", disabled=True)
        # If image tool, store dict so chat history can render image above text
        if tool_name == "generate_image_gemini_tool" and isinstance(response, dict):
            state.history.append((user_input, response, tool_name))
            st.rerun()
        else:
            state.history.append((user_input, str(response), tool_name))
            st.rerun()
