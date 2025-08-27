import streamlit as st
import google.generativeai as genai

# Configure Gemini with your API key
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat" not in st.session_state:
    model = genai.GenerativeModel("gemini-1.5-flash")
    st.session_state.chat = model.start_chat(history=[])
st.title("💬 Welcome To My ChatBot ")
# --- Clear Chat Button ---
if st.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    model = genai.GenerativeModel("gemini-1.5-flash")
    st.session_state.chat = model.start_chat(history=[])
    st.rerun()
# Display previous chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
prompt = st.chat_input("Ask Anything :")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Send message to Gemini with streaming
        response = st.session_state.chat.send_message(prompt, stream=True)
        # stream=False → Waits until Gemini finishes the full reply, then shows it all at once.
        # stream=True → Gets the reply in real time (chunk by chunk), so you can display a typing effect.
        for chunk in response:
            if chunk.text:
                full_response += chunk.text
                message_placeholder.markdown(full_response + "▌")

        # Final clean response
        message_placeholder.markdown(full_response)
    # Save assistant reply
    st.session_state.messages.append({"role": "assistant", "content": full_response})
