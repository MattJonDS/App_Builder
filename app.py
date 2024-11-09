import streamlit as st
import openai
import os

# Load OpenAI API key
openai.api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

st.title("Sophisticated Chatbot with GPT-4 Turbo")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant with expertise in various domains."}]

user_input = st.text_area("You:", "", height=100)

# Display conversation history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.write(f"You: {message['content']}")
    elif message["role"] == "assistant":
        st.write(f"Bot: {message['content']}")

# Generate response
if st.button("Send") and user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=st.session_state.messages,
        max_tokens=200,
        temperature=0.7
    ).choices[0].message['content'].strip()

    # Append response to chat history and display it
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.write(f"Bot: {response}")

if st.button("Clear Chat"):
    st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant with expertise in various domains."}]



