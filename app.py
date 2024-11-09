import streamlit as st
import openai
import os

# Load OpenAI API key
openai.api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

# Title and description for the app
st.title("Chat Completion with GPT-4 Turbo")
st.write("Chat with OpenAI's GPT-4-turbo model. Enjoy multi-turn conversations with chat history!")

# Initialize conversation history in session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "system", "content": "You are a helpful assistant."}]

# User input
user_input = st.text_input("You:", "")

# Display conversation history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.write(f"You: {message['content']}")
    elif message["role"] == "assistant":
        st.write(f"Bot: {message['content']}")

# Generate a response if user has entered input and clicked send
if st.button("Send") and user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Call OpenAI API to generate chat completion
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=st.session_state.messages,
        max_tokens=200,
        temperature=0.7
    )
    
    # Get assistant response and add it to the chat history
    assistant_message = response.choices[0].message['content'].strip()
    st.session_state.messages.append({"role": "assistant", "content": assistant_message})
    
    # Display assistant response
    st.write(f"Bot: {assistant_message}")

# Button to clear the chat history
if st.button("Clear Chat"):
    st.session_state["messages"] = [{"role": "system", "content": "You are a helpful assistant."}]




