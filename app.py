import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import openai
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import os

# Load OpenAI API key from Streamlit secrets or environment variable
openai.api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

# Title and description for the chatbot
st.title("Simple Chatbot with Streamlit")
st.write("Choose between an OpenAI model and a Hugging Face model for text generation.")

# Model selection
model_choice = st.selectbox("Choose a model", ["Hugging Face (Transformers)", "OpenAI API"])

# Load Hugging Face model and tokenizer
@st.cache_resource
def load_huggingface_model():
    model_name = "gpt2"  # You can use any suitable Hugging Face model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model, tokenizer

if model_choice == "Hugging Face (Transformers)":
    model, tokenizer = load_huggingface_model()

# Maintain chat history in session state
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# User input
user_input = st.text_input("You:", "")

# Display chat history
for i, (user, bot) in enumerate(st.session_state.chat_history):
    st.write(f"You: {user}")
    st.write(f"Bot: {bot}")

# Generate response on button click
if st.button("Send") and user_input:
    st.session_state.chat_history.append((user_input, ""))  # Temporarily store user input

    if model_choice == "Hugging Face (Transformers)":
        # Tokenize and generate response with Hugging Face model
        inputs = tokenizer(user_input, return_tensors="pt").to(model.device)
        outputs = model.generate(inputs.input_ids, max_length=100, num_return_sequences=1)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    elif model_choice == "OpenAI API":
        # Generate response with OpenAI API
        try:
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=user_input,
                max_tokens=100,
                temperature=0.7
            ).choices[0].text.strip()
        except Exception as e:
            response = f"Error: {e}"

    # Update chat history with bot response
    st.session_state.chat_history[-1] = (user_input, response)
    st.write(f"Bot: {response}")


