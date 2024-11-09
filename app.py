import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import openai
import os

# Load OpenAI API key from Streamlit secrets or environment variable
openai.api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

# Set up the app title and description
st.title("Chat with LLaMA and OpenAI Models")
st.write("Choose between Meta's LLaMA model and OpenAI's models to chat.")

# Model selection
model_choice = st.selectbox("Choose a model", ["LLaMA (Hugging Face)", "OpenAI GPT"])

# Load Meta's LLaMA model from Hugging Face
@st.cache_resource
def load_llama_model():
    MODEL_NAME = "meta-llama/Llama-2-7b-hf"  # Update with available LLaMA model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model, tokenizer

if model_choice == "LLaMA (Hugging Face)":
    model, tokenizer = load_llama_model()

# Chat history to keep context
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# User input for chat prompt
user_input = st.text_input("You:", "")

# Display chat history
for i, (user, bot) in enumerate(st.session_state.chat_history):
    st.write(f"You: {user}")
    st.write(f"Bot: {bot}")

# Generate response based on selected model
if st.button("Send") and user_input:
    st.session_state.chat_history.append((user_input, ""))  # Append user input temporarily

    if model_choice == "LLaMA (Hugging Face)":
        # Prepare input and generate response with LLaMA
        inputs = tokenizer(user_input, return_tensors="pt").to(model.device)
        outputs = model.generate(inputs.input_ids, max_length=100, num_return_sequences=1)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    elif model_choice == "OpenAI GPT":
        # Generate response with OpenAI API
        try:
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=user_input,
                max_tokens=100,
                stop=None,
                temperature=0.7
            ).choices[0].text.strip()
        except Exception as e:
            response = f"Error: {e}"

    # Update chat history with bot response
    st.session_state.chat_history[-1] = (user_input, response)
    st.write(f"Bot: {response}")

