import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import openai
import os

# Set up the app title and description
st.title("LLM with Llama and OpenAI in Streamlit")
st.write("Choose between Llama (Meta) and OpenAI's models for text generation.")

# Set up OpenAI API key
# Ensure this is set in your environment variables
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Model selection
model_choice = st.selectbox("Choose a model", ["Llama (Meta)", "OpenAI"])

# Load Meta's Llama model


@st.cache_resource
def load_llama_model():
    # Replace with an available Llama model variant
    MODEL_NAME = "meta-llama/Llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model, tokenizer


if model_choice == "Llama (Meta)":
    model, tokenizer = load_llama_model()

# User input for text generation
prompt = st.text_input("Enter your prompt:", "Once upon a time,")

# Slider for maximum token length
max_length = st.slider("Max tokens for generation:", 10, 100, 50)

# Generate text based on the model choice
if st.button("Generate Text"):
    if model_choice == "Llama (Meta)":
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            inputs.input_ids, max_length=max_length, num_return_sequences=1)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.write("Generated Text (Llama):")
        st.write(generated_text)

    elif model_choice == "OpenAI":
        try:
            response = openai.Completion.create(
                # Choose your OpenAI model (or "gpt-3.5-turbo" for chat-based ones)
                model="text-davinci-003",
                prompt=prompt,
                max_tokens=max_length
            )
            generated_text = response['choices'][0]['text'].strip()
            st.write("Generated Text (OpenAI):")
            st.write(generated_text)
        except Exception as e:
            st.error(f"OpenAI API Error: {e}")
