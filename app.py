import streamlit as st
import openai
import PyPDF2
import os

# Load OpenAI API key
openai.api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

# Set up Streamlit app title and description
st.title("Advanced Document Question-Answering Bot")
st.write("Upload a PDF document and ask questions directly based on its content.")

# Upload document
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
document_text = ""

if uploaded_file is not None:
    # Read PDF content
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    document_text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        document_text += page.extract_text()
    
    st.write("Document successfully uploaded.")

    # Prompt the user for a question
    user_question = st.text_input("Ask a question about the document:")
    
    if user_question:
        # Prepare the conversation for the ChatCompletion endpoint
        messages = [
            {"role": "system", "content": "You are an assistant that answers questions based on a provided document."},
            {"role": "user", "content": f"Document content:\n{document_text}"},
            {"role": "user", "content": f"Question: {user_question}"}
        ]

        # Call OpenAI's ChatCompletion endpoint with gpt-4
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Use gpt-4 for higher sophistication
            messages=messages,
            max_tokens=150,
            temperature=0.5
        )
        
        # Extract and display the response
        answer = response['choices'][0]['message']['content'].strip()
        st.write("Answer:", answer)








