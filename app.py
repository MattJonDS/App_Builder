import streamlit as st
import openai
import PyPDF2
import os

# Load OpenAI API key
openai.api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

# Set up Streamlit app title and description
st.title("Basic Document Question-Answering Bot")
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
        # Directly call OpenAI to answer the question based on document content
        prompt = f"{document_text}\n\nQuestion: {user_question}\nAnswer:"
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=150,
            temperature=0.5
        )
        answer = response.choices[0].text.strip()
        st.write("Answer:", answer)







