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
    
    # Summarize the document in smaller sections to reduce token usage
    max_tokens_per_section = 2000  # Adjust as needed
    sections = [document_text[i:i+max_tokens_per_section] for i in range(0, len(document_text), max_tokens_per_section)]
    
    summaries = []
    for section in sections:
        summary_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Use a lighter model for summarization
            messages=[
                {"role": "system", "content": "Summarize the following text."},
                {"role": "user", "content": section}
            ],
            max_tokens=300,
            temperature=0.3
        )
        summaries.append(summary_response['choices'][0]['message']['content'].strip())

    # Combine summaries into a single summarized document
    summarized_text = " ".join(summaries)

    # Prompt the user for a question
    user_question = st.text_input("Ask a question about the document:")
    
    if user_question:
        # Use the summarized text to answer the user's question
        messages = [
            {"role": "system", "content": "You are an assistant that answers questions based on a provided document summary."},
            {"role": "user", "content": f"Summary of document:\n{summarized_text}"},
            {"role": "user", "content": f"Question: {user_question}"}
        ]

        # Call OpenAI's ChatCompletion endpoint with gpt-4 for Q&A
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            max_tokens=150,
            temperature=0.5
        )
        
        # Extract and display the response
        answer = response['choices'][0]['message']['content'].strip()
        st.write("Answer:", answer)









