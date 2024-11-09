import streamlit as st
import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import PyPDF2

# Load OpenAI API key
openai.api_key = st.secrets.get("OPENAI_API_KEY")

# Set up Streamlit app title and description
st.title("Document Question-Answering Chatbot")
st.write("Upload a PDF document, then ask questions based on the document content.")

# Initialize an empty document text variable
document_text = ""

# Upload document
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Read PDF content
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    document_text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        document_text += page.extract_text()

    st.write("Document successfully uploaded and processed.")

    # Step 1: Split text into chunks for embeddings
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_text(document_text)

    # Step 2: Create embeddings for the document chunks
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectorstore = FAISS.from_texts(texts, embeddings)

    # Step 3: Set up LangChain Retrieval QA chain
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(model="gpt-4-turbo"),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    # Ask questions based on the uploaded document
    user_question = st.text_input("Ask a question about the document:")
    if user_question:
        # Run the question through the retrieval QA chain
        response = qa_chain({"query": user_question})
        answer = response['result']
        st.write("Answer:", answer)

        # Optional: Show source documents
        st.write("Source documents:")
        for doc in response["source_documents"]:
            st.write("- ", doc.page_content[:200] + "...")  # Displaying a snippet of each source





