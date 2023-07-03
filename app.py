import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os
from dotenv import load_dotenv, find_dotenv
from langchain.llms import OpenAI
from langchain.chains import VectorDBQA

load_dotenv(find_dotenv())
pdf = st.file_uploader("Upload Pdf File", type="pdf")

def get_pdf_text(pdf):
  
    try:
        if not PdfReader.is_pdf(pdf): 
         st.write('Please upload a valid PDF file.')
        else:
         pdf_reader = PdfReader(pdf)
     
    except Exception as e:
       st.write('Error reading PDF file:', e)
    text_list = []
    for page in pdf_reader.pages:
        text_list.append(page.extract_text())
    text = ''.join(text_list)
    return text
text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len)
def get_chunks(text):
    
    chunks = text_splitter.split_text(text=text)
    return chunks

def get_filename(pdf):
    store_name = os.path.join("files", pdf.name[:-4])
    return store_name

def setup_qa(pdf, embeddings, persist_directory):
    store_name = get_filename(pdf)
    if os.path.exists(f"{store_name}.pdf"):
        VectorStore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    else:
        with open(f"{store_name}.pdf", "wb") as f:
            # filename = f"{store_name}.pdf"
            chunks = get_chunks(get_pdf_text(pdf))
            VectorStore = Chroma.from_texts(chunks, embedding=embeddings, persist_directory=persist_directory)
            VectorStore.persist()
    return VectorStore, VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=VectorStore)

if pdf is not None:
    persist_directory = 'db'
    embeddings = OpenAIEmbeddings()

    VectorStore, qa = setup_qa(pdf, embeddings, persist_directory)

    prompt = st.text_input('Input your question here')
    if prompt:
        res = qa.run(prompt)
        st.write(res)
