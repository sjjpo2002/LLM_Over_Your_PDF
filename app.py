from json import load
from sqlalchemy import true
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceHubEmbeddings, HuggingFaceEmbeddings
import os
import requests
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from html_templates import css, bot_template, user_template

hf_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")


def get_pdf_text(pdfs):
    text = ""
    for pdf in pdfs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chuncks(text):
    splitter = CharacterTextSplitter(
        separator="\n", chunk_size=500, chunk_overlap=50, length_function=len
    )
    chunks = splitter.split_text(text)
    return chunks


def get_vectors_from_hf_API(chunks):
    repo_id = "sentence-transformers/all-mpnet-base-v2"
    # wrapper around Sentence Transformers
    hf = HuggingFaceHubEmbeddings(
        repo_id=repo_id, task="feature-extraction", huggingfacehub_api_token=hf_api_key
    )
    vectors = hf.embed_documents(chunks)
    return vectors


def get_vectors_from_sentence_transformer(chunks):
    model_name = "thenlper/gte-small"
    # model = SentenceTransformer(model_name)  # using HF wrapper instead
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": False}
    model = HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )
    vectorstore = FAISS.from_texts(
        texts=chunks, embedding=model
    )  # FAISS will run the encode function from sentence transformer object creating the vectors
    return vectorstore


def get_similarity_from_hf_API(chunks):
    API_URL = "https://api-inference.huggingface.co/models/thenlper/gte-small"
    headers = {"Authorization": f"Bearer {hf_api_key}"}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    output = query(
        {
            "inputs": {
                "source_sentence": "question about the PDF",
                "sentences": ["chunk1", "chunk2", "chunk3"],
            },
        }
    )


def get_convchain(vectorstore, llm_choice):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    repo_id = llm_choice
    llm = HuggingFaceHub(
        repo_id=repo_id,
        model_kwargs={"temperature": 0.5},
    )
    conv_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )
    return conv_chain


def handle_question(user_question):
    response = st.session_state.conv_chain({"question": user_question})
    st.session_state.chat_history = response["chat_history"]
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(
                unsafe_allow_html=True,
            )
        else:
            st.write(
                bot_template.replace("{{place_holder}}", message.content),
                unsafe_allow_html=True,
            )


def main():
    load_dotenv()
    st.set_page_config(
        page_title="interact with your data through LLMs", page_icon=":bird:"
    )
    st.write(css, unsafe_allow_html=True)

    if "conv_chain" not in st.session_state:
        st.session_state.conv_chain = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("interact with your data through LLMs :bird:")
    llm_choice = st.selectbox(
        "select which LLM to use:", ("google/flan-t5-xxl", "gpt2")
    )
    user_question = st.text_input("ask your questions here")
    st.write(user_template.replace("{{place_holder}}", "Human"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{place_holder}}", "Bot"), unsafe_allow_html=True)

    if user_question:
        handle_question(user_question)

    with st.sidebar:
        st.subheader("your docs")
        pdf_docs = st.file_uploader("upload your pdfs", accept_multiple_files=True)

        if st.button("process"):
            with st.spinner("in progress..."):
                text_docs = get_pdf_text(pdf_docs)

                chunks = get_text_chuncks(text_docs)
                st.write(chunks)

                vectorstore = get_vectors_from_sentence_transformer(chunks)

                st.session_state.conv_chain = get_convchain(vectorstore, llm_choice)

    hide_st_style = """
        <style>
        #MainMenu {visibility:hidden;}
        footer {visibility:hidden;}
        </style>
    """
    st.markdown(hide_st_style, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
