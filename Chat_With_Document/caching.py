import spacy
import streamlit as st
import tiktoken
from sentence_transformers import CrossEncoder

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from .config import require_env


@st.cache_resource(show_spinner=False)
def get_tokenizer():
    return tiktoken.get_encoding("cl100k_base")


@st.cache_resource(show_spinner=False)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


@st.cache_resource(show_spinner=False)
def get_llm() -> ChatGroq:
    api_key = require_env("GROQ_API_KEY")
    return ChatGroq(api_key=api_key, model_name="llama-3.1-8b-instant", temperature=0)


@st.cache_resource(show_spinner=False)
def get_reranker() -> CrossEncoder:
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


@st.cache_resource(show_spinner=False)
def get_spacy_nlp():
    # Requires: python -m spacy download en_core_web_sm
    return spacy.load("en_core_web_sm")

