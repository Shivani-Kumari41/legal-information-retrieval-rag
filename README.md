# AI-Powered Legal Information Retrieval System

Privacy-preserving RAG pipeline for semantic legal document retrieval.

## Project Overview
Built an end-to-end Retrieval-Augmented Generation (RAG) pipeline for context-aware legal question answering using LangChain and FAISS.

## Tech Stack
- Python, LangChain
- FAISS (Vector Database)
- HuggingFace Sentence Transformers
- RAG (Retrieval Augmented Generation)

## Features
- Document chunking and preprocessing
- Semantic embeddings using all-MiniLM-L6-v2
- FAISS vector store for fast similarity search
- Context-aware legal query retrieval
- Covers IPC sections, RTI Act, Consumer Protection Act

## How to Run
pip install langchain langchain-community faiss-cpu sentence-transformers
python legal_chatbot.py

## Results
- Successfully retrieves relevant legal sections for user queries
- Example: Query "What is punishment for murder?" correctly retrieves Section 302 IPC
