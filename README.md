# AI-Powered Conversational PDF Chatbot with Voice Support

A scalable, AI-driven PDF chatbot that enables users to interact with PDF documents using voice commands. Built with Google Gemini LLM, this application combines natural language understanding, semantic search, and voice input/output to deliver a highly interactive user experience.

---

## Features

- **Conversational PDF Chatbot** – Ask context-aware questions based on uploaded PDFs.
- **Powered by Google Gemini LLM** – Ensures intelligent and accurate responses.
- **Semantic Search with FAISS** – Fast and relevant document retrieval.
- **Voice Interaction** – Integrated speech-to-text and text-to-speech for hands-free interaction.
- **Streamlit Web Interface** – Simple, user-friendly UI for easy interaction.
- **Secure API Management** – Environment variables handled via `dotenv`.

---

## Tech Stack

- **LLM:** Google Gemini  
- **Framework:** LangChain  
- **Vector Store:** FAISS  
- **Web Interface:** Streamlit  
- **PDF Parsing:** PyPDF2  
- **Voice Support:** SpeechRecognition, pyttsx3  
- **Environment Management:** python-dotenv

---

## Architecture Overview

1. **PDF Upload & Parsing:** Users upload a PDF which is processed and parsed using `PyPDF2`.
2. **Embedding & Storage:** Document content is embedded and stored in a FAISS vector database.
3. **Conversational Flow:** LangChain connects the user queries to Google Gemini LLM using context-aware prompting.
4. **Voice Support:** Integrated STT (Speech-to-Text) using `SpeechRecognition` and TTS (Text-to-Speech) using `pyttsx3`.
5. **UI Layer:** Streamlit provides an intuitive web interface for uploading PDFs, asking questions, and hearing responses.

---

## Optimization Highlights

- Optimized Gemini API calls for reduced latency and cost.
- Efficient vector storage with FAISS to handle large documents.
- Scalable design allowing easy integration of new features or LLMs.
