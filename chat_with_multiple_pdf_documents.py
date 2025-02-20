import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

import speech_recognition as sr
import pyttsx3

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to capture speech input
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Please speak your question.")
        try:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            st.error("Sorry, could not understand the audio.")
        except sr.RequestError:
            st.error("Error with the speech recognition service.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    return ""


# Function to get AI-generated answer
def get_answer(user_question):
    try:
        response = user_input(user_question)
        return response.text
    except Exception as e:
        st.error(f"Error in getting AI response: {e}")
        return ""


# Initialize text-to-speech engine
def initialize_tts():
    try:
        engine = pyttsx3.init()
        return engine
    except Exception as e:
        st.error(f"Text-to-speech initialization error: {e}")
        return None


# Speak the answer
def speak_answer(text):
    try:
        engine = initialize_tts()
        if engine:
            engine.say(text)
            engine.runAndWait()
    except Exception as e:
        st.error(f"Speech error: {e}")


def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    print(chunks)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain.invoke(
        {"input_documents":docs, "question": user_question})
    print(response)
    st.write("Reply: ", response["output_text"])


# Streamlit app
def main():
    st.title("Voice-Powered AI PDF ChatBot")
    st.header("Voice-Powered AI PDF ChatBot using Gemini💁")

    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'spoken_text' not in st.session_state:
        st.session_state.spoken_text = ""

    # Speech input column
    col1, col2 = st.columns([4, 1])
    with col1:
        # Use st.text_input with a key and default value
        user_query = st.text_input(
            "Ask a question:",
            value=st.session_state.get('user_query', ''),
            key='query_input'
        )
    with col2:
        # Speech recognition button
        if st.button("🎙️", help="Click to speak your question"):
            recognized_text = recognize_speech()
            if recognized_text:
                st.session_state.spoken_text = recognized_text
                st.session_state.user_query = recognized_text
                st.rerun()

    # Button to get answer
    if st.button("Get Answer"):
        if user_query:
            answer = get_answer(user_query)
            if answer:
                st.session_state.chat_history.append((user_query, answer))
                speak_answer(answer)
    # Display chat history
    st.write("### Chat History:")
    for question, answer in st.session_state.chat_history:
        st.write(f"*You:* {question}")
        st.write(f"*AI:* {answer}")
        st.write("---")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()