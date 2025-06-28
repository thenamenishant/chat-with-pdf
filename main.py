import streamlit as st
import pyttsx3
import speech_recognition as sr
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from utils import load_pdf_and_create_vectorstore
import os

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙️ Listening...")
        audio = r.listen(source)
        try:
            query = r.recognize_google(audio)
            st.success(f"You said: {query}")
            return query
        except:
            st.error("Sorry, could not recognize speech.")
            return None

load_dotenv()

st.set_page_config(page_title="Chat with Your PDF", page_icon="📄")
st.title("📄 Chat with Your PDF using LangChain")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

query = st.text_input("Ask a question about the PDF (or click below to speak):")

if st.button("🎤 Speak Your Question"):
    query = listen()

if query:
    result = qa.run(query)
    st.write("🧠 Answer:", result)
    speak(result)


if uploaded_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF uploaded successfully!")

    vectorstore = load_pdf_and_create_vectorstore("temp.pdf")
    llm = OpenAI(temperature=0)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

    query = st.text_input("Ask a question about the PDF:")
    if query:
        result = qa.run(query)
        st.write("🧠 Answer:", result)
