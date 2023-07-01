# https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps
'''
streamlit run streamlit_yt_QA_bot.py
'''
import re
import os
import random
import time
import streamlit as st
import textwrap
from gtts import gTTS
import base64
import textwrap
# from langchain.llms import OpenAI
from langchain.llms import VertexAI
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain.chat_models import ChatVertexAI
from langchain.document_loaders import YoutubeLoader
from langchain.embeddings import VertexAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA


os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './aiap-13-ds-7e16bb946970.json'


def youtube_video_url_is_valid(url: str) -> bool:

    pattern = r'^https:\/\/www\.youtube\.com\/watch\?v=([a-zA-Z0-9_-]+)(\&ab_channel=[\w\d]+)?$'
    match = re.match(pattern, url)
    return match is not None

def obtain_transcript(url: str) -> str:

    try:
        print('url1 is: \n', url)
        loader = YoutubeLoader.from_youtube_url(url, language="en-US")
        return loader.load() # full transcript is return
    except Exception as e:
        return f"Error while loading YouTube video and transcript: {e}"


def summarize(transcript: str) -> str:
    try:
        # llm = OpenAI(temperature=0.6, openai_api_key=api_key)
        llm = VertexAI(temperature=0.3)
        prompt = PromptTemplate(
            template="""Summarize the youtube video whose transcript is provided within backticks \
            ```{text}```
            """, input_variables=["text"]
        )
        combine_prompt = PromptTemplate(
            template="""Combine all the youtube video transcripts  provided within backticks \
            ```{text}```
            Provide a summary between 50-100 sentences.
            """, input_variables=["text"]
        )
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100000, chunk_overlap=50)
        text = text_splitter.split_documents(transcript)
        chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True,
                                     map_prompt=prompt, combine_prompt=combine_prompt)
        answer = chain.run(text)
    except Exception as e:
        return f"Error while processing and summarizing text: {e}"

    return answer.strip()

# Main part of the Streamlit app
def get_file_download_link(filename):

    with open(filename, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

def create_qa_retriever(transcript: str):

    # Initialize text splitter for QA
    text_splitter_qa = TokenTextSplitter(chunk_size=1000, chunk_overlap=200)

    # Split text into docs for QA
    docs_qa = text_splitter_qa.split_documents(transcript)

    # Create the LLM model for the question answering
    llm_question_answer = ChatVertexAI(temperature=0.2)

    # Create the vector database and RetrievalQA Chain
    embeddings = VertexAIEmbeddings()
    db = FAISS.from_documents(docs_qa, embeddings)
    qa = RetrievalQA.from_chain_type(
        llm=llm_question_answer, chain_type="stuff", retriever=db.as_retriever())

    return qa

def qa_bot(qa):

    # Display previous Q&A
    # if len(st.session_state.history) >= 0:
    for q_and_a in st.session_state.history:
        st.markdown(f"**Q:** {q_and_a['question']}")
        st.markdown(f"**A:** {q_and_a['answer']}")

    # Get the user question
    question = st.text_input("Ask a question:")

    if question:
        # Run the QA chain to query the data
        answer = qa.run(question)
        st.session_state.history.append({"question": question, "answer": answer})

        # Display the answer        
        st.markdown(f"**Q:** {question}")
        st.markdown(f"**A:** {answer}")

def main():
    """
    Main function to run the Streamlit application for YouTube video summarization.
    """
    st.title("Youtube Video Summarizer")

    url = st.text_input("Enter Youtube video URL here")

    # Initialize qa in session state if it doesn't exist
    if 'qa' not in st.session_state:
        st.session_state['qa'] = None

    # Initialize qa in session state if it doesn't exist
    if 'summary' not in st.session_state:
        st.session_state['summary'] = None

    if st.button("Summarize"):
        if not youtube_video_url_is_valid(url):
            st.error("Please enter a valid Youtube video URL.")
            return

        # if "history" not in st.session_state:
        st.session_state.history = []

        with st.spinner("Summarizing..."):

            transcript = obtain_transcript(url)
            summary = summarize(transcript)

            # Word wrapping
            width = 80
            wrapped_summary = textwrap.fill(summary, width=width)
            # st.session_state.history.append({"summary": wrapped_summary})
            st.session_state["summary"] = wrapped_summary

            # Display the summary
            st.write(wrapped_summary)
                        
            # qa = create_qa_retriever(transcript)
            st.session_state['qa'] = create_qa_retriever(transcript)
            
            # if qa is not None:
            #     qa_bot(qa)

    if st.session_state.get('summary') is not None:
        st.markdown(st.session_state["summary"])

    if st.session_state.get('qa') is not None:
        qa_bot(st.session_state['qa'])

if __name__ == "__main__":
    main()


