# https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps
'''
completed tasks:
1. implement langchain query
2. implement translation
3. implement audio output
4. implement video embedding

remaining tasks:
1. implement audio input
2. fix the error in _last.mp3
3. position the question input box at the bottom
4. add docstring and type hints
5. modularize the qa_bot() function


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
from translation import Translation
import translators as ts

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './aiap-13-ds-7e16bb946970.json'

lang_dict = {
            'English' : 'en',
            'Bahasa Melayu' : 'ms',
            'Chinese (Simplified)' : 'zh-CN'
        }

def youtube_video_url_is_valid(url: str) -> bool:

    # pattern = r'^https:\/\/www\.youtube\.com\/watch\?v=([a-zA-Z0-9_-]+)(\&ab_channel=[\w\d]+)?$'
    pattern = r'^https:\/\/www\.youtube\.com\/watch\?v=([a-zA-Z0-9_-]+)(\&[a-zA-Z0-9_]+=[\w\d]+)*$'

    match = re.match(pattern, url)
    return match is not None

def obtain_transcript(url: str) -> str:

    try:
        loader = YoutubeLoader.from_youtube_url(url, language="en-US")
        return loader.load() # full transcript is return
    except Exception as e:
        return f"Error while loading YouTube video and transcript: {e}"


def summarize(transcript: str) -> str:
    try:
        # llm = OpenAI(temperature=0.6, openai_api_key=api_key)
        llm = VertexAI(temperature=0.3, max_output_tokens = 512)
        prompt = PromptTemplate(
            template="""Summarize the youtube video whose transcript is provided within backticks \
            ```{text}```
            """, input_variables=["text"]
        )
        combine_prompt = PromptTemplate(

            template="""Combine all the youtube video transcripts  provided within backticks \
            ```{text}```
            Provide a summary within 400 words.
            """, input_variables=["text"]
        )
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100000, chunk_overlap=50)
        text = text_splitter.split_documents(transcript)

        chain = load_summarize_chain(llm, chain_type="map_reduce",
                                map_prompt=prompt, combine_prompt=combine_prompt)
        answer = chain.run(text)
    except Exception as e:
        return f"Error while processing and summarizing text: {e}"

    st.session_state["summary_en"] = answer.strip()

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

def qa_bot(qa, language):

    # Display previous Q&A
    for i, q_and_a in enumerate(st.session_state.history):
        if language != 'English':

            q_trans = ts.translate_text(query_text = q_and_a['q_en'],
                                        from_language='en',
                                        to_language=lang_dict[language],
                                        translator = 'google')      
                    
            a_trans = ts.translate_text(query_text = q_and_a['a_en'],
                                        from_language='en',
                                        to_language=lang_dict[language],
                                        translator = 'google')  

            # if not already translated to the same language before
            if st.session_state.history[i].get(f'q_{lang_dict[language]}') is None:
                st.session_state.history[i][f'q_{lang_dict[language]}'] = q_trans
                st.session_state.history[i][f'a_{lang_dict[language]}'] = a_trans
            
            if not os.path.exists(f"q_audio_{lang_dict[language]}_{i}.mp3"): 
                q_audio = gTTS(text=q_and_a[f'q_{lang_dict[language]}'], lang=lang_dict[language], slow=False)
                a_audio = gTTS(text=q_and_a[f'a_{lang_dict[language]}'], lang=lang_dict[language], slow=False)
                q_audio.save(f"q_audio_{lang_dict[language]}_{i}.mp3")              
                a_audio.save(f"a_audio_{lang_dict[language]}_{i}.mp3")           
               
            st.markdown(f"**Q_translate:** {q_and_a[f'q_{lang_dict[language]}']}")
            st.audio(f"q_audio_{lang_dict[language]}_{i}.mp3", format='audio/mp3')

            st.markdown(f"**A_translate:** {q_and_a[f'a_{lang_dict[language]}']}")
            st.audio(f"a_audio_{lang_dict[language]}_{i}.mp3", format='audio/mp3')
        
        # everything in English
        else:
            if not os.path.exists(f"q_audio_{lang_dict[language]}_{i}.mp3"): 
                q_audio = gTTS(text=q_and_a['q_en'], lang='en', slow=False)
                a_audio = gTTS(text=q_and_a['a_en'], lang='en', slow=False)
                q_audio.save(f"q_audio_{lang_dict[language]}_{i}.mp3")              
                a_audio.save(f"a_audio_{lang_dict[language]}_{i}.mp3")

            st.markdown(f"**Q:** {q_and_a['q_en']}")
            st.audio(f"q_audio_{lang_dict[language]}_{i}.mp3", format='audio/mp3')

            st.markdown(f"**A:** {q_and_a['a_en']}")
            st.audio(f"a_audio_{lang_dict[language]}_{i}.mp3", format='audio/mp3')

    # Get the new question
    question = st.text_input("Ask a question:")

    # generate answer for the user's question
    if question:
        # Run the QA chain to query the data
        if language != 'English':
            q_trans = question
            q_en = ts.translate_text(query_text = q_trans,
                                        from_language=lang_dict[language],
                                        to_language='en',
                                        translator = 'google')  
            q_audio = gTTS(text=q_trans, lang=lang_dict[language], slow=False)
            q_audio.save(f"q_audio_{lang_dict[language]}_last.mp3")              
            st.markdown(f"**Q_translate:** {q_trans}")
            st.audio(f"q_audio_{lang_dict[language]}_last.mp3", format='audio/mp3')

            a_en = st.session_state['qa'].run(q_en)
            a_trans = ts.translate_text(query_text = a_en,
                                        from_language='en',
                                        to_language=lang_dict[language],
                                        translator = 'google')  
            a_audio = gTTS(text=a_trans, lang=lang_dict[language], slow=False)
            a_audio.save(f"a_audio_{lang_dict[language]}_last.mp3")
            st.markdown(f"**A_translate:** {a_trans}")
            st.audio(f"a_audio_{lang_dict[language]}_last.mp3", format='audio/mp3')

            st.session_state.history.append({"q_en": q_en, 
                                             "a_en": a_en,
                                             f"q_{lang_dict[language]}": q_trans,
                                             f"a_{lang_dict[language]}": a_trans
                                             })
            
        # everything in English
        else: 
            q_en = question
            q_audio = gTTS(text=q_en, lang=lang_dict[language], slow=False)
            q_audio.save(f"q_audio_{lang_dict[language]}_last.mp3")   
            st.markdown(f"**Q:** {q_en}")
            st.audio(f"q_audio_{lang_dict[language]}_last.mp3", format='audio/mp3')

            a_en = qa.run(q_en)
            a_audio = gTTS(text=a_en, lang=lang_dict[language], slow=False)
            a_audio.save(f"a_audio_{lang_dict[language]}_last.mp3")
            st.markdown(f"**A:** {a_en}")
            st.audio(f"a_audio_{lang_dict[language]}_last.mp3", format='audio/mp3')

            st.session_state.history.append({"q_en": q_en, "a_en": a_en})

def initialize_lang():
    # Dropdown for language selection
    selected_language  = st.selectbox('Language:', 
                                    options=list(lang_dict.keys()))

    # if button is clicked, show the selectbox
    if selected_language:
        st.session_state["language"] = selected_language
    else:
        st.session_state["language"] = 'English'

    return selected_language

def initialize_others(language):

    # Initialize qa in session state if it doesn't exist
    if 'qa' not in st.session_state:
        st.session_state['qa'] = None

    # Initialize qa in session state if it doesn't exist
    if 'summary_en' not in st.session_state:
        st.session_state['summary_en'] = None

    # Initialize translated summary in session state if it doesn't exist
    if f'summary_{lang_dict[language]}' not in st.session_state:
        st.session_state[f'summary_{lang_dict[language]}'] = None

def translate_summmary():
    language = st.session_state["language"]
    language_code = lang_dict[language]

    summary = st.session_state["summary_en"]

    if language != 'English':                
        translated_summary = ts.translate_text(query_text = summary,
                                                from_language='en',
                                                to_language=language_code,
                                                translator = 'google')   
        st.session_state[f"summary_{language_code}"] = translated_summary

def display_summary():
    language = st.session_state["language"]
    language_code = lang_dict[language]

    if language != 'English':
        st.markdown(st.session_state[f"summary_{language_code}"])

    elif st.session_state.get('summary_en') is not None:
        st.markdown(st.session_state["summary_en"])    

    # audio out summary
    if st.session_state.get(f"summary_{language_code}") is not None:
        summary_audio = gTTS(text=st.session_state[f"summary_{language_code}"], lang=language_code, slow=False)  
        summary_audio.save(f"summary_{language_code}_audio.mp3")
        st.audio(f"summary_{language_code}_audio.mp3", format='audio/mp3')

def main():
    """
    Main function to run the Streamlit application for YouTube video summarization.
    """
    st.title("Game Video Guru")

    url = st.text_input("Enter Youtube video URL here")

    if url != "":
        st.video(url)

    language = initialize_lang()

    if st.button("Summarize"):
        
        initialize_others(language)

        if not youtube_video_url_is_valid(url):
            st.error("Please enter a valid Youtube video URL.")
            return

        st.session_state.history = []

        # complete the code block before 'Summarizing...' Button status change
        with st.spinner("Summarizing..."):

            transcript = obtain_transcript(url)
            summarize(transcript)
                        

            # create qa retriever for subsequent QA session
            st.session_state['qa'] = create_qa_retriever(transcript)

    # Translate the summary
    if language != 'English':
        translate_summmary()

    if st.session_state.get("summary_en") is not None:
        display_summary()

    # execute Q&A session
    if st.session_state.get('qa') is not None:
        qa_bot(st.session_state['qa'], language = st.session_state.get("language"))

if __name__ == "__main__":
    main()


