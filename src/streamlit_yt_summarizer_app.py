'''
To run this app, run the following command in your terminal:
streamlit run src/streamlit_yt_summarizer_app.py

Then input the YouTube video URL and click the "Summarize" button.

Download the summary textfile and mp3 file by clicking on their respective
links generated by the app.
'''
import streamlit as st
import textwrap
from gtts import gTTS
import os
import base64

import re
import os
import textwrap
# from langchain.llms import OpenAI
from langchain.llms import VertexAI
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './aiap-13-ds-7e16bb946970.json'


def youtube_video_url_is_valid(url: str) -> bool:
    pattern = r'^https:\/\/www\.youtube\.com\/watch\?v=([a-zA-Z0-9_-]+)(\&ab_channel=[\w\d]+)?$'
    match = re.match(pattern, url)
    return match is not None


def find_insights(url: str) -> str:
    try:
        loader = YoutubeLoader.from_youtube_url(url, language="en-US")
        transcript = loader.load()
    except Exception as e:
        return f"Error while loading YouTube video and transcript: {e}"
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


def main():
    st.title("Youtube Video Summarizer")

    url = st.text_input("Enter Youtube video URL here")

    if st.button("Summarize"):
        if not youtube_video_url_is_valid(url):
            st.error("Please enter a valid Youtube video URL.")
            return
        with st.spinner("Summarizing..."):
            summary = find_insights(url)

            # Word wrapping
            width = 80
            wrapped_summary = textwrap.fill(summary, width=width)

            # Display the summary
            st.write(wrapped_summary)

            # Save the summary to a text file
            with open(f'summary_english.txt', 'w', encoding='utf-8') as f:
                f.write(wrapped_summary)

            # Convert summary to audio
            myobj = gTTS(text=wrapped_summary, lang='en', slow=False)
            myobj.save("./data/speech/output.mp3")

            # Create download links for the text and audio files
            st.markdown(get_file_download_link(
                'summary_english.txt'), unsafe_allow_html=True)
            st.markdown(get_file_download_link(
                './data/speech/output.mp3'), unsafe_allow_html=True)


def get_file_download_link(filename):
    with open(filename, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href


if __name__ == "__main__":
    main()
