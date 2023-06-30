To start using the python packages for this repository


$ conda create -n vertex-ai-sdk-env python==3.10.11`

$ conda activate vertex-ai-sdk-env

$ pip install -r requirements.txt`

To run the Streamlit app for the Youtube transcript text summarizer and text-to-speech convertor, use the following command

$ streamlit run src/streamlit_yt_summarizer_app.py

Input a Youtube url such as 
https://www.youtube.com/watch?v=jlivBvu3Jrc

to output

1. a text file containing the summary
2. a audio file corresponding to the summary text file
