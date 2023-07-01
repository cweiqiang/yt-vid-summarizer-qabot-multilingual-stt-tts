To start using the python packages for this repository


```bash 
conda create -n vertex-ai-sdk-env python==3.10.11
```

```bash
conda activate vertex-ai-sdk-env
```

```bash
pip install -r requirements.txt
```

To run the Streamlit app for the Youtube transcript text summarizer and text-to-speech convertor, use the following command

```bash
streamlit run src/streamlit_yt_summarizer_app.py
```

Input a Youtube url such as the Genshin Impact version 3.8 Special Program
`https://www.youtube.com/watch?v=jlivBvu3Jrc`

to output

1. a text file containing the summary
2. a audio file corresponding to the summary text file.

To run the QA_bot.py script, we can run

```bash
streamlit run src/streamlit_yt_QA_bot.py
```
However, we need to first replace one of the python scripts in line 798

`~/anaconda3/envs/vertexai-sdk-env/lib/python3.10/site-packages/vertexai/language_models/_language_models.py`


        ```python
        response_obj = TextGenerationResponse(
            text=prediction["candidates"][0]["content"]
            if prediction.get("candidates")
            else None,
            _prediction_response=prediction_response,
            
            is_blocked=safety_attributes[0].get("blocked", False),
            safety_attributes=dict(
                zip(
                   safety_attributes[0].get("categories", []),
                    safety_attributes[0].get("scores", []),
                )
            #
            #is_blocked=safety_attributes.get("blocked", False),
            #safety_attributes=dict(
            #    zip(
            #        safety_attributes.get("categories", []),
            #        safety_attributes.get("scores", []),
            #    )
            ),
        )
        response_text = response_obj.text
        ```
        
Alternatively, we can use a docker container to run the streamlit apps


```bash
docker build -t streamlit_yt_qa_bot_container:1.0.0 .
```
Warning! This docker build can take more than 10 minutes! To save time, the user can consider using pulling the docker container from Dockerhub

```bash
docker pull cweiqiang/streamlit_yt_qa_bot_container:1.0.0
```

```bash
docker run -p 8501:8501 -v ./aiap-13-ds-7e16bb946970.json:/usr/src/app/aiap-13-ds-7e16bb946970.json streamlit_yt_qa_bot_container:1.0.0
```

Remarks: 

(1) For debugging purposes, the user interact with the container with bash commands via:

```bash
docker run -it --entrypoint /bin/bash streamlit_yt_qa_bot_container:1.0.0
```

(2) If the user want to runs `streamlit_yt_summarizer_app.py` instead when the container launcher, the last line in Dockerfile can be replaced by

```dockerfile
ENTRYPOINT ["streamlit", "run", "src/streamlit_yt_summarizer_app.py"]
``` 
