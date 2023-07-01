# Use an official Python runtime as a parent image
FROM python:3.10.11

# Set the working directory in the container
WORKDIR /usr/src/app

# Install Anaconda
RUN apt-get update && \
    apt-get install -y wget && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/anaconda3 && \
    rm -f Miniconda3-latest-Linux-x86_64.sh && \
    echo ". $HOME/anaconda3/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# Make RUN commands use the new environment
SHELL ["/bin/bash", "-c"]

# Create a new conda environment
RUN /root/anaconda3/bin/conda create -n vertexai-sdk-env python=3.10.11 -y

# Activate conda environment
RUN echo "conda activate vertexai-sdk-env" >> ~/.bashrc

# Make RUN commands use the new environment
ENV PATH /root/anaconda3/envs/vertexai-sdk-env/bin:$PATH

# Copy the current directory contents into the container at /app
COPY . /usr/src/app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy _language_models.py to the specified location
# RUN mkdir -p ~/anaconda3/envs/vertexai-sdk-env/lib/python3.10/site-packages/vertexai/language_models/
COPY src/_language_models.py /root/anaconda3/envs/vertexai-sdk-env/lib/python3.10/site-packages/vertexai/language_models/_language_models.py

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Run streamlit_yt_QA_bot.py when the container launches
ENTRYPOINT ["streamlit", "run", "src/streamlit_yt_QA_bot.py"]
