FROM continuumio/miniconda3:latest

RUN conda install fastai::fastai && \
    conda install openai && \
    conda install python-dotenv && \
    conda clean -afy

SHELL ["/bin/bash", "-c"]

# Set working directory
WORKDIR /app

