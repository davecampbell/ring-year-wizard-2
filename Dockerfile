FROM continuumio/miniconda3:latest

RUN conda clean --all

RUN conda update -n base -c defaults conda && \
    conda install -c conda-forge -c fastai -y fastai

RUN pip install opencv-python-headless

RUN conda install -c conda-forge openai && \
    conda install python-dotenv && \
    conda clean -afy

SHELL ["/bin/bash", "-c"]

# Set working directory
WORKDIR /app

