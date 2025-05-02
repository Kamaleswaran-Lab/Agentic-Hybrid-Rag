FROM python:3.10.1-buster

## DO NOT EDIT these 3 lines.
RUN mkdir /Agentic-Hybrid-Rag
COPY ./ /Agentic-Hybrid-Rag
WORKDIR /Agentic-Hybrid-Rag

## Install your dependencies here using apt install, etc.

## Include the following line if you have a requirements.txt file.
RUN pip install -r requirements.txt

