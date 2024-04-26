FROM registry.access.redhat.com/ubi9/python-311@sha256:78aa5022cb9e28652b5923162fee2e6e6192df84166abb5c7438910f70f9d02e

WORKDIR /app

## sentence-transformer CPU only version
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install transformers tqdm numpy scikit-learn scipy nltk sentencepiece
RUN pip3 install --no-deps sentence-transformers

## install the rest
COPY requirements.txt /app/requirements.txt
RUN pip3 install -r requirements.txt

COPY ./.streamlit /app/.streamlit
COPY rag-bot.py /app/rag-bot.py

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "rag-bot.py", "--server.port=8501", "--server.address=0.0.0.0"]