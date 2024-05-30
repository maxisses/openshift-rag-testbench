import os
import hmac
from queue import Empty, Queue
from threading import Thread
from collections.abc import Generator
from typing import Optional, List, Dict, Any
import requests
import time

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Milvus
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.llms import VLLMOpenAI
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from pymilvus import utility, connections

load_dotenv()

class QueueCallback(BaseCallbackHandler):
    """Callback handler for streaming LLM responses to a queue."""

    def __init__(self, queue: Queue):
        self.queue = queue

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.queue.put(token)

    def on_llm_end(self, *args: Any, **kwargs: Any) -> None:
        return self.queue.empty()

def check_password():
    """Returns `True` if the user has entered the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False

if not check_password():
    st.stop()  # Do not continue if check_password is not True. """

def get_vllm_models(model_url):
    """Fetch model names from VLLM."""
    url = model_url +'/models'
    response = requests.get(url, headers={'accept': 'application/json'})
    if response.status_code == 200:
        data = response.json()
        return [model['id'] for model in data['data']]
    else:
        print(f"Failed to retrieve models from VLLM: {response.status_code}")
        return None

def get_ollama_models(model_url):
    """Fetch model names from Ollama."""
    url = model_url + '/api/tags'
    response = requests.get(url, headers={'accept': 'application/json'})
    if response.status_code == 200:
        data = response.json()
        return [model['name'] for model in data['models']]
    else:
        print(f"Failed to retrieve models from Ollama: {response.status_code}")
        return None

def get_openai_models(model_url, api_key):
    """Fetch model names from OpenAI."""
    url = model_url + '/models'
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        return [model['id'] for model in data['data']]
    else:
        print(f"Failed to retrieve models from OpenAI: {response.status_code}")
        return None

@st.cache_data
def list_milvus_collections(milvus_host: str, milvus_port: str, milvus_username: str, milvus_password: str):
    connections.connect(
        host=milvus_host,
        port=milvus_port,
        user=milvus_username,
        password=milvus_password
    )

    collections = utility.list_collections()
    return collections

# Get default values of environment variables
milvus_host = os.environ.get("MILVUS_HOST", "vectordb-milvus")
milvus_port = os.environ.get("MILVUS_PORT", "19530")
milvus_username = os.environ.get("MILVUS_USERNAME", "root")
milvus_password = os.environ.get("MILVUS_PASSWORD", "Milvus")
max_tokens = os.environ.get("MAX_TOKENS", "2048")
top_p = os.environ.get("TOP_P", "0.95")
temperature = os.environ.get("TEMPERATURE", "0.01")
presence_penalty = os.environ.get("PRESENCE_PENALTY", "1.03")
openai_api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
max_retrieved_docs = os.environ.get("MAX_RETRIEVED_DOCS", "5")
app_name = os.environ.get("APP_NAME", "OpenShift")
prompt_template_str = os.environ.get("PROMPT_TEMPLATE", """
<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant answering questions named HatBot .
You will be given a question you need to answer, and sometimes a context to provide you with information. You must answer the question based as much as possible on this context if it exists.
Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something is not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

Context: 
{context}

Question: {question} [/INST]
""")

st.title(f'{app_name}Bot')

# create dropdowns for url and model selection
model_endpoints = {
    "vLLM": os.getenv("VLLM_ENDPOINT", "http://vllm:8000/v1"),
    "Ollama": os.getenv("OLLAMA_ENDPOINT", "http://ollama:11434"),
    "OpenAI": os.getenv("OPENAI_ENDPOINT", "https://api.openai.com/v1"),
     "Custom Endpoint": "Custom Model Endpoint"
}

model_url_names = list(model_endpoints.keys())
model_url_key = st.sidebar.selectbox('Select Model Endpoint', options=['select'] + model_url_names)

if model_url_key != 'select':
    if model_url_key != 'Custom Endpoint':
        model_url = model_endpoints[model_url_key]
        if model_url_key == "vLLM":
            try:
                llm_model_name = get_vllm_models(model_url)[0]
                llm_model_name = st.sidebar.text_input("Enter your model name...", value=llm_model_name)
            except Exception as e:
                st.error(f"An error occurred while retrieving Models from {model_url}: {e}")
        elif model_url_key == "Ollama":
            try:
                retrieved_models = get_ollama_models(model_url)
                llm_model_name = st.sidebar.selectbox('Select Model', options=['select'] + retrieved_models)
            except Exception as e:
                st.error(f"An error occurred while etrieving Models from {model_url}: {e}")
        elif model_url_key == "OpenAI":
            # API Key if OpenAI
            openai_api_key = st.sidebar.text_input('OpenAI API Key', openai_api_key)
            if openai_api_key.startswith("sk"):
                try:
                    retrieved_models = get_openai_models(model_url, openai_api_key)
                    llm_model_name = st.sidebar.selectbox('Select Model', options=['select'] + retrieved_models)
                except Exception as e:
                    st.error(f"An error occurred while etrieving Models from {model_url}, check your API Key: {e}")
            else:
                st.warning("Please enter a valid OpenAI API key starting with 'sk'.")
    else:
        model_url = st.sidebar.text_input("Enter your model endpoint...")
        model_option = st.sidebar.radio(
            "Help us retrieve available models by selecting the service type:",
            options=["Ollama", "vLLM"]
        )
        if model_option == "vLLM":
            try:
                llm_model_name = get_vllm_models(model_url)[0]
                llm_model_name = st.sidebar.text_input("Enter your model name...", value=llm_model_name)
                model_url_key == "vLLM"
            except Exception as e:
                st.error(f"An error occurred while retrieving vLLM models: {e}")
        elif model_option == "Ollama":
            try:
                retrieved_models = get_ollama_models(model_url)
                llm_model_name = st.sidebar.selectbox('Select Model', options=['select'] + retrieved_models)
                model_url_key == "Ollama"
            except Exception as e:
                st.error(f"An error occurred while retrieving Ollama models: {e}")
else:
    st.warning("Please select a Model Endpoint")


# Milvus Credentials
with st.sidebar.expander("Modify Your Vector Database credentials"):
    milvus_host = st.text_input('Milvus Host', value=milvus_host)
    milvus_port = st.text_input('Milvus Port', value=milvus_port)
    milvus_username = st.text_input('Milvus Username', value=milvus_username)
    milvus_password = st.text_input('Milvus Password', value=milvus_password)

# create dropdown for Knowledge Bases
try:
    milvus_collections = list_milvus_collections(milvus_host, milvus_port, milvus_username, milvus_password)
    milvus_collection = st.sidebar.selectbox("Knowledge Collection", options=['select'] + milvus_collections)
    if milvus_collection == "select":
        st.warning("Please select a Knowledge Base")
except Exception as e:
    st.error(f"Review the Connection Details to the Vector Dabase: {e}")

# RAG parameters
with st.sidebar.expander("Modify Behaviour of RAG"):
    max_retrieved_docs = st.text_input('Amount of Knowledge Base matches to feed', value=max_retrieved_docs)
    prompt_template_str = st.text_area("Prompt Template", value=prompt_template_str)

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template_str,
)

# Parameter Tuning
with st.sidebar.expander("Modify Model Parameters"):
    max_tokens = st.text_input('Max Tokens', value=max_tokens)
    top_p = st.text_input('Top P', value=top_p)
    temperature = st.text_input('Temperature', value=temperature)
    presence_penalty = st.text_input('Presence Penalty', value=presence_penalty)

@st.cache_resource
def load_embedding_model():
    embeddings = HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1",
        model_kwargs={'trust_remote_code': True},
        show_progress=False
    )
    return embeddings

embeddings = load_embedding_model()

def remove_duplicates(input_list):
    unique_doc_titles = []
    unique_doc_contents = []
    for item in input_list:
        if item.metadata['source'] not in unique_doc_titles:
            unique_doc_titles.append(item.metadata['source'])
            unique_doc_contents.append(item.page_content)
    return unique_doc_titles, unique_doc_contents


def stream(input_text: str, selected_collection: str, model_url_key: str) -> Generator:
    queue = Queue()
    if model_option == "vLLM":
        llm = VLLMOpenAI(
            openai_api_key=openai_api_key,
            openai_api_base=model_url,
            model_name=llm_model_name,
            max_tokens=int(max_tokens),
            top_p=float(top_p),
            temperature=float(temperature),
            presence_penalty=float(presence_penalty),
            streaming=True,
            verbose=False,
            callbacks=[QueueCallback(queue)]
        )
        st.success('vLLM runs on a GPU at ' + model_url + " using " + llm_model_name, icon="✅")
    elif model_option == "Ollama":
        llm = Ollama(
            base_url=model_url,
            model=llm_model_name,
            num_predict=int(max_tokens),  # Assuming num_predict corresponds to max_tokens
            top_p=float(top_p),
            temperature=float(temperature),
            repeat_penalty=float(presence_penalty),
            verbose=False,
            callbacks=[QueueCallback(queue)]
        )
        st.success('Ollama can run on GPU & CPU. You are at ' + model_url + " using " + llm_model_name, icon="✅")
    elif model_url_key == "OpenAI":
        llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name=llm_model_name,
            max_tokens=int(max_tokens),
            top_p=float(top_p),
            temperature=float(temperature),
            presence_penalty=float(presence_penalty),
            streaming=True,
            verbose=False,
            callbacks=[QueueCallback(queue)]
        )
        st.success('You are using the OpenAI Cloud Service at ' + model_url + " using " + llm_model_name, icon="✅")
    else:
        if model_url_key == "vLLM":
            llm = VLLMOpenAI(
                openai_api_key=openai_api_key,
                openai_api_base=model_url,
                model_name=llm_model_name,
                max_tokens=int(max_tokens),
                top_p=float(top_p),
                temperature=float(temperature),
                presence_penalty=float(presence_penalty),
                streaming=True,
                verbose=False,
                callbacks=[QueueCallback(queue)]
            )
            st.success('vLLM runs on a GPU at ' + model_url + " using " + llm_model_name, icon="✅")
        elif model_url_key == "Ollama":
            st.success('Ollama can run on GPU & CPU. You are at ' + model_url + " using " + llm_model_name, icon="✅")
            llm = Ollama(
                base_url=model_url,
                model=llm_model_name,
                num_predict=int(max_tokens),  # Assuming num_predict corresponds to max_tokens
                top_p=float(top_p),
                temperature=float(temperature),
                repeat_penalty=float(presence_penalty),
                verbose=False,
                callbacks=[QueueCallback(queue)]
            )
            st.success('Ollama can run on GPU & CPU. You are at ' + model_url + " using " + llm_model_name, icon="✅")

    store = Milvus(
        embedding_function=embeddings,
        connection_args={"host": milvus_host, "port": milvus_port, "user": milvus_username, "password": milvus_password},
        collection_name=selected_collection,
        metadata_field="metadata",
        text_field="page_content",
        drop_old=False
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=store.as_retriever(search_type="similarity", search_kwargs={"k": int(max_retrieved_docs)}),
        chain_type_kwargs={
            "prompt": prompt_template, 
            "verbose": True,
        },
        return_source_documents=True
    )

    job_done = object()

    def task():
        response = qa_chain.invoke({"query": input_text})
        global sources, contents
        sources, contents = remove_duplicates(response['source_documents'])
        queue.put(job_done)

    thread = Thread(target=task)
    thread.start()

    content = ""

    while True:
        try:
            next_token = queue.get(True, timeout=1)
            if next_token is job_done:
                break
            if isinstance(next_token, str):
                content += next_token
                yield next_token, content
        except Empty:
            continue

def ask_llm(message: str, selected_collection: str, model_url_key: str):
    for next_token, content in stream(message, selected_collection, model_url_key):
        yield next_token

def generate_response(question: str, collection: str, model_url_key: str):
    st.write_stream(ask_llm(question, collection, model_url_key))
    with st.expander("Context provided to LLM"):
        df = pd.DataFrame(list(zip(sources, contents)), columns=["DocTitle", "Content"])
        st.write(df)

with st.form('my_form'):
    text = st.text_area('Enter Question:', 'Dear Bot, tell me something about ........')

    submitted = st.form_submit_button('Submit')
    if submitted:
        with st.spinner('Wait for my response...'):
            generate_response(text, milvus_collection, model_url_key)