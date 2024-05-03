import streamlit as st
import os
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Milvus
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.llms import VLLMOpenAI
from langchain.chat_models import ChatOpenAI
import pandas as pd
import hmac
import streamlit as st

from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler

load_dotenv()

import os
from collections.abc import Generator
from queue import Empty, Queue
from threading import Thread
from typing import Optional, List, Dict, Any


############################
# some protection #
############################
def check_password():
    """Returns `True` if the user had the correct password."""

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
    st.stop()  # Do not continue if check_password is not True.

#######################################################
# Making everything configurable from env vars and UI #
#######################################################

# Get default values of environment variables
model_url = os.environ.get("INFERENCE_SERVER_URL", "http://llm-port.stefanb-llm-test.svc.cluster.local:80/v1")
llm_model_name = os.environ.get("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2")
MILVUS_HOST = os.environ.get("MILVUS_HOST", "vectordb-milvus")
MILVUS_PORT = os.environ.get("MILVUS_PORT", "19530")
MILVUS_USERNAME = os.environ.get("MILVUS_USERNAME", "root")
MILVUS_PASSWORD = os.environ.get("MILVUS_PASSWORD", "Milvus")
MILVUS_COLLECTION = os.environ.get("MILVUS_COLLECTION", "redhat_notes")
MAX_TOKENS = os.environ.get("MAX_TOKENS", "2048")
TOP_P = os.environ.get("TOP_P", "0.95")
TEMPERATURE = os.environ.get("TEMPERATURE", "0.01")
PRESENCE_PENALTY = os.environ.get("PRESENCE_PENALTY", "1.03")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "EMPTY")
MAX_RETRIEVED_DOCS = os.environ.get("MAX_RETRIEVED_DOCS", "5")
APP_NAME = os.environ.get("APP_NAME", "OpenShift")
PROMPT_TEMPLATE = os.environ.get("PROMPT_TEMPLATE", """
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

st.title(APP_NAME + 'Bot')

#####################################
# create dropdowns for url and model selection #
#####################################
model_url_names = {
    "http://vllm:8000/v1":['mistralai/Mistral-7B-Instruct-v0.2','meta-llama/Meta-Llama-3-8B-Instruct'] + ["Another model name..."],
    "https://api.openai.com/v1":['gpt-4-turbo','gpt-4-32k'] + ["Another model name..."],
    "http://ollama:11434":['mistral','llama2','llama3'] + ["Another model name..."],
    "Another option":[],
}

# adding "select" as the first and default choice
model_url = st.sidebar.selectbox('Select Model Endpoint', options=['select']+list(model_url_names.keys()))
# display selectbox 2 if model_url is not "select"
if model_url != 'select':
    if model_url != 'Another option':
        llm_model_name = st.sidebar.selectbox('Select Model Name', options=model_url_names[model_url])
        if llm_model_name == 'Another model name...':
            llm_model_name = st.sidebar.text_input("Enter your model name...")
    else:
        model_url = st.sidebar.text_input("Enter your model endpoint...")
        llm_model_name = st.sidebar.text_input("Enter your model name...")

#######################################
# create dropdown for Knowledge Bases #
#######################################
collections = ["redhat_notes","snemeis_notes"]+ ["Another collection name..."]
MILVUS_COLLECTION = st.sidebar.selectbox("Knowledge Collection", collections)

# Create text input for custom entry
if MILVUS_COLLECTION == "Another collection name...": 
    MILVUS_COLLECTION = st.sidebar.text_input("Enter your collection name...")

MILVUS_HOST = st.sidebar.text_input('Milvus Host', MILVUS_HOST)
MILVUS_PORT = st.sidebar.text_input('Milvus Port', MILVUS_PORT)
MILVUS_USERNAME = st.sidebar.text_input('Milvus Username', MILVUS_USERNAME)
MILVUS_PASSWORD = st.sidebar.text_input('Milvus Password', MILVUS_PASSWORD)
MAX_TOKENS = st.sidebar.text_input('Max Tokens (only vllm)', MAX_TOKENS)
TOP_P = st.sidebar.text_input('Top P (only vllm)', TOP_P)
TEMPERATURE = st.sidebar.text_input('Temperature (only vllm)', TEMPERATURE)
PRESENCE_PENALTY = st.sidebar.text_input('Presence Penalty (only vllm)', PRESENCE_PENALTY)
OPENAI_API_KEY = st.sidebar.text_input('OpenAI API Key (only vllm)', OPENAI_API_KEY)
MAX_RETRIEVED_DOCS = st.sidebar.text_input('Amount of Knowledge Base matches to feed', MAX_RETRIEVED_DOCS)

############################
# define a prompt template #
############################

PROMPT_TEMPLATE = st.sidebar.text_area("Prompt Template", PROMPT_TEMPLATE)

prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=PROMPT_TEMPLATE,
    )

# Initialize embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1",
    model_kwargs={'trust_remote_code': True},
    show_progress=False
)

############################
# Streaming call functions #
############################
class QueueCallback(BaseCallbackHandler):
    """Callback handler for streaming LLM responses to a queue."""

    def __init__(self, q):
        self.q = q

    def on_llm_new_token(self, token: str, **kwargs: any) -> None:
        self.q.put(token)

    def on_llm_end(self, *args, **kwargs: any) -> None:
        return self.q.empty()

def remove_duplicates(input_list):
    unique_list_doctitle = []
    unique_list_doccontent = []
    for item in input_list:
        if item.metadata['source'] not in unique_list_doctitle:
            unique_list_doctitle.append(item.metadata['source'])
            unique_list_doccontent.append(item.page_content)
    return unique_list_doctitle, unique_list_doccontent

def stream(input_text, selected_collection, model_endpoint) -> Generator:
    # A Queue is needed for Streaming implementation
    q = Queue()

    # Instantiate LLM
    if model_endpoint == "http://vllm:8000/v1" or model_endpoint == "http://llm-port.stefanb-llm-test.svc.cluster.local:80/v1":
        st.success('GPU', icon="✅")
        llm = VLLMOpenAI(
            openai_api_key=OPENAI_API_KEY,
            openai_api_base=model_url,
            model_name=llm_model_name,
            max_tokens=int(MAX_TOKENS),
            top_p=float(TOP_P),
            temperature=float(TEMPERATURE),
            presence_penalty=float(PRESENCE_PENALTY),
            streaming=True,
            verbose=False,
            callbacks=[QueueCallback(q)]
        )
    elif model_endpoint == "http://ollama:11434":
        llm = Ollama(
            base_url=model_url,
            model=model_name,
            verbose=False,
            callbacks=[QueueCallback(q)]
            )
        st.warning('No GPU', icon="⚠️")
    elif model_endpoint == "https://api.openai.com/v1":
        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
          # openai_api_base=model_url,
            model_name=llm_model_name,
            max_tokens=int(MAX_TOKENS),
            top_p=float(TOP_P),
            temperature=float(TEMPERATURE),
            presence_penalty=float(PRESENCE_PENALTY),
            streaming=True,
            verbose=False,
            callbacks=[QueueCallback(q)]
            )    
    else:
        st.warning('Model URL unknown - trying to initialize via vLLM', icon="⚠️")
        llm = VLLMOpenAI(
            openai_api_key=OPENAI_API_KEY,
            openai_api_base=model_url,
            model_name=llm_model_name,
            max_tokens=int(MAX_TOKENS),
            top_p=float(TOP_P),
            temperature=float(TEMPERATURE),
            presence_penalty=float(PRESENCE_PENALTY),
            streaming=True,
            verbose=False,
            callbacks=[QueueCallback(q)]
        )
    
    store = Milvus(
        embedding_function=embeddings,
        connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT, "user": MILVUS_USERNAME, "password": MILVUS_PASSWORD},
        collection_name=selected_collection,
        metadata_field="metadata",
        text_field="page_content",
        drop_old=False
        )

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=store.as_retriever(search_type="similarity", search_kwargs={"k": int(MAX_RETRIEVED_DOCS)}),
        chain_type_kwargs={
            "prompt": prompt_template, 
            "verbose": True,},
        return_source_documents=True
    )

    # Create a Queue
    job_done = object()

    # Create a function to call - this will run in a thread
    def task():
        resp = qa_chain.invoke({"query": input_text})
        global sources
        global contents
        sources, contents = remove_duplicates(resp['source_documents'])
        # if len(sources) != 0:
        #     q.put("\n*Sources:* \n")
        #     for source in sources:
        #         q.put("* " + str(source) + "\n")
        #     for content in contents:
        #         q.put("* " + str(content) + "\n")
        q.put(job_done)

    # Create a thread and start the function
    t = Thread(target=task)
    t.start()

    content = ""

    # Get each new token from the queue and yield for our generator
    while True:
        try:
            next_token = q.get(True, timeout=1)
            if next_token is job_done:
                break
            if isinstance(next_token, str):
                content += next_token
                yield next_token, content
        except Empty:
            continue

def ask_llm(message, selected_collection, model_endpoint):
    for next_token, content in stream(message, selected_collection, model_endpoint):
        yield(next_token)

def generate_response(question, collection, model_endpoint):
     st.write_stream(ask_llm(question, collection, model_endpoint))
     with st.expander("Context provided to LLM"):
         df = pd.DataFrame(list(zip(sources, contents)), columns=["DocTitle", "Content"])
         st.write(df)

with st.form('my_form'):
    text = st.text_area('Enter Question:', 'Dear Bot, tell me something about .....')

    submitted = st.form_submit_button('Submit')
    if submitted:
        with st.spinner('Wait for my response...'):
            generate_response(text, MILVUS_COLLECTION, model_url)