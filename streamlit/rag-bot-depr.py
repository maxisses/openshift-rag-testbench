import streamlit as st
import os
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Milvus
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import VLLMOpenAI
import pandas as pd
import hmac
import streamlit as st


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

# Get default values of environment variables
INFERENCE_SERVER_URL = os.environ.get("INFERENCE_SERVER_URL", "http://vllm:8000/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2")
MILVUS_HOST = os.environ.get("MILVUS_HOST", "vectordb-milvus.milvus.svc.cluster.local")
MILVUS_PORT = os.environ.get("MILVUS_PORT", "19530")
MILVUS_USERNAME = os.environ.get("MILVUS_USERNAME", "root")
MILVUS_PASSWORD = os.environ.get("MILVUS_PASSWORD", "Milvus")
MILVUS_COLLECTION = os.environ.get("MILVUS_COLLECTION", "openshift_redhat_notes")
MAX_TOKENS = os.environ.get("MAX_TOKENS", "2048")
TOP_P = os.environ.get("TOP_P", "0.95")
TEMPERATURE = os.environ.get("TEMPERATURE", "0.01")
PRESENCE_PENALTY = os.environ.get("PRESENCE_PENALTY", "1.03")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "EMPTY")
RAG_NEAREST_NEIGHBOURS = os.environ.get("RAG_NEAREST_NEIGHBOURS", "5")
APP_NAME = os.environ.get("APP_NAME", "Openshift")

st.title(APP_NAME + 'Bot')

INFERENCE_SERVER_URL = st.sidebar.text_input('Inference Server URL', INFERENCE_SERVER_URL)
MODEL_NAME = st.sidebar.text_input('Model Name', MODEL_NAME)
MILVUS_HOST = st.sidebar.text_input('Milvus Host', MILVUS_HOST)
MILVUS_PORT = st.sidebar.text_input('Milvus Port', MILVUS_PORT)
MILVUS_USERNAME = st.sidebar.text_input('Milvus Username', MILVUS_USERNAME)
MILVUS_PASSWORD = st.sidebar.text_input('Milvus Password', MILVUS_PASSWORD)
MILVUS_COLLECTION = st.sidebar.text_input('Milvus Collection', MILVUS_COLLECTION)
MAX_TOKENS = st.sidebar.text_input('Max Tokens (only vllm)', MAX_TOKENS)
TOP_P = st.sidebar.text_input('Top P (only vllm)', TOP_P)
TEMPERATURE = st.sidebar.text_input('Temperature (only vllm)', TEMPERATURE)
PRESENCE_PENALTY = st.sidebar.text_input('Presence Penalty (only vllm)', PRESENCE_PENALTY)
OPENAI_API_KEY = st.sidebar.text_input('OpenAI API Key (only vllm)', OPENAI_API_KEY)
RAG_NEAREST_NEIGHBOURS = st.sidebar.text_input('Amount of Knowledge Base matches to feed', RAG_NEAREST_NEIGHBOURS)

def initialize_chain(model_endpoint):
    embeddings = HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1",
        model_kwargs={'trust_remote_code': True},
        show_progress=False
    )
    
    store = Milvus(
        embedding_function=embeddings,
        connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT, "user": MILVUS_USERNAME, "password": MILVUS_PASSWORD},
        collection_name=MILVUS_COLLECTION,
        metadata_field="metadata",
        text_field="page_content",
        drop_old=False
        )
    
#    template = """[INST] <>\nYou are a helpful, respectful and honest assistant named OpenshiftBot answering questions.\nYou will be given a question you need to answer, and a context about everything a sales team wrote down about its customers to provide you with information. You must answer the question based as much as possible on this context.\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, don't share false information.\n<>\n\nContext: \n{context}\n\nQuestion: {question} [/INST]"""
    template = """<s>[INST] <<SYS>>
                You are a helpful, respectful and honest assistant named OpenshiftBot. You are answering a question.
                You will be given a question you need to answer, and a context about everything a sales team wrote down about its customers to provide you with information. You must answer the question based as much as possible on this context.

                If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, don't share false information.
                <</SYS>>
                
                Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question in a helpful, respectful manner:
                ------
                <ctx>
                {context}
                </ctx>
                ------
                <hs>
                {history}
                </hs>
                ------
                {question}
                Answer:
                """

#    prompt_template = PromptTemplate.from_template(template)
    prompt_template = PromptTemplate(
            input_variables=["history", "context", "question"],
            template=template,
        )

    if model_endpoint == "http://vllm:8000/v1":
        st.success('GPU', icon="✅")
        llm = VLLMOpenAI(
            openai_api_key=OPENAI_API_KEY,
            openai_api_base=INFERENCE_SERVER_URL,
            model_name=MODEL_NAME,
            max_tokens=MAX_TOKENS,
            top_p=TOP_P,
            temperature=TEMPERATURE,
            presence_penalty=PRESENCE_PENALTY,
            streaming=True,
            verbose=False,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
    elif model_endpoint == "http://ollama:11434":
        llm = Ollama(base_url=INFERENCE_SERVER_URL, model=MODEL_NAME)
        st.warning('No GPU', icon="⚠️")
    else:
        st.warning('Model URL unknown - trying to initialize via vLLM', icon="⚠️")
        llm = VLLMOpenAI(
            openai_api_key=OPENAI_API_KEY,
            openai_api_base=INFERENCE_SERVER_URL,
            model_name=MODEL_NAME,
            max_tokens=MAX_TOKENS,
            top_p=TOP_P,
            temperature=TEMPERATURE,
            presence_penalty=PRESENCE_PENALTY,
            streaming=True,
            verbose=False,
            callbacks=[StreamingStdOutCallbackHandler()]
        )

    
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=store.as_retriever(search_type="similarity", search_kwargs={"k": int(RAG_NEAREST_NEIGHBOURS)}),
        chain_type_kwargs={
            "prompt": prompt_template, 
            "verbose": True,
            "memory": ConversationBufferMemory(
            memory_key="history",
            input_key="question"),},
        return_source_documents=True
    )
    
    return qa_chain

qa_chain = initialize_chain(INFERENCE_SERVER_URL)

def remove_duplicates(input_list):
    unique_list_doctitle = []
    unique_list_doccontent = []
    for item in input_list:
        if item.metadata['source'] not in unique_list_doctitle:
            unique_list_doctitle.append(item.metadata['source'])
            unique_list_doccontent.append(item.page_content)
    return unique_list_doctitle, unique_list_doccontent

def generate_response(input_text):
    result = qa_chain.invoke({"query": input_text})
    with st.expander("Context provided to LLM"):
        title, content = remove_duplicates(result['source_documents'])
        df = pd.DataFrame(list(zip(title, content)), columns=["Title", "Content"])
        st.write(df)
    st.write(result["result"])

with st.form('my_form'):
    text = st.text_area('Enter Question:', 'Dear Bot, tell me something about MindSphere and the collaboration with Red Hat / IBM.')

    submitted = st.form_submit_button('Submit')
    if submitted:
        with st.spinner('Wait for my response...(which takes w/o GPU a looong time)'):
            generate_response(text)