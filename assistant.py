import streamlit as st
import openai
from langchain.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    UnstructuredHTMLLoader,
    UnstructuredExcelLoader,
)
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI

def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    return loader, pages

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks

def get_vectorstore(text_chunks, openai_keys):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_keys, model='text-embedding-ada-002')
    vector_store = FAISS.from_documents(documents=text_chunks, embedding=embeddings)
    return vector_store

class ConversationalChain:
    def __init__(self, vector_store, openai_keys):
        # Store initial parameters in instance variables
        self.vector_store = vector_store
        self.openai_keys = openai_keys
        self.model = self._initialize_model()
        self.memory = self._initialize_memory()
        self.qa_prompt = self._initialize_prompt_template()
        
    def _initialize_model(self):
        # Initialize the ChatOpenAI model
        return ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.3,
            openai_api_key=self.openai_keys
        )

    def _initialize_memory(self):
        # Initialize memory
        return ConversationBufferMemory(
            memory_key='chat_history', 
            return_messages=True
        )

    def _initialize_prompt_template(self):
        # Method to set up the prompt template
        template = """
        As a highly competent AI assistant, your role is to use the document provided as a guide, 
        Your responses should be grounded in the context contained within these documents.
        For each user query provided in the form of chat, apply the following guidelines:
        - If the answer is within the document's context, provide a detailed and precise response.
        - If the answer is not available based on the given context, clearly state that you don't 
        have the information to answer.
        - If the query falls outside the scope of the context, politely clarify that your expertise 
        is limited to answering questions directly related to the provided documents.

        When responding, prioritize accuracy and relevance:
        Context:
        {context}

        Question:
        {question}
        """

        system_message_prompt = SystemMessagePromptTemplate.from_template(template=template)
        human_template = "{question}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        return ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    def create_qa_chain(self):
        # Create and return the conversational retrieval chain
        return ConversationalRetrievalChain.from_llm(
            llm=self.model, 
            retriever=self.vector_store.as_retriever(), 
            memory=self.memory, 
            combine_docs_chain_kwargs={'prompt': self.qa_prompt}
        )



def initialize_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Como posso te ajudar?"}]
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None

def load_qa_chain(saved_files_info, openai_keys):
    docs_splits = split_documents(saved_files_info)
    vectordb = get_vectorstore(docs_splits, openai_keys)
    CC = ConversationalChain(vectordb, openai_keys)
    return CC