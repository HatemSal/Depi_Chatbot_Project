# %%
# %%
import streamlit as st

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from typing import Sequence
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory

from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict


# %%

 
 
# %%

# load the local model 
#paraphrase-MiniLM-L6-v2 choose better one
lc_embed_model = HuggingFaceEmbeddings(
    model_name="local_model_para/"
)

# %%
def load_and_process_pdfs(pdf_folder_path):
    documents = []
    for file in os.listdir(pdf_folder_path):
        if file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder_path, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    return splits

splits = load_and_process_pdfs("nuitrions")

# %%
def initialize_vectorstore(splits):
    return FAISS.from_documents(documents=splits, embedding=lc_embed_model)

vectorstore = initialize_vectorstore(splits)


# %%

# %%
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
llm = ChatOllama(model="llama3.1")
 
question_answer_chain = create_stuff_documents_chain(llm, prompt)
retriever = vectorstore.as_retriever()

 

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)


qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)



# %%

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

 

# %%
connection_string = "mongodb+srv://hossam6ht:fJduz2FCWxUw44tk@cluster0.tuiqy.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# %%


# %%
 
if "pages" not in st.session_state:
    st.session_state.pages = ["Chat Session 1", "Chat Session 2"]

# Set page config
st.set_page_config(page_title="Nutrition Buddy", page_icon=":nut:", layout="wide")
# Header for the app
st.header("Nutrition buddy 🤓!")

# Sidebar for session navigation
st.sidebar.title("Choose your session")
if st.sidebar.button("Add new session"):
    st.session_state.currentsession="new session"

    # Append new session to the session_state list
    st.session_state.pages.append(f"Chat Session {len(st.session_state.pages) + 1}")

# Dropdown to select a session
page = st.sidebar.selectbox("Go to", st.session_state.pages)

def open_chat(session_id: str):
    try:
        chat_with_history = MongoDBChatMessageHistory(
            session_id=session_id,
            connection_string=connection_string,
            database_name="langchain",
            collection_name="history"
        )
        
        for message in chat_with_history.messages:
            if isinstance(message, HumanMessage):
                with st.chat_message("user",avatar="👤"):
                    st.markdown(message.content)
            else:
                with st.chat_message("assistant",avatar="🤖"):
                    st.markdown(message.content)

        if "current_response" not in st.session_state:
            st.session_state.current_response = ""



    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return chat_with_history

 
chat_with_history=open_chat(page)

if user_prompt := st.chat_input("Your message here", key="user_input"):
    with st.chat_message("user"):
        st.markdown(user_prompt)
    assistant_response = rag_chain.invoke({
        "input": user_prompt,
        "chat_history": chat_with_history.messages
    })['answer']
    
    chat_with_history.add_user_message(user_prompt)
    chat_with_history.add_ai_message(assistant_response)
    with st.chat_message("assistant"):
        st.write(assistant_response)
    