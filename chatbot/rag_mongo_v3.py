# %%
# %%
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from rag_mongo_v2 import return_rag_chain
st.set_page_config(page_title="Nutrition Buddy", page_icon=":nut:", layout="wide")

@st.cache_resource
def get_rag_chain():
    return return_rag_chain()

rag_chain = get_rag_chain()
# %%
connection_string = "mongodb+srv://hossam6ht:fJduz2FCWxUw44tk@cluster0.tuiqy.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# %%


# %%

if "pages" not in st.session_state:
    st.session_state.pages = ["Chat Session 1", "Chat Session 2"]

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

if open_chat(page) is not None:
    chat_with_history=open_chat(page)
else:
    chat_with_history = MongoDBChatMessageHistory(
        session_id=page,
        connection_string=connection_string,
        database_name="langchain",
        collection_name="history"
    )
    

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
    