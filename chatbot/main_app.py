import streamlit as st
import pymongo
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import yaml
from yaml.loader import SafeLoader
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from rag_mongo_v2 import return_rag_chain

user_icon = "icons/person.png"
bot_icon = "icons/bot.png"

def load_CSS():
    with open('static/style.css') as f:
        css = f"<style>{f.read()}</style>"
        st.markdown(css, unsafe_allow_html=True)
    

def open_chat(session_id: str):
    try:
        chat_with_history = MongoDBChatMessageHistory(
            session_id=session_id,
            connection_string=connection_string,
            database_name="nutrition_buddy",
            collection_name="history"
        )
        
        for message in chat_with_history.messages:
            div = f"""
            <div class="chat-row 
                {'' if isinstance(message, AIMessage) else 'row-reverse'}">
                <img class="chat-icon" src="app/static/{
                    'bot.png' if isinstance(message, AIMessage) 
                              else 'person.png'
                }"
                width="32" height="32"/>
                <div class="chat-bubble
                {'ai-bubble' if isinstance(message, AIMessage) else 'human-bubble'}">
                &#8203;{message.content}
                </div>
            </div>
            """
            st.markdown(div, unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return chat_with_history

    return chat_with_history

def load_sessions(username):
    sessions = db["sessions"].find({"username": username})
    return [session["session_id"] for session in sessions]

def create_new_session(username):
    session_id = f"{username}_session_{len(load_sessions(username)) + 1}"
    db["sessions"].insert_one({"username": username, "session_id": session_id})
    return session_id

# Set up the page
st.set_page_config(page_title="Nutrition buddy 🤓!")
load_CSS()
st.session_state.username = ""

@st.cache_resource
def get_rag_chain():
    return return_rag_chain()

rag_chain = get_rag_chain()

# Load MongoDB configuration from YAML
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# MongoDB connection
uri = f"mongodb+srv://{config['mongodb']['user']}:{config['mongodb']['password']}@cluster0.1t2os.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = pymongo.MongoClient(uri)
db = client['test']
users_collection = db['users']

connection_string = uri

# Check if session_state is not present, create it
if 'page' not in st.session_state:
    st.session_state.page = 'welcome'
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'current_user' not in st.session_state:
    st.session_state.current_user = {'name': '', 'username': ''}
if 'current_session' not in st.session_state:
    st.session_state.current_session = None
if 'sessions_list' not in st.session_state:
    st.session_state.sessions_list = []

# Functions for page navigation
def go_to_login():
    st.session_state.page = 'login'

def go_to_signup():
    st.session_state.page = 'signup'

def go_to_home():
    st.session_state.page = 'welcome'

def go_to_nutrition_buddy():
    st.session_state.page = 'nutrition_buddy'

def logout():
    st.session_state.logged_in = False
    st.session_state.current_user = {'name': '', 'username': ''}
    go_to_login()

# Welcome page
if st.session_state.page == 'welcome':
    st.title('Welcome to Nutrition buddy!')
    st.write("Please choose an option:")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button('Login', key="login_button"):
            go_to_login()  
            st.rerun()
    with col2:
        if st.button('Sign Up', key="signup_button"):
            go_to_signup()  
            st.rerun()

# Login page
elif st.session_state.page == 'login':
    st.title('Login Page')
    username = st.text_input("Username")
    password = st.text_input("Password", type='password')
    
    col1, col2 = st.columns([1,.25])  

    with col1:
        if st.button('Login', key="login_submit_button"):
            account = users_collection.find_one({'username': username, 'password': password})
            
            if account:
                st.success("Logged in successfully!")
                st.session_state.logged_in = True
                st.session_state.current_user = {'name': account['name'], 'username': account['username']}
                st.session_state.sessions_list = load_sessions(username)
                if len(st.session_state.sessions_list) == 0:
                    new_session_id = create_new_session(username)
                    st.session_state.sessions_list.append(new_session_id)
                    st.session_state.current_session = new_session_id
                else:
                    # Set the current session to the first available one
                    st.session_state.current_session = st.session_state.sessions_list[0]
                go_to_nutrition_buddy()  
                st.rerun()

            else:
                st.error("User or Password is wrong")

    with col2:
        if st.button("Back to Home", key="back_to_home_button"):
            go_to_home()  
            st.rerun()

# Sign Up page
elif st.session_state.page == 'signup':
    st.title('Sign Up Page')
    name = st.text_input("Name")
    new_username = st.text_input("New Username (must contain '@')")
    new_password = st.text_input("New Password (at least 4 characters)", type='password')
    confirm_password = st.text_input("Confirm Password", type='password')
    
    col1, col2 = st.columns([1,.25])  

    with col1:
        if st.button('Sign Up', key="signup_submit_button"):
            if "@" not in new_username:
                st.error("Username must contain '@'.")
            elif len(new_password) < 4:
                st.error("Password must be at least 4 characters long.")
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            elif name.strip() == "":
                st.error("Name cannot be empty.")
            else:
                try:
                    users_collection.insert_one({'name': name, 'username': new_username, 'password': new_password})
                    st.success("Account created successfully!")
                    st.session_state.logged_in = True
                    st.session_state.current_user = {'name': name, 'username': new_username}
                
                    if len(st.session_state.sessions_list) == 0:
                        new_session_id = create_new_session(new_username)
                        st.session_state.sessions_list.append(new_session_id)
                        st.session_state.current_session = new_session_id
                    else:
                        # Set the current session to the first available one
                        st.session_state.current_session = st.session_state.sessions_list[0]
                    go_to_nutrition_buddy()  
                    st.rerun()

                except pymongo.errors.DuplicateKeyError:
                    st.error("Username already exists. Please choose a different one.")

    with col2:
        if st.button("Back to Home", key="back_to_home_signup_button"):
            go_to_home()  
            st.rerun()

# Nutrition buddy page after successful sign up or login
if st.session_state.page == 'nutrition_buddy' and st.session_state.logged_in:
    st.header("Nutrition buddy 🤓!")
    user = st.session_state.current_user['username']

    # Sidebar with sessions list and new session creation
    with st.sidebar:
        st.write(f"Welcome, {st.session_state.current_user['name']}!")
        if st.button("Show User Info", key="show_user_info_button"):
            st.write(f"Username: {st.session_state.current_user['username']}")
            st.write(f"Name: {st.session_state.current_user['name']}")

        if st.button("Logout", key="logout_button"):
            logout()  
            st.rerun()

        st.markdown("-----")
        st.write("Session")    
        if st.button("New Session", key="new_session_button"):
            new_session_id = create_new_session(user)
            st.session_state.sessions_list.append(new_session_id)
            st.session_state.current_session = new_session_id
            st.rerun()
        for session in st.session_state.sessions_list:
            if st.button(session, key=f"session_button_{session}"):
                st.session_state.current_session = session
                st.rerun()

        

    if st.session_state.current_session:
        chat_with_history = open_chat(st.session_state.current_session)
        if chat_with_history is None:
            chat_with_history = MongoDBChatMessageHistory(
                session_id=st.session_state.current_session,
                connection_string=uri,
                database_name="test",
                collection_name="history"
            )

        if user_prompt := st.chat_input("Your message here", key="user_input"):
            # Render user message with custom styles
                user_div = f"""
            <div class="chat-row row-reverse">
                <img class="chat-icon" src="app/static/person.png" width="32" height="32"/>
                <div class="chat-bubble human-bubble">
                &#8203;{user_prompt}
                </div>
            </div>
            """
                st.markdown(user_div, unsafe_allow_html=True)

            # Generate assistant's response
                assistant_response = rag_chain.invoke({
                "input": user_prompt,
                "chat_history": chat_with_history.messages
            })['answer']
            
                chat_with_history.add_user_message(user_prompt)
                chat_with_history.add_ai_message(assistant_response)

            # Render assistant message with custom styles
                assistant_div = f"""
            <div class="chat-row">
                <img class="chat-icon" src="app/static/bot.png" width="32" height="32"/>
                <div class="chat-bubble ai-bubble">
                &#8203;{assistant_response}
                </div>
            </div>
            """
                st.markdown(assistant_div, unsafe_allow_html=True)

    if "current_response" not in st.session_state:
        st.session_state.current_response = ""
