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
user_icon="icons/person.png"
bot_icon="icons/bot.png"
def open_chat(session_id: str):
    try:
        chat_with_history = MongoDBChatMessageHistory(
            session_id=session_id,
            connection_string=connection_string,
            database_name="nutrition_buddy",
            collection_name="history"
        )
        
        for message in chat_with_history.messages:
            if isinstance(message, HumanMessage):
                with st.chat_message("user",avatar=user_icon):
                    st.markdown(message.content)
            else:
                with st.chat_message("ai",avatar=bot_icon):
                    st.markdown(message.content)

        if "current_response" not in st.session_state:
            st.session_state.current_response = ""



    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return chat_with_history


# Set up the page
st.set_page_config(page_title="Nutrition buddy ??!")
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
db = client['nutrition_buddy']  # Replace 'nutrition_buddy' with your actual database name
users_collection = db['users']

connection_string=uri
globaluser=""
# Check if session_state is not present, create it
if 'page' not in st.session_state:
    st.session_state.page = 'welcome'  # Start on the welcome page
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'current_user' not in st.session_state:
    st.session_state.current_user = {'name': '', 'username': ''}

# Functions for page navigation
def go_to_login():
    st.session_state.page = 'login'

def go_to_signup():
    st.session_state.page = 'signup'

def go_to_nutrition_buddy():
    st.session_state.page = 'nutrition_buddy'

def logout():
    st.session_state.logged_in = False
    st.session_state.current_user = {'name': '', 'username': ''}
    go_to_login()  # Return to the login page
    st.rerun()  # Reload the page after logging out

# Welcome page
if st.session_state.page == 'welcome':
    st.title('Welcome to Nutrition buddy!')
    st.write("Please choose an option:")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button('Login'):
            go_to_login()
            st.rerun()  # Reload the page immediately after clicking

    with col2:
        if st.button('Sign Up'):
            go_to_signup()
            st.rerun()  # Reload the page immediately after clicking

# Login page
elif st.session_state.page == 'login':
    st.title('Login Page')
    username = st.text_input("Username")
    globaluser=username
    password = st.text_input("Password", type='password')
    
    if st.button('Login'):
        # Query to check if the user exists in the database
        account = users_collection.find_one({'username': username, 'password': password})
        
        if account:
            st.success("Logged in successfully!")
            st.session_state.logged_in = True
            st.session_state.current_user = {'name': account['name'], 'username': account['username']}
            go_to_nutrition_buddy()  # Go to Nutrition buddy
            st.rerun()  # Reload the page after logging in
        else:
            st.error("User or Password is wrong")

# Sign Up page
elif st.session_state.page == 'signup':
    st.title('Sign Up Page')
    name = st.text_input("Name")  # Input for user's name
    new_username = st.text_input("New Username (must contain '@')")
    new_password = st.text_input("New Password (at least 4 characters)", type='password')
    confirm_password = st.text_input("Confirm Password", type='password')
    globaluser=new_username
    if st.button('Sign Up'):
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
                # Insert user data into the database
                users_collection.insert_one({'name': name, 'username': new_username, 'password': new_password})
                st.success("Account created successfully!")
                st.session_state.logged_in = True
                st.session_state.current_user = {'name': name, 'username': new_username}
                go_to_nutrition_buddy()  # Go to Nutrition buddy after signing up
                st.rerun()  
            except pymongo.errors.DuplicateKeyError:
                st.error("Username already exists. Please choose a different one.")
# Nutrition buddy page after successful sign up or login
if st.session_state.page == 'nutrition_buddy' and st.session_state.logged_in:
    # Initialize the Nutrition buddy page
    st.header("Nutrition buddy 🤓!")
    # Check if open_chat returns anything and initialize chat_with_history
    chat_with_history = open_chat(globaluser)
    if chat_with_history is None:
        chat_with_history = MongoDBChatMessageHistory(
            session_id=globaluser,
            connection_string=uri,
            database_name="nutrition_buddy",
            collection_name="history"
        )

    # Handle user input and display chat messages
    if user_prompt := st.chat_input("Your message here", key="user_input"):
        with st.chat_message("user",avatar=user_icon):
            st.markdown(user_prompt)
        
        assistant_response = rag_chain.invoke({
            "input": user_prompt,
            "chat_history": chat_with_history.messages
        })['answer']
        
        chat_with_history.add_user_message(user_prompt)
        chat_with_history.add_ai_message(assistant_response)
        
        with  st.chat_message("assistant",avatar=bot_icon):
            st.write(assistant_response)

    # Sidebar with user info and logout button
    with st.sidebar:
        st.write(f"Welcome, {st.session_state.current_user['name']}!")
        if st.button("Show User Info"):
            st.write(f"Username: {st.session_state.current_user['username']}")
            st.write(f"Name: {st.session_state.current_user['name']}")
        
        # Logout button
        if st.button("Logout"):
            logout()  # Execute logout process

    # Initialize session state for current response if not already
    if "current_response" not in st.session_state:
        st.session_state.current_response = ""

    # Display previous messages


    
client.close()
