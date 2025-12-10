import streamlit as st
import pandas as pd
from genai_agent import EmployAidAgent
import time



if 'agent' not in st.session_state:
    api_key = None
    print(st.secrets)
    if 'GEMINI_API_KEY' in st.secrets:
        api_key = st.secrets['GEMINI_API_KEY']
        print(f'secrets key: {api_key}')
    agent = EmployAidAgent(api_key=api_key)
    st.session_state.agent = agent

agent = st.session_state.agent

if "messages" not in st.session_state:
    st.session_state.messages = []

 
        
if 'feedback' not in st.session_state:
    st.session_state.feedback = []


st.set_page_config(page_title="EmployAID", layout="wide")
st.title(" 👋 EmployAID")
st.subheader("Navigate Reviews. Empower Decisions.")
st.markdown("""
Welcome to **EmployAID**, your ultimate tool for navigating employer reviews with the power of AI. Whether you're a job seeker looking for insights on potential employers or an HR professional aiming to understand employee sentiments, EmployAID has got you covered.""")

# # Text Alignment
# # Custom CSS to center the title with a specific ID
# title_alignment = """
#     <style>
#     #my-centered-title {
#         text-align: center;
#     }
#     </style>
#     """
# st.markdown(title_alignment, unsafe_allow_html=True)

# # Display the title with the assigned ID
# st.markdown("<h1 id='my-centered-title'>EmployAID</h1>", unsafe_allow_html=True)

# Side Bar
st.sidebar.title("Navigate EmployAID")
st.initial_sidebar_state = ""


with st.sidebar:
    st.write("This tool allows you as a prospective candidate to find companies and/or job roles that suit your preferences.")
    st.subheader("How it works:")
    st.write("""
    - **1. Enter your employer/job-specific question.**
    - **2. The AI agent will analyze the reviews and provide insights based on your query.**
    - **3. Generate Reports and Review the AI-generated insights and summaries.**
    """)
    # EmployAID Company Logo
    st.logo(image="logo-1.png", size="large")
    st.html("""
  <style>
    [alt=Logo] {
      height: 6rem;
    }
  </style>
        """)

# Select the Language Model
st.subheader("Select LLM")
st.write("(**at times queries may get stuck, switching models can help)")
model_option = st.selectbox(
    "Choose a Language Model:",
    options=list(agent.models.keys()),
    index=0
)

if model_option:
    agent.launch_agent(model_name=model_option)

for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message['role'] == 'employaid':
            r = message['feedback']
            if r < 0:
                st.write(f'Rating: Not rated')
            else:
                st.write(f'Rating: {r}')


if prompt := st.chat_input("Enter the name of the company you would like to search:"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    resp_cols = st.columns([1.0, 0.05])
    with resp_cols[0]:
        with st.chat_message("employaid"):
            message_placeholder = st.empty()
            full_response = ""
            # Simulate streaming response

            query_time = time.time()

            with st.spinner("..."):
                response, current_tokens, total_tokens = agent.call_agent(prompt)
                #response, current_tokens, total_tokens = "testing", 100, 100
                message_placeholder.markdown(response)
                
            resp_time = time.time()
            duration = resp_time - query_time
            #st.info(f"Query Process Time: {duration:.2f} seconds**.")
            st.session_state.messages.append({"role": "employaid",
                                              "content": response,
                                              "response_time": duration,
                                              "feedback": -1})
            stat_cols = st.columns(2)
            with stat_cols[0]:
                st.info(f"Query Process Time: {duration:.2f} seconds**.")
            with stat_cols[1]:
                st.info(f'Total Tokens: {total_tokens}, Current Tokens: {current_tokens}')
            
    with resp_cols[1]:
        rate_opts = [1, 2, 3, 4, 5]
        rating = st.radio("⭐?", options=rate_opts, index=None, key='rating_radio')
        if rating:
            st.session_state.messages[-1]['feedback'] = rating
            
            
            
        

        
            
        
        
                                    

        
        

