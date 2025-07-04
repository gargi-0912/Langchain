import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="💻 Offline LangChain Chatbot", page_icon="🤖")

# ----------------------------
# Title and Instructions
# ----------------------------
st.markdown("""
# 🤖 Offline LangChain Chatbot
Powered locally using **Ollama** and **LangChain**.
""")

# ----------------------------
# Sidebar Settings
# ----------------------------
with st.sidebar:
    st.header("Settings ⚙️")
    selected_model = st.selectbox("Choose LLM model:", ["mistral", "llama3", "phi3", "gemma"], index=0)
    st.markdown("Make sure the model is pulled using `ollama pull <model_name>` before use.")

# ----------------------------
# LangChain Setup
# ----------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to user queries."),
    ("user", "Question: {question}")
])
llm = ChatOllama(model=selected_model)
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# ----------------------------
# User Input
# ----------------------------
input_text = st.text_input("Ask your question:", placeholder="e.g. What is LangChain?")

# ----------------------------
# Response
# ----------------------------
if input_text:
    with st.spinner("Generating response..."):
        try:
            result = chain.invoke({"question": input_text})
            st.success("Response:")
            st.write(result)
        except Exception as e:
            st.error(f"❌ Error: {e}")
