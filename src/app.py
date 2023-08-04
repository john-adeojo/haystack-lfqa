import streamlit as st
import multiprocessing

from data_preprocessing import preprocess_docs, vector_stores
from create_pipeline import make_document_qa_pipeline
from create_ai_agent import create_agent
from utils import SingletonToken
import time


if __name__ == '__main__':

    def preprocess():
        doc_dir = r"C:\Users\johna\anaconda3\envs\lfqa_env\haystack-lfqa\documents"
        
        with st.spinner("Preprocessing docs..."):
            time.sleep(5)
            docs = preprocess_docs(doc_dir) 
        
        with st.spinner("Creating document store..."):
            time.sleep(2)
            document_store = vector_stores(docs)
        
        with st.spinner("Building QA pipeline..."):
            time.sleep(3)
            document_qa = make_document_qa_pipeline(document_store)
                
        return docs, document_store, document_qa
    
    if "docs" not in st.session_state:
        st.session_state["docs"], st.session_state["document_store"], st.session_state["document_qa"] = preprocess()


st.markdown(
    """
    #### Prototype Built by [Data-Centric Solutions](https://www.data-centric-solutions.com/)
    """,
    unsafe_allow_html=True
)

# Side panel for OpenAI token input
st.sidebar.title("Configuration")
API_KEY = st.sidebar.text_input("Enter OpenAI Key", type="password")

# Initialize an empty placeholder
placeholder = st.empty()

if API_KEY:
    SingletonToken.set_token(API_KEY)
    API_KEY = SingletonToken.get_token()

    with st.spinner("Creating agent..."):
        st.session_state["agent"] = create_agent(st.session_state["document_qa"], API_KEY)

    # If OpenAI key and data_url are set, enable the chat interface
    st.title("Ask me about your docs")
    query_user = placeholder.text_input("ask me a question...")
    
    if st.button("Submit"):
        with st.spinner('Agent is working...'):
            result = st.session_state["agent"].run(query_user)
            output = result["transcript"].split("---")[0]
            st.write(output)
else:
    # If OpenAI key and data_url are not set, show a message
    placeholder.markdown(
        """
        **Please enter your OpenAI key and data URL in the sidebar.**
        
        Follow this [link](https://www.howtogeek.com/885918/how-to-get-an-openai-api-key/) to get your OpenAI API key.
        """,
        unsafe_allow_html=True,
    )
