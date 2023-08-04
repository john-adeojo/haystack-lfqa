import streamlit as st
import multiprocessing

from data_preprocessing import preprocess_docs, vector_stores
from create_pipeline import make_document_qa_pipeline
from create_ai_agent import create_agent
from utils import SingletonToken
import time


if __name__ == '__main__':

    multiprocessing.freeze_support()

    # Precursor processes
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
    # docs = preprocess_docs(doc_dir)
    # document_store = vector_stores(docs)
    # document_qa = make_document_qa_pipeline(document_store)

st.markdown(
    """
    #### Prototype Built by [Data-Centric Solutions](https://www.data-centric-solutions.com/)
    """,
    unsafe_allow_html=True
)

# # Get docs from local store
# # with st.spinner('Loading Docs...'):
# doc_dir = r"C:\Users\johna\anaconda3\envs\lfqa_env\haystack-lfqa\documents"
# docs = preprocess_docs(doc_dir)

# # Store docs in FAISS vector db
# # with st.spinner('Storing Data...'):
# document_store = vector_stores(docs)

# # Make document QA pipeline 
# # with st.spinner('Building pipelines...'):
# document_qa = make_document_qa_pipeline(document_store)

# # Create agent
# # with st.spinner('Creating agent...'):
# agent = create_agent(document_qa, API_KEY)


# Side panel for OpenAI token input
st.sidebar.title("Configuration")
API_KEY = st.sidebar.text_input("Enter OpenAI Key", type="password")
doc_dir = st.sidebar.text_input("Enter directory to document store", type="password")
# Initialize an empty placeholder
placeholder = st.empty()

if API_KEY:
    SingletonToken.set_token(API_KEY)
    API_KEY = SingletonToken.get_token()

    # Create agent
    with st.spinner("Building QA pipeline..."):
        time.sleep(3)
        agent = create_agent(document_qa, API_KEY)

    # If OpenAI key and data_url are set, enable the chat interface
    st.title("Ask me about your docs")
    query_user = placeholder.text_input("ask me a question...")
    
    if st.button("Submit"):
        # with st.spinner('Agent is working...'):
        result = agent.run(query_user)
        output = result["transcript"].split("---")[0]
        st.write(output)  # or st.markdown(output)
        # st.markdown(f"Here's your suggested Journey: : {response}")

else:
    # If OpenAI key and data_url are not set, show a message
    placeholder.markdown(
        """
        **Please enter your OpenAI key and data URL in the sidebar.**
        
        Follow this [link](https://www.howtogeek.com/885918/how-to-get-an-openai-api-key/) to get your OpenAI API key.
        """,
        unsafe_allow_html=True,
    )

# if __name__ == '__main__':
#     freeze_support()
    
#     # Code to start processes
#     p = multiprocessing.Process(...) 
#     p.start()