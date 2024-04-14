import os

from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings

import streamlit as st

from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma

from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo)
from dotenv import load_dotenv
import os

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv("API_KEY")

llm = OpenAI(temperature=0.1, verbose=True)
embeddings = OpenAIEmbeddings()

with st.sidebar:
    st.title("RAG app")
    choice = st.radio("Navigation", ["Upload File","Get Answer"])
    st.info("This application is designed to allow ChatGPT or other LLMs to answer highly targeted questions using your files!")

if choice == "Upload File":
    st.title("Upload File")

    name = st.text_input("enter a name of project")

    desc = st.text_input("enter a description of project")

    if not name:
        name = "default"

    if not desc:
        desc = "default"



    file = st.text_input("Enter path or url to file") # st.file_uploader("Upload a file", type=["pdf"])


    if st.button("Create Project"):
        if file is not None:
            # file_path = os.path.join(file.temporary_folder, file.name)
            
            loader = PyPDFLoader(file)
            pages = loader.load_and_split()
            store = Chroma.from_documents(pages, embeddings, collection_name=name)

            vectorstoreinfo = VectorStoreInfo(vectorstore=store,name=name, description=desc)
            
            toolkit = VectorStoreToolkit(vectorstore_info=vectorstoreinfo)
            st.session_state.agent = create_vectorstore_agent(llm, toolkit, verbose=True)
            
            st.write("Success!")

if choice == "Get Answer":
    prompt = st.text_input('Input your prompt here')

    if prompt and st.button("Get Answer"):
        response = st.session_state.agent.run(prompt)

        st.write(response)

        with st.expander('Document Similarity Search'):
            search = store.similarity_search_with_score(prompt)
            if search:
                st.write(search[0][0].page_content) 