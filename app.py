import os
from model import Document
from common.reader import pdf_reader, pdf_reader_pyu
from common.parser import sentence_splitter
from common.llm import LlmConfig, LocalLLM
from common.storage import get_pg_storage_context, get_local_file_storage_context
from llama_index.core import VectorStoreIndex

import streamlit as st
llm_config =  LlmConfig(LocalLLM.LM_STUDIO)

def read_documents_to_list(file_path):
    import json
    try:
        with open(file_path) as f:
            data = json.load(f)
            return [Document(**doc) for doc in data]
    except:
        return []

documents = read_documents_to_list("document.json")
    
def ui_sidebar():
    if st.sidebar.button("upload new documents"):
        ui_sidebar_upload_files()

    if len(documents) == 0:
        st.sidebar.write("no avaiable documents now.")
    else:
        options=[doc.file_name for doc in documents if doc.file_name]
        st.sidebar.write("Available documents:")
        for option in options:
            item  = st.sidebar.checkbox(f'{option}')
            if item:
                doc  = get_document_index_id(option, documents)
                if doc is None:
                    st.toast(f"{item} not found")
                else:
                    ui_main(doc=doc)

def ui_main(doc : Document):
    st.write(f"正在对{doc.file_name}进行对话")
    query = st.text_input(f'请输入要提问的内容')
    if query:
        chat(doc.file_name, query)

@st.dialog("上传新文件")
def ui_sidebar_upload_files():
    
    uploaded_files = st.file_uploader("选择文件", accept_multiple_files=True)
    finish_upload = st.button('FINISH UPLOAD')
    if finish_upload and uploaded_files:
        saved_file_names = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join("files", uploaded_file.name)
            # print(file_path)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_file_names.append(uploaded_file.name)
            hit = "正在处理文件"
            with st.spinner(hit):
                for doc in saved_file_names:
                    hit = f'正在处理: {doc}'
                    index_id = index_file(doc)
                    print(index_id)
                    document = Document(file_name=doc, index_id=index_id)
                    documents.append(document)
                    print(document)
                    save_documents_to_json(documents=documents)
                st.write("文件处理完毕")
                st.rerun()


def save_documents_to_json(documents, file_path="document.json"):
    import json
    try:
        with open(file_path, "w") as f:
            json.dump([doc.__dict__ for doc in documents], f)
    except Exception as e:
        print(f"Error saving documents: {e}")


def index_file(file_name) -> str: 
    storage_context = get_local_file_storage_context(file_name)
    llm_config = LlmConfig(LocalLLM.LM_STUDIO)
    file = f'./files/{file_name}'
    print(file)
    docs = pdf_reader_pyu(file)
    print(f'docs: {len(docs)}')
    nodes = sentence_splitter(docs)
    print(f'node: {len(nodes)}')
    vector_index = VectorStoreIndex(nodes=nodes, storage_context=storage_context, 
                                    embed_model=llm_config.embedding)
    storage_context.persist()
    print(vector_index.index_id)
    return vector_index.index_id


def get_document_index_id(file_name, documents):
    """Get index_id for a document by file_name from documents list"""
    for doc in documents:
        if doc.file_name == file_name:
            return doc
    return None

def chat(index_id, query):
    with st.spinner("thinking..."):
        storage_context = get_local_file_storage_context(index_id)
        vector_index = VectorStoreIndex.from_vector_store(storage_context.vector_store, index_id=index_id)
        query_engine = vector_index.as_chat_engine(llm=llm_config.llm)
        resp = query_engine.chat(query)
        st.write(resp.response)


                
def main():
    st.title("Personal Knowledge Management.")
    ui_sidebar()

if __name__ == "__main__":
    main()