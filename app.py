import os
from model import Document
from common.reader import pdf_marker_reader
from common.parser import sentence_splitter
from common.llm import LlmConfig, LocalLLM
from common.storage import get_pg_storage_context
from llama_index.core import VectorStoreIndex

import streamlit as st


config = LlmConfig(LocalLLM.LM_STUDIO)
storage_context = get_pg_storage_context("qwen", 4096)


def read_documents_to_list(file_path):
    import json
    try:
        with open(file_path) as f:
            data = json.load(f)
            return [Document(**doc) for doc in data]
    except:
        return []

def save_documents_to_json(documents, file_path="document.json"):
    import json
    try:
        with open(file_path, "w") as f:
            json.dump([doc.__dict__ for doc in documents], f)
    except Exception as e:
        print(f"Error saving documents: {e}")


def index_file(file_name) -> str: 
    docs = pdf_marker_reader(file_name=file_name)
    nodes = sentence_splitter(docs)
   
    vector_index = VectorStoreIndex(nodes=nodes, storage_context=storage_context)
    storage_context.persist()
    return vector_index.index_id

def get_document_index_id(file_name, documents):
    """Get index_id for a document by file_name from documents list"""
    for doc in documents:
        if doc.file_name == file_name:
            return doc.index_id
    return None

def chat(index_id, query):
    pg_vector_index = VectorStoreIndex.from_vector_store(storage_context.vector_store, index_id=index_id)
    query_engine = pg_vector_index.as_query_engine(llm=config.llm)
    resp = query_engine.query(query)
    st.write(resp)

# Read document.txt file
documents = read_documents_to_list("document.json")
task_type = st.sidebar.selectbox("选择一项任务", [ "提问", "上传", ])

if task_type == "对话":
    if len(documents) == 0:
        st.write("no files found.")
    else:
        st.header("请选择左侧已完成的文件")
        # Display txt content as radio options in sidebar
        if documents:
            selected_file = st.sidebar.radio(
                "选择文件",
                 options=[doc.file_name for doc in documents if doc.file_name],
                key="selected_file"
            )
            index_id = get_document_index_id(selected_file)
            if index_id is None:
                st.write(f"未找到文件{selected_file}的索引数据")
            else:
                st.write(f"正在对{selected_file}进行对话")
                query = st.text_input(f'请输入要提问的内容')
                if query:
                    chat(index_id, query)


if task_type == '上传':
    st.title("上传新文档")
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
            st.toast(f"You have uploaded {uploaded_file.name}")

            hit = "正在处理文件"
            with st.spinner(hit):
                for doc in saved_file_names:
                    hit = f'正在处理: {doc}'
                    index_id = index_file(doc)
                    document = Document(file_name=doc, index_id=index_id)
                    documents.append(document)
                    save_documents_to_json(documents=documents)
                st.write("文件处理完毕")