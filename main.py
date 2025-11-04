# __import__("pysqlite3")  # Ensure pysqlite3 is imported for ChromaDB
# import sys

# sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA


# 제목
st.title("ChatPDF")
st.write("---")

# 파일 업로드
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
st.write("---")


def pdf_to_document(uploaded_file):
    # 'with' 문을 사용하여 임시 디렉터리를 안전하게 생성하고 자동 삭제
    with tempfile.TemporaryDirectory() as temp_dir:
        tmp_file_path = os.path.join(temp_dir, uploaded_file.name)

        # 업로드된 파일의 내용을 임시 파일에 쓴다 (binary write mode 'wb')
        with open(tmp_file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        # PyPDFLoader로 임시 파일 로드
        loader = PyPDFLoader(tmp_file_path)
        pages = loader.load_and_split()
        return pages


# 업로드 되면 실행
if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)

    # --- RAG 로직 (Split, Embed, Store) ---
    # 이 모든 로직은 'if uploaded_file is not None:' 블록 내부에 있어야 합니다.

    # Split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    texts = text_splitter.split_documents(pages)

    # Embedding
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

    # load it into Chroma
    vectordb = Chroma.from_documents(documents=texts, embedding=embeddings_model)

    # --- RAG 로직 끝 ---

    # Question (파일이 업로드되었을 때만 질문 입력창이 나타나도록 함)
    st.header("PDF에 대해 질문해보세요!")
    question = st.text_input("질문을 입력하세요:")

    if st.button("질문하기"):
        # LLM 및 QA Chain도 버튼 클릭 시 생성
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectordb.as_retriever(),
        )
        result = qa_chain.invoke({"query": question})
        st.write("답변:")
        st.write(result["result"])  # 결과 딕셔너리에서 'result' 값만 깔끔하게 출력
