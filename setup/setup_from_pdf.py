import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from pinecone import ServerlessSpec
from pinecone import Pinecone


load_dotenv()


def read_pdf(path: str) -> list[Document]:
    loader = PyPDFLoader(path)
    pages = loader.load_and_split()
    return pages


def create_pinecone_index(api_key: str, index_name: str) -> None:
    pinecone_client = Pinecone(api_key=api_key)

    if index_name not in pinecone_client.list_indexes().names():
        pinecone_client.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )


def store_vector(
    md_header_splits: list[Document],
    index_name: str,
    namespace: str,
    embedding: OpenAIEmbeddings,
) -> None:
    try:
        print("Storing vectors...")
        PineconeVectorStore.from_documents(
            documents=md_header_splits,
            index_name=index_name,
            namespace=namespace,
            embedding=embedding,
        )
        time.sleep(1)  # wait for the vectors to be stored
        return
    except Exception as e:
        print(f"Error storing vectors: {e}")
        return


def main():
    # 環境変数からAPIキーをとる
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # PDFファイルを読み込んで分割する
    pdf_path = input("PDFファイルのパスを入力してください: ")
    assert Path(pdf_path).exists(), f"{pdf_path} does not exist"

    md_header_splits = read_pdf(pdf_path)

    # OpenAIのEmbeddingsを初期化
    embedding = OpenAIEmbeddings(api_key=openai_api_key, model="text-embedding-3-small")

    # Pineconeのインデックスが存在しない場合は作成
    index_name = os.getenv("PINECONE_INDEX_NAME")
    create_pinecone_index(api_key=pinecone_api_key, index_name=index_name)

    # ドキュメントをPineconeに保存
    namespace = "rag-demo-app"
    store_vector(
        md_header_splits=md_header_splits,
        index_name=index_name,
        namespace=namespace,
        embedding=embedding,
    )


if __name__ == "__main__":
    main()
