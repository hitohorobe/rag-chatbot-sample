import os
from typing import Any, Optional

import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSerializable
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# botのロール表記
USER = "User"
ASSISTANT = "Assistant"

# prompt templete
message = ""
message += "あなたはTwitter社のカスタマーサポートです。"
message += "取得したコンテキストの次の部分を使用して質問に答えます。"
message += "答えがわからない場合は、わからないと言ってください。"
message += "最大 3 つの文を使用し、回答は簡潔にしてください。\n"
message += "質問: {question}\n"
message += "コンテキスト: {context}\n"
message += "答え：\n"


def init_llm(api_key: str) -> ChatOpenAI:
    # llm
    llm = ChatOpenAI(
        api_key=api_key, # type: ignore
        model="gpt-4",
        temperature=0.5,
    )
    return llm


def make_rag_chain(llm: ChatOpenAI, openai_api_key: str) -> Optional[RunnableSerializable[Any, Any]]:
    index_name = os.getenv("PINECONE_INDEX_NAME")
    if not index_name:
        raise ValueError("PINECONE_INDEX_NAME is not set")

    vector_store = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=OpenAIEmbeddings(
            api_key=openai_api_key, # type: ignore
            model="text-embedding-3-small"
        ),
    )
    if vector_store is None:
        return
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 5}
    )
    prompt = ChatPromptTemplate.from_messages(["human", message])
    rag_chain: RunnableSerializable[Any, Any] = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm
    return rag_chain


def main():
    #  streamlitのサイドバーにopenai apiキーの入力欄を表示
    openai_api_key = None

    st.title("Twitter 利用規約のRAG Chatbot")

    with st.sidebar:
        openai_api_key = st.sidebar.text_input("OpenAI API Keyをここに入力", type="password")
        "[Twitter利用規約](https://cdn.cms-twdigitalassets.com/content/dam/legal-twitter/site-assets/privacy-policy-new/pp-tos-ja.pdf)"
        "Pinecone + OpenAIのRAGアプリのデモです。Twitter利用規約に関する質問に回答します。"
    # セッションの初期化
    if "chat_log" not in st.session_state:
        st.session_state.chat_log = []

    # 質問を入力
    user_msg = st.chat_input("質問を入力してください")

    if user_msg:
        # チャットログの表示
        for chst in st.session_state.chat_log:
            with st.chat_message(chst["name"]):
                st.write(chst["message"])
        
        with st.chat_message(USER):
            st.write(user_msg)

        with st.spinner("AIが回答を生成中です..."):
            # rag chatbotの回答を生成
            if not openai_api_key:
                st.write("OpenAI API Keyを入力してください")
                return
            try:
                llm = init_llm(openai_api_key)
                rag_chain = make_rag_chain(llm, openai_api_key)
                response = rag_chain.invoke(user_msg)
            except Exception as e:
                st.write(f"エラーが発生しました: {e}")
                return
        with st.chat_message(ASSISTANT):
            assistant_msg = ""
            assistant_msg_area = st.empty()
            assistant_msg += response.content
            assistant_msg_area.write(assistant_msg)

        # チャットログに追加
        st.session_state.chat_log.append({"name": USER, "message": user_msg})
        st.session_state.chat_log.append({"name": ASSISTANT, "message": assistant_msg})


if __name__ == "__main__":
    main()
