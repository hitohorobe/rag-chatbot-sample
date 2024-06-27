# rag-chatbot-sample
gpt-4 + pinecone + streamlitで作成したチャットボットのサンプルです  
[デモページ](https://twitter-rag-chatbot.streamlit.app/)

## セットアップ
### アカウント取得
- [Pinecone](https://www.pinecone.io/)のアカウントを作成して、API KEYを取得
- [OpenAI API](https://openai.com/index/openai-api/)のアカウントを作成してAPI KEYを取得

### 環境変数の設定
- `cp .env.sample .env`
- `.env` に Pinecone の API KEY, OpenAIのAPI キーを記載する

### ベクトルデータの投入
- twitter の利用規約PDFをダウンロードして `setup/data` へと格納
- `poetry install`
- `poetry shell`
- `cd setup`
- `python setup_from_pdf.py`
- ダウンロードしたPDFファイルのパスをターミナルに入力
- PDFの内容が分割されてベクトルデータベースに格納される

### RAGの実行
- `cd ../app`
- `streamlit run main.py`
- サイドバーにOpenAIのAPIキーを貼り付ける
- チャットを実施する
