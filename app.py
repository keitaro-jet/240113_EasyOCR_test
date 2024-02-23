import streamlit as st
import requests
from PIL import Image
import base64
import io

# Streamlit アプリケーションのタイトルとページの設定
st.set_page_config(page_title="自動採点アプリ", page_icon=":pencil2:")

# アプリケーションのタイトルを表示
st.title("自動採点アプリ")

# 画像アップロードのUI
uploaded_file = st.file_uploader("画像ファイルをアップロードしてください", type=['jpg', 'jpeg', 'png'])

# アップロードされた画像を表示
if uploaded_file is not None:
    st.subheader('アップロードされた画像:')
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # アップロードされた画像を FastAPI に送信して解析結果を取得
    if st.button('採点開始'):
        # FastAPI のエンドポイント URL
        # api_endpoint = "http://127.0.0.1:8000/analyze_image/"
        api_endpoint = "https://calculation-scoring.onrender.com/analyze_image/"

        try:
            # POST リクエストを送信して解析結果を取得
            response = requests.post(api_endpoint, files={'file': uploaded_file})
            data = response.json()

            # 画像データを取得し、PILのImageオブジェクトに変換
            image_data = base64.b64decode(data['image'])
            image = Image.open(io.BytesIO(image_data))

            # 解析結果を表示
            st.subheader('採点結果:')
            st.image(image, caption='Analyzed Image', use_column_width=True)

            # 成績と結果詳細を表示
            st.subheader('成績:')
            st.table(data['df2'])
            st.subheader('解析結果詳細:')
            st.table(data['df1'])

        except Exception as e:
            st.error(f"エラーが発生しました: {e}")
