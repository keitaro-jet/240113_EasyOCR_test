import streamlit as st
import requests

# Streamlit アプリケーションのタイトル
st.title("自動採点アプリ")

# 画像アップロードのUI
uploaded_file = st.file_uploader("画像ファイルをアップロードしてください", type=['jpg', 'jpeg', 'png'])

# アップロードされた画像を表示
if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # アップロードされた画像を FastAPI に送信して解析結果を取得
    if st.button('解析開始'):
        # FastAPI のエンドポイント URL
        api_endpoint = "http://127.0.0.1:8000/analyze_image/"

        try:
            # POST リクエストを送信して解析結果を取得
            response = requests.post(api_endpoint, files={'file': uploaded_file})
            data = response.json()

            # 解析結果を表示
            st.write(f"問題数: {data['問題数']}")
            st.write(f"正解数: {data['正解数']}")
            st.write(f"正解率: {data['正解率']:.2%}")
        except Exception as e:
            st.error(f"エラーが発生しました: {e}")
