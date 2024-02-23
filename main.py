from fastapi import FastAPI, File, UploadFile
import cv2
import easyocr
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
import base64
import io

app = FastAPI()

# easyocrのresultから問題位置座標を返す
def extract_coordinates(data):
    problem_coordinates = []

    for entry in data:
        coordinates, text, confidence = entry

        # テキストに '(' が含まれる場合のみ抽出
        if '(' in text:
            # 矩形のlefttop/leftbottomの中間点を抽出
            left_top = coordinates[0]
            left_bottom = coordinates[3]
            left_center = [(left_top[0] + left_bottom[0]) // 2, (left_top[1] + left_bottom[1]) // 2]
            problem_coordinates.append(left_center)
    return problem_coordinates

# 問題位置座標から列ごとのデータに分ける
def column_clustering(problem_destination):

    # X座標の抽出
    x_coordinates = [point[0] for point in problem_destination] # x 座標を抽出
    X = np.array(x_coordinates).reshape(-1, 1) # リストをNumPy配列に変換

    # DBSCANクラスタリング
    dbscan = DBSCAN(eps=50, min_samples=2)  # eps: データ点の影響範囲, min_samples: クラスターとみなす最小のデータ点数
    dbscan.fit(X)

    # クラスターごとの座標をリスト形式に戻す
    clustered_coordinates = [[] for _ in range(len(np.unique(dbscan.labels_)))]
    for i, label in enumerate(dbscan.labels_):
        clustered_coordinates[label].append(problem_destination[i])
    clustered_coordinates = sorted(clustered_coordinates, key=lambda x: x[0]) #クラスターをx座標に対して昇順となるよう並び替え

    # クラスター数の出力
    num_clusters = len(np.unique(dbscan.labels_))
    print(f"Number of clusters: {num_clusters}")

    return clustered_coordinates

# 問題位置座標から1問あたりの切り取りサイズを決める
def calculate_box_width(problem_clusters):
    cluster_centers = []
    for cluster in problem_clusters:
        cluster_x = [point[0] for point in cluster]
        cluster_center_x = np.mean(cluster_x)
        cluster_centers.append(cluster_center_x)
    print(f"cluster_centers: {cluster_centers}")
    
    cluster_center_diffs = [abs(cluster_centers[i] - cluster_centers[i+1]) for i in range(len(cluster_centers)-1)]
    box_width = np.mean(cluster_center_diffs)
    
    return box_width

def calculate_box_height(problem_clusters):
    cluster_y_gaps = []
    for cluster in problem_clusters:
        cluster_y = [point[1] for point in cluster]
        cluster_y_sorted = sorted(cluster_y)
        cluster_y_gaps.extend([cluster_y_sorted[i+1] - cluster_y_sorted[i] for i in range(len(cluster_y_sorted)-1)])
    
    cluster_y_gap_mean = np.mean(cluster_y_gaps)
    
    return cluster_y_gap_mean

def calculate_box_dimensions(problem_clusters):
    box_width = int(calculate_box_width(problem_clusters))
    box_height = int(calculate_box_height(problem_clusters))
    return box_width, box_height



# 印字の誤認識に対する修正関数
def correct_recognition(text):
    # 誤認識が発生する可能性のある文字と、それを代替する文字の辞書
    correction_dict = {
        'O': '0',
        'o': '0',
        'D': '0',
        '｜': '1',
        '|': '1',
        'I': '1',
        '[': '1',
        'L': '1', # 碧特有
        'l': '1', # 碧特有
        'し': '1', # 碧特有
        'ろ': '3', # 碧特有
        '十':'+',
        '二':'=',
        '三':'='
        # 他にも必要ならば追加
    }
    # 誤認識が発生する可能性のある文字を修正
    for wrong_char, correct_char in correction_dict.items():
        text = text.replace(wrong_char, correct_char)
    return text

@app.post("/analyze_image/")
async def analyze_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    # img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE) # grayscaleの場合
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # Colorの場合

    # # 画像の二値化処理(閾値130を超えた画素を255にする。)
    # threshold = 130
    # ret, img_binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

    # 画像の二値化処理をしない
    img_binary = img

    # easyocrのモデル選択
    reader = easyocr.Reader(['ja', 'en'])

    # easyocrの実行（問題位置検出のため）
    result = reader.readtext(img_binary)
    print("位置検知reader完了")

    # 問題座標の取得
    problem_destination = extract_coordinates(result)
    print("問題座標の取得完了")

    # 問題を列ごとの分類
    problem_clusters = column_clustering(problem_destination)
    print("問題を列ごとの分類完了")

    # 問題ごとの切り取りサイズを算出
    box_width, box_height = calculate_box_dimensions(problem_clusters)
    print("切り取りサイズを算出完了")

    # 結果格納用のリストを初期化
    df1_data = []
    df2_data = []

    # 画像のコピーを作成
    img_with_marks = img.copy()

    # 初期化
    problem_count = 0
    correct_count = 0
    print("初期化完了")


    # 1問ごとにeasyocrを実行
    for cluster_index, cluster in enumerate(problem_clusters):
        for problem_index, problem_destination in enumerate(cluster):
            problem_count += 1

            print("test1")

            # 問題の座標算出
            x1 = int(problem_destination[0])
            x2 = int(problem_destination[0] + box_width)
            y1 = int(problem_destination[1] - box_height // 2)
            y2 = int(problem_destination[1] + box_height // 2)

            # ×を描画する際の座標
            batsu_x1 = problem_destination[0] - box_height // 2
            batsu_x2 = problem_destination[0] + box_height // 2
            batsu_y1 = problem_destination[1] - box_height // 2
            batsu_y2 = problem_destination[1] + box_height // 2

            # 座標位置の画像切り取り
            problem_img = img_binary[y1:y2, x1:x2] # img[top : bottom, left : right]

            # easyocrで問題文と回答を読み取り
            problem_result_simple = reader.readtext(problem_img, detail=0)
            recognition = correct_recognition(''.join(problem_result_simple))
            start_index = recognition.find(')')
            end_index = recognition.find('=')
            problem_text = recognition[start_index + 1:end_index].strip()
            answer_text = recognition[end_index + 1:].strip()

            # 回答の評価
            try:
                calculation = eval(problem_text)
                result_match = (answer_text == str(calculation))
                if result_match:
                    correct_count += 1
                    # 正解の場合は赤丸を描画
                    cv2.circle(img_with_marks, tuple(problem_destination), box_height//2, (0, 0, 255), thickness=5)
                else:
                    # 不正解の場合は赤い×を描画
                    cv2.line(img_with_marks, (batsu_x1, batsu_y1), (batsu_x2, batsu_y2), (0, 0, 255), thickness=5)
                    cv2.line(img_with_marks, (batsu_x1, batsu_y2), (batsu_x2, batsu_y1), (0, 0, 255), thickness=5)

                # 結果をリストに追加
                df1_data.append({
                    "問題番号": f"{cluster_index + 1}-{problem_index + 1}",
                    "問題列": cluster_index + 1,
                    "問題位置": problem_destination,
                    "OCR結果": ''.join(problem_result_simple),
                    "problem": problem_text,
                    "answer": answer_text,
                    "calculation": calculation,
                    "正解/不正解": result_match
                })
            
            except Exception as e:
                print("エラーが発生しました:", e)
        print(df1_data)

    
    # 結果をリストに追加
    df2_data.append({
        "問題数": problem_count,
        "正解数": correct_count,
        "正答率": correct_count / problem_count if problem_count > 0 else 0
    })
    print(df2_data)

    # 画像に正解数を描画
    # 点数を描画する座標を設定
    score_position = (int(img_with_marks.shape[1] * 0.5), int(img_with_marks.shape[0] * 0.05))
    # 画像に点数を描画
    #cv2.putText(img_with_marks, f"Score: {int(correct_count/problem_count * 100)} / 100", score_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(img_with_marks,
            text=f"Score: {int(correct_count / problem_count * 100)}/100",
            org=score_position,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=5,
            color=(0, 0, 255),
            thickness=4,
            lineType=cv2.LINE_AA)


    # df1の各辞書内のnumpy.int64型の値をint型に変換
    for item in df1_data:
        item['問題列'] = int(item['問題列'])
        item['問題位置'] = list(map(int, item['問題位置']))
        item['calculation'] = int(item['calculation'])

    # df2の各辞書内のnumpy.int64型の値をint型に変換
    for item in df2_data:
        item['問題数'] = int(item['問題数'])
        item['正解数'] = int(item['正解数'])
        item['正答率'] = float(item['正答率'])
    
    # 画像をBase64形式にエンコードして返す
    _, img_with_marks_encoded = cv2.imencode('.png', img_with_marks)
    img_with_marks_base64 = base64.b64encode(img_with_marks_encoded).decode()


    # 結果を返す
    return {"df1": df1_data, "df2": df2_data, "image": img_with_marks_base64}