from fastapi import FastAPI, File, UploadFile
import cv2
import easyocr
from sklearn.cluster import DBSCAN
import numpy as np

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
    
    # クラスター数の出力
    num_clusters = len(np.unique(dbscan.labels_))
    print(f"Number of clusters: {num_clusters}")
    
    return clustered_coordinates

# 問題位置座標から1問あたりの切り取りサイズを決める
def get_box_size(problem_clusters):

    num_clusters = len(problem_clusters)
    cluster1 = np.array(problem_clusters)[0]
    cluster2 = np.array(problem_clusters)[1]

    # 1問の平均幅
    x_interval = cluster2-cluster1
    box_width = int(np.mean(x_interval, axis=0)[0])

    # 1問の平均高さ
    box_height_list = []
    for i in range(len(cluster1)-1):
        y_interval = (cluster1[i+1]-cluster1[i])[1]
        box_height_list.append(y_interval)
    box_height = int(np.mean(box_height_list))

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
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    # 画像の二値化処理(閾値130を超えた画素を255にする。)
    threshold = 130
    ret, img_binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

    # easyocrのモデル選択
    reader = easyocr.Reader(['ja', 'en'])

    # easyocrの実行（問題位置検出のため）
    result = reader.readtext(img_binary)

    # 問題座標の取得
    problem_destination = extract_coordinates(result)

    # 問題を列ごとの分類
    problem_clusters = column_clustering(problem_destination)

    # 問題ごとの切り取りサイズを算出
    box_width, box_height = get_box_size(problem_clusters)

    # 初期化
    problem_count = 0
    correct_count = 0

    # 1問ごとにeasyocrを実行
    for cluster in problem_clusters:
        for problem in cluster:
            problem_count += 1

            # 問題の座標算出
            x1 = problem[0]
            x2 = problem[0] + box_width
            y1 = problem[1] - box_height // 2
            y2 = problem[1] + box_height // 2

            # 座標位置の画像切り取り
            problem_img = img_binary[y1:y2, x1:x2] # img[top : bottom, left : right]

            # easyocrで問題文と回答を読み取り
            problem_result_simple = reader.readtext(problem_img, detail=0)
            recognition = correct_recognition(''.join(problem_result_simple))
            start_index = recognition.find(')')
            end_index = recognition.find('=')
            problem_text = recognition[start_index + 1:end_index].strip()
            answer = recognition[end_index + 1:].strip()

            # 回答の評価
            try:
                calculation = eval(problem_text)
                result_match = (answer == str(calculation))
                if result_match:
                    correct_count += 1
            except Exception as e:
                print("エラーが発生しました:", e)

    # 結果を返す
    return {
        "問題数": problem_count,
        "正解数": correct_count,
        "正解率": correct_count / problem_count if problem_count != 0 else 0
    }
