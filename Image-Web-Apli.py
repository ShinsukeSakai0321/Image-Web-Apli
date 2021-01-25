from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import numpy as np
from keras.models import load_model
from keras.applications.mobilenet import MobileNet, preprocess_input, decode_predictions
from keras.preprocessing.image import img_to_array, load_img

# DB接続用のデータを設定
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
# アップロード先のフォルダを指定
UPLOAD_FOLDER = './static/image/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def img_pred(image):
    # 保存したモデルをロード
    model = load_model('mobile_net_model.h5')

    # 読み込んだ画像を行列に変換
    img_array = img_to_array(image)

    # 3次元を4次元に変換、入力画像は1枚なのでsamples=1
    img_dims = np.expand_dims(img_array, axis=0)

    # Top2のクラスの予測
    preds = model.predict(preprocess_input(img_dims))
    results = decode_predictions(preds, top=2)[0]

    # resultsを整形
    result = [result[1] + str(result[2]) for result in results]

    return result

@app.route('/')
def index():
    return render_template('./flask_api_index.html')

@app.route('/result', methods=['POST'])
def result():
    # submitした画像が存在したら処理する
    if request.method=='POST':


        # ファイルを読み込む
        img_file = request.files['image']

        # ファイル名を取得する
        filename = secure_filename(img_file.filename)

        # 画像のアップロード先URLを生成する
        img_url = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        abs_url=os.path.abspath(img_url)

        # 画像をアップロード先に保存する
        img_file.save(img_url)

        # 画像の読み込み
        image_load = load_img(abs_url, target_size=(224,224))

        # クラスの予測をする関数の実行
        predict_Confidence = img_pred(image_load)

        # render_template('./result.html')
        return render_template('flask_api_index.html', title='予想クラス', predict_Confidence=predict_Confidence, result_img=img_url)

if __name__ == '__main__':
    app.debug = True
    app.run(host='localhost', port=5000)