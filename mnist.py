import os
from io import BytesIO
import base64
from flask import Flask, request, redirect, render_template, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing import image
from matplotlib import pyplot as plt
import cv2
from PIL import Image
import numpy as np

classes = ["0","1","2","3","4","5","6","7","8","9"]
image_size = 28

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

model = load_model('./model.h5')#学習済みモデルをロード
def save_and_display_img(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    return "data:image/png;base64," + img_str.decode('utf-8')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    images = {}
    pred_answer = ""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            filepath = os.path.join(UPLOAD_FOLDER, filename)

            #受け取った画像を読み込み、np形式に変換
            img_pil = Image.open(filepath).resize((image_size, image_size)).convert('L')
            img_np = np.asarray(img_pil, dtype=np.float32) / 255.0


import os
from io import BytesIO
import base64
from flask import Flask, request, redirect, render_template, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing import image
import cv2
from PIL import Image
import numpy as np

classes = ["0","1","2","3","4","5","6","7","8","9"]
image_size = 28

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

model = load_model('./model.h5')

def save_and_display_img(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    return "data:image/png;base64," + img_str.decode('utf-8')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    images = {}
    pred_answer = ""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            filepath = os.path.join(UPLOAD_FOLDER, filename)

            # Read and preprocess the image
            img_pil = Image.open(filepath).resize((image_size, image_size)).convert('L')
            img_np = np.asarray(img_pil, dtype=np.float32) / 255.0

            # Image preprocessing steps
            blur_kernel = (3, 3)
            img_data_smoothed = cv2.GaussianBlur(img_np, blur_kernel, 0)
            img_data_smoothed_8u = (img_data_smoothed * 255).astype(np.uint8)
            _, img_data_binarized = cv2.threshold(img_data_smoothed_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = np.ones((2, 2), np.uint8)
            img_data_dilated = cv2.dilate(img_data_binarized, kernel, iterations=1)
            img_data_eroded = cv2.erode(img_data_dilated, kernel, iterations=1)
            img_data_final = img_data_eroded / 255.0

            # Save processed images for display
            images["Loaded Image"] = save_and_display_img(img_pil)
            images["Binarized"] = save_and_display_img(Image.fromarray(img_data_binarized))
            images["Dilated"] = save_and_display_img(Image.fromarray(img_data_dilated))
            images["Eroded"] = save_and_display_img(Image.fromarray(img_data_eroded))
            images["Final Processed"] = save_and_display_img(Image.fromarray((img_data_final * 255).astype(np.uint8)))

            # Make predictions
            data = img_data_final.reshape(1, 28, 28, 1)
            result = model.predict(data)[0]
            predicted = result.argmax()
            pred_answer = "これは " + classes[predicted] + " です"
    
    return render_template("index.html", answer=pred_answer, images=images)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host ='0.0.0.0',port = port)