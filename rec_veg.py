import base64
from io import BytesIO
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import tensorflow as tf
import numpy as np
import csv

# 載入模型
model = load_model('model_mnV2(best).keras')

def load_classes(csv_path='classes.csv'):
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        return [row[1] for row in reader]

# 載入類別名稱
classes = load_classes('classes.csv')

def rec_veg(base64_string):


        # 處理 base64 字串
    if base64_string.startswith("data:image"):
        base64_string = base64_string.split(",")[1]
    image_bytes = base64.b64decode(base64_string)
    image_file = BytesIO(image_bytes)
    print(image_file)


    # 載入圖片並前處理
    img = load_img(image_file, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = tf.expand_dims(img_array, axis=0)

    # 預測
    preds = model.predict(img_array)
    pred_idx = tf.argmax(preds, axis=1).numpy()[0]
    confidence = tf.reduce_max(preds).numpy() * 100

    print(f"預測類別：{classes[pred_idx]}")
    print(f"信心度：{confidence:.2f}%")
