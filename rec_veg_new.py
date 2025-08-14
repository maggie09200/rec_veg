import base64
from io import BytesIO
import numpy as np
import csv
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array


from tensorflow.keras.applications import efficientnet, mobilenet_v3

# --- 步驟 1: 載入模型與設定 ---
model = load_model('1best_tuned_model.keras')
print("✅ 模型 '1best_tuned_model.keras' 載入成功。")


model_architecture = 'MobileNetV3-Large'

if model_architecture == 'MobileNetV3-Large':
    preprocessing_function = mobilenet_v3.preprocess_input
    target_image_size = (224, 224)
    print("🚀 使用 MobileNetV3-Large 的預處理函式。")
else:
    raise ValueError("不支援的模型架構! 請選擇 'EfficientNet-B0' 或 'MobileNetV3-Large'")

# --- 步驟 2: 載入類別名稱 ---

def load_classes(csv_path='classes_new.csv'):
    """
    從 CSV 檔案載入類別名稱。
    修正: 從 row[1] 改為 row[0] 來讀取正確的欄位。
    """
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            # 假設每個類別名稱佔據一行中的第一欄
            classes = [row[1] for row in reader]
        print(f"✅ 成功從 '{csv_path}' 載入 {len(classes)} 個類別。")
        return classes
    except FileNotFoundError:
        print(f"❌ 錯誤: 找不到類別檔案 '{csv_path}'。")
        return []
    except IndexError:
        print(f"❌ 錯誤: CSV 檔案 '{csv_path}' 格式不正確，請確保每行至少有一欄。")
        return []

# 載入類別名稱
class_names = load_classes('classes_new.csv')

# --- 步驟 3: 定義辨識函式 ---

def rec_veg(base64_string: str):
    """
    接收 Base64 編碼的圖片字串，進行解碼、預處理、預測，並回傳結果。
    
    Args:
        base64_string (str): Base64 編碼的圖片字串。

    Returns:
        dict: 一個包含 'prediction' 和 'confidence' 的字典，如果出錯則回傳 None。
    """
    if not class_names:
        print("❌ 類別名稱未載入，無法進行預測。")
        return None

    try:
        # 處理 base64 字串，移除 data URI scheme (如果有的話)
        if base64_string.startswith("data:image"):
            base64_string = base64_string.split(",")[1]
        
        image_bytes = base64.b64decode(base64_string)
        image_file = BytesIO(image_bytes)

        # 載入圖片並調整為模型需要的尺寸
        # 修正: 從 (128, 128) 改為訓練時使用的 target_image_size
        img = load_img(image_file, target_size=target_image_size)
        
        # 將圖片轉換為 numpy 陣列
        img_array = img_to_array(img)
        
        # 擴展維度以符合模型輸入格式 (batch_size, height, width, channels)
        img_array_expanded = np.expand_dims(img_array, axis=0)
        
        # *** 關鍵修正 ***
        # 使用與訓練時完全相同的預處理函式
        img_preprocessed = preprocessing_function(img_array_expanded)

        # 進行預測
        predictions = model.predict(img_preprocessed)
        
        # 解析預測結果
        scores = predictions[0]
        predicted_index = np.argmax(scores)
        predicted_class = class_names[predicted_index]
        confidence = np.max(scores) * 100
        
        result = {
            "prediction": predicted_class,
            "confidence": f"{confidence:.2f}%"
        }
        return result

    except Exception as e:
        print(f"❌ 預測過程中發生錯誤: {e}")
        return None





# --- 步驟 4: 執行辨識測試 ---
# 這邊在串接的時候要拿掉用用串接的內容就好

if __name__ == '__main__':
    try:
        # 讀取本地圖片檔案並轉換為 base64 字串
        image_path = "白苦瓜.jpg" # 請確認此圖片與程式碼在同一個目錄下
        with open(image_path, "rb") as img_file:
            b64_string = base64.b64encode(img_file.read()).decode('utf-8')
        
        print(f"\n🔍 正在辨識圖片: '{image_path}'...")
        # 呼叫辨識函式
        prediction_result = rec_veg(b64_string)

        # 印出回傳的結果
        if prediction_result:
            print("\n--- 辨識結果 ---")
            print(f"🥒 預測類別: {prediction_result['prediction']}")
            print(f"📈 信心度: {prediction_result['confidence']}")
            
            print("------------------")

    except FileNotFoundError:
        print(f"❌ 測試錯誤: 找不到圖片檔案 '{image_path}'。請將圖片放在正確的路徑。")
    except Exception as e:
        print(f"❌ 執行時發生未知錯誤: {e}")