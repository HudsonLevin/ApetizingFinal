import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# ==================== ⚡ 1. โหลดโมเดล ====================
print("📌 กำลังโหลดโมเดล...")
model = load_model("apetizing_model_test_uni.keras")
print("✅ โหลดโมเดลสำเร็จ!")

# ==================== ⚡ 2. โหลดชุดข้อมูลทดสอบ ====================
print("📌 กำลังโหลดข้อมูลจาก CSV...")

CSV_PATH = "data_from_questionaire.csv"
IMAGE_FOLDER = "datasets/"

df = pd.read_csv(CSV_PATH)

# ใช้ข้อมูลทดสอบ 200 คู่ (หากมีน้อยกว่านั้นก็ใช้ทั้งหมด)
df = df.sample(n=min(200, len(df)), random_state=42).reset_index(drop=True)

def load_and_preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img) / 255.0  # Normalize เป็น 0-1
    return img

x1, x2, y_true = [], [], []

for _, row in df.iterrows():
    img1_path = os.path.join(IMAGE_FOLDER, row["Menu"], row["Image 1"])
    img2_path = os.path.join(IMAGE_FOLDER, row["Menu"], row["Image 2"])
    
    if os.path.exists(img1_path) and os.path.exists(img2_path):
        x1.append(load_and_preprocess_image(img1_path))
        x2.append(load_and_preprocess_image(img2_path))
        y_true.append(row["Winner"] - 1)  # แปลง 1 -> 0, 2 -> 1

x1, x2, y_true = np.array(x1), np.array(x2), np.array(y_true)

print(f"✅ โหลดรูปภาพสำหรับทดสอบทั้งหมด {len(x1)} คู่!")

# ==================== ⚡ 3. ใช้โมเดลพยากรณ์ ====================
print("📌 กำลังพยากรณ์ผลลัพธ์...")
y_pred_prob = model.predict([x1, x2])  # ได้ค่า probability (0 - 1)
y_pred = (y_pred_prob > 0.5).astype(int)  # แปลงเป็น 0 หรือ 1

print("✅ ทำการพยากรณ์เสร็จสิ้น!")

# ==================== ⚡ 4. คำนวณและแสดง Confusion Matrix ====================
print("📌 กำลังสร้าง Confusion Matrix...")

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Image 1", "Image 2"], yticklabels=["Image 1", "Image 2"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

print("✅ แสดง Confusion Matrix สำเร็จ!")
