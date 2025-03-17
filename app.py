from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO

# ✅ โหลดโมเดล YOLOv8
model = YOLO("yolo12n.pt")

app = Flask(__name__)

# ✅ อนุญาตให้รับไฟล์ได้สูงสุด 50MB
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

def preprocess_image(image):
    """แปลงไฟล์ภาพจาก Flask request เป็นฟอร์แมตที่ YOLO รับได้"""
    image = Image.open(image.stream).convert("RGB")
    return image

def draw_boxes(image, detections):
    """วาด Bounding Box และชื่อคลาสลงบนภาพ"""
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for obj in detections:
        x1, y1, x2, y2 = obj["bbox"]["x1"], obj["bbox"]["y1"], obj["bbox"]["x2"], obj["bbox"]["y2"]
        class_name = obj["class"]
        confidence = obj["confidence"]

        # ✅ วาดกรอบ
        color = (0, 255, 0)
        thickness = 2
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        # ✅ ใส่ข้อความ
        label = f"{class_name} ({confidence:.2f}%)"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return image

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image found"}), 400

    file = request.files["image"]
    image = preprocess_image(file)

    # ✅ ทำการตรวจจับวัตถุด้วย YOLOv8
    results = model(image)

    detected_objects = []
    for result in results:
        for box in result.boxes:
            cls = int(box.cls.item())  
            class_name = model.names[cls]  
            conf = float(box.conf.item()) * 100  

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            detected_objects.append({
                "class": class_name,
                "confidence": round(conf, 2),
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            })

    # ✅ วาด Bounding Box ลงบนภาพ
    output_image = draw_boxes(image, detected_objects)

    # ✅ บันทึกภาพเป็นไฟล์ชั่วคราวp
    output_path = "output.jpg"
    cv2.imwrite(output_path, output_image)

    # ✅ ส่ง JSON และให้ Node-RED ดาวน์โหลดภาพแยกต่างหาก
    return jsonify({
        "image_url": "https://yolo-ldh8.onrender.com/image",  # URL สำหรับดาวน์โหลดภาพ
        "detected_objects": detected_objects
    })

@app.route("/image", methods=["GET"])
def get_image():
    """ส่งภาพที่มีการตรวจจับ Bounding Box กลับไป"""
    output_path = "output.jpg"  # เปลี่ยนเป็นพาธของภาพที่ต้องการส่ง
    try:
        return send_file(output_path, mimetype="image/jpeg")
    except Exception as e:
        return str(e), 500  # ส่งข้อความ Error ถ้ามีปัญหา

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
