import logging
import os
import numpy as np
import zipfile
import onnxruntime
from transformers import AutoTokenizer
from underthesea import word_tokenize
from flask import Flask, request, jsonify
import requests

logging.getLogger("transformers").setLevel(logging.ERROR)

# Khởi tạo Flask app
app = Flask(__name__)

file_zip = "phobert_quantized.zip"
destination_folder = os.path.dirname(file_zip)
onnx_file_path = os.path.join(destination_folder, "phobert_quantized.onnx")
label_map = {0: 'Angry', 1: 'Happy', 2: 'InLove', 3: 'Neutral', 4: 'Sad', 5: 'Worry'}
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
onnx_model_path = "phobert_quantized.onnx"
url = 'https://thanglongedu-my.sharepoint.com/:u:/g/personal/a44212_thanglong_edu_vn/ESdkpmkSmpZNh37XXbGG-HcBM2DTT0LBfsKAaUaSKKK5pg?download=1'

if not os.path.exists(file_zip):
    print("File ZIP chưa tồn tại, đang tải về...")
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_zip, 'wb') as f:
            f.write(response.content)
        print("Tải file thành công.")
    else:
        print(f"Không thể tải file, mã trạng thái: {response.status_code}")
else:
    print("File ZIP đã tồn tại.")

if not os.path.exists(onnx_file_path):
    with zipfile.ZipFile(file_zip, 'r') as zip_ref:
        zip_ref.extractall(destination_folder)
    print("Đã giải nén file.")
else:
    print("File phobert_quantized.onnx đã tồn tại, không cần giải nén.")

ort_session = onnxruntime.InferenceSession(onnx_model_path)

def encode_text(text):
    tokenized_text = word_tokenize(text, format="text")
    encoding = tokenizer(tokenized_text, truncation=True, padding=True, max_length=128)
    return {
        "input_ids": np.array(encoding["input_ids"]),  
        "attention_mask": np.array(encoding["attention_mask"])
    }

def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))  # Tránh overflow
    return np.round(exp_logits / np.sum(exp_logits),2)

def predict_onnx(text):
    encoded_text = encode_text(text.lower())
    input_ids_np = np.array([encoded_text['input_ids']], dtype=np.int64)
    attention_mask_np = np.array([encoded_text['attention_mask']], dtype=np.int64)
    ort_outputs = ort_session.run(["logits"], {"input_ids": input_ids_np, "attention_mask": attention_mask_np})
    predicted_class = np.argmax(ort_outputs[0], axis=1)[0]
    logits = ort_outputs[0][0]
    softmax_values = softmax(logits)
    possibilities = [{label_map[i]: float(softmax_values[i]) for i in range(len(softmax_values))}]
    return label_map[predicted_class], possibilities

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        text = data.get("text", "")
        
        if not text:
            return jsonify({"error": "Text is required"}), 400
        
        predicted_label, possibilities = predict_onnx(text)
        result = {
            "result": predicted_label,
            "possibilities": possibilities
        }
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
