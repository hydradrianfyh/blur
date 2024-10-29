import os
import sys
import cv2
import numpy as np
from deepface import DeepFace
import torch
from facenet_pytorch import MTCNN
from ultralytics import YOLO

# 解决 OpenMP 初始化问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 添加 LPRNet_Pytorch 路径
sys.path.append('LPRNet_Pytorch')
from LPRNet_Pytorch.model.LPRNet import LPRNet
from LPRNet_Pytorch.data.load_data import CHARS

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
# # 初始化 LPRNet
# print("Initializing LPRNet...")
# lprnet = LPRNet(lpr_max_len=8, class_num=len(CHARS), dropout_rate=0.5, phase=False)
# lprnet.load_state_dict(torch.load('/home/yanhao/Code/blur/LPRNet_Pytorch/weights/Final_LPRNet_model.pth', map_location=torch.device(device)))
# lprnet.eval()

# 初始化 YOLOv8
print("Initializing YOLOv8 for license plate detection...")
yolo_model = YOLO('/home/yanhao/Code/blur/license_plate_detector.pt')
yolo_model.to(device)

def detect_faces_with_deepface(frame):
    try:
        # 使用DeepFace进行人脸检测
        face_objs = DeepFace.extract_faces(frame, detector_backend='retinaface')
        boxes = [(face['facial_area']['x'], face['facial_area']['y'], face['facial_area']['x'] + face['facial_area']['w'], face['facial_area']['y'] + face['facial_area']['h']) for face in face_objs]
        print(f"Detected faces with DeepFace: {boxes}")
        return boxes
    except Exception as e:
        print(f"DeepFace detection error: {e}")
        return []

def detect_license_plates(frame):
    results = yolo_model(frame)
    plates = []
    for result in results:
        for box in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = box
            if score > 0.05:  # 置信度阈值
                plates.append((int(x1), int(y1), int(x2), int(y2)))
    print(f"Detected plates: {plates}")
    return plates

def detect_and_blur(frame):
    faces = detect_faces_with_deepface(frame)
    for (x1, y1, x2, y2) in faces:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        face_region = frame[y1:y2, x1:x2]
        if face_region.size != 0:
            face_region = cv2.GaussianBlur(face_region, (99, 99), 30)
            frame[y1:y2, x1:x2] = face_region
    plates = detect_license_plates(frame)
    for (x1, y1, x2, y2) in plates:
        plate_region = frame[y1:y2, x1:x2]
        if plate_region.size != 0:
            plate_region = cv2.GaussianBlur(plate_region, (99, 99), 30)
            frame[y1:y2, x1:x2] = plate_region
    return frame

def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4v 编码器
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video properties - Width: {frame_width}, Height: {frame_height}, FPS: {fps}")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            print(f"Processing frame {frame_count}")
            blurred_frame = detect_and_blur(frame)
            out.write(blurred_frame)
    except Exception as e:
        print(f"Error during processing: {e}")
    finally:
        cap.release()
        out.release()
        print(f"Released video capture and writer resources.")
    print(f"Video processed and saved to: {output_path}")

if __name__ == "__main__":
    input_path = "/home/yanhao/Code/blur/downloaded_video.mp4"
    output_path = "/home/yanhao/Code/blur/blur_downloaded_video.mp4"  # 使用 mp4 扩展名
    process_video(input_path, output_path)
