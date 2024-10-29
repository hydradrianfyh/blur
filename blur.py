import os
import sys
import cv2
import numpy as np
from deepface import DeepFace
import easyocr
import torch
from facenet_pytorch import MTCNN
import yt_dlp
from ultralytics import YOLO

# 解决 OpenMP 初始化问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 添加 LPRNet_Pytorch 路径
sys.path.append('LPRNet_Pytorch')  # 假设 LPRNet_Pytorch 目录在同一层级
sys.path.append('FaceDetection-DSFD')
# from FaceDetection-DSFD.models.dsfd import DSFDDetector
from LPRNet_Pytorch.model.LPRNet import LPRNet
from LPRNet_Pytorch.data.load_data import CHARS


print(torch.cuda.is_available())

backends = [
  'opencv', 
  'ssd', 
  'dlib', 
  'mtcnn', 
  'fastmtcnn',
  'retinaface', 
  'mediapipe',
  'yolov8',
  'yunet',
  'centerface',
]

# # 设置 MTCNN 模型权重的本地路径
# mtcnn_weights = os.path.expanduser('~/.cache/torch/hub/checkpoints/mtcnn.pt')

# 初始化MTCNN人脸检测器
# print("Initializing MTCNN...")
# print(torch.cuda.is_available())
# mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')


# # 初始化EasyOCR车牌检测器
# reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

# 初始化LPRNet车牌检测器
print("Initializing LPRNet...")
lprnet = LPRNet(lpr_max_len=8, class_num=len(CHARS), dropout_rate=0.5, phase=False)
lprnet.load_state_dict(torch.load('/home/yanhao/Code/blur/LPRNet_Pytorch/weights/Final_LPRNet_model.pth', map_location=torch.device('cpu')))
lprnet.eval()

# 加载YOLOv5模型，用于检测车辆和行人
# print("Loading YOLOv5 model...")
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device='cuda' if torch.cuda.is_available() else 'cpu', trust_repo=True)
# names = model.names

# 初始化YOLOv8模型进行车牌检测
# 初始化YOLOv8模型进行车牌检测
print("Initializing YOLOv8 for license plate detection...")
yolo_model = YOLO('/home/yanhao/Code/blur/license_plate_detector.pt')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def detect_faces_with_mtcnn(frame):
    mtcnn = MTCNN(keep_all=True, device=device)
    boxes, _ = mtcnn.detect(frame)
    print(f"Detected faces with MTCNN: {boxes}")
    return boxes

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

# def detect_faces_with_dsfd(frame):
#     dsfd = DSFD(pretrained=True, device=device)
#     boxes = dsfd.detect(frame)
#     print(f"Detected faces with DSFD: {boxes}")
#     return boxes

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


# def detect_license_plates(frame):
#     results = reader.readtext(frame)
#     plates = []
#     for (bbox, text, prob) in results:
#         if prob > 0.5:  # 只考虑置信度大于0.5的结果
#             (x1, y1), (x2, y2), (x3, y3), (x4, y4) = bbox
#             x_min = min(x1, x2, x3, x4)
#             x_max = max(x1, x2, x3, x4)
#             y_min = min(y1, y2, y3, y4)
#             y_max = max(y1, y2, y3, y4)
#             plates.append((int(x_min), int(y_min), int(x_max), int(y_max)))
#     print(f"Detected plates: {plates}")
#     return plates

# def detect_license_plates(frame):
#     print("detect lincense plates....")
#     results = model(frame)
#     plates = []
#     detected_classes = set()
#     for *xyxy, conf, cls in results.xyxy[0].numpy():
#         detected_classes.add(int(cls))
#         if cls == 2:
#             x1, y1, x2, y2 = map(int, xyxy)
#             plate_region = frame[y1:y2, x1:x2]
#             plate_region = cv2.resize(plate_region, (94, 24))
#             if plate_region.shape[2] == 1:
#                 plate_region = cv2.cvtColor(plate_region, cv2.COLOR_GRAY2BGR)
#             plate_region = torch.tensor(plate_region, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
#             lprnet_output = lprnet(plate_region)
#             if lprnet_output is not None:
#                 plates.append([x1, y1, x2, y2])
#     return plates

# def detect_license_plates(frame):
#     results = reader.readtext(frame)
#     plates = []
#     for (bbox, text, prob) in results:
#         if prob > 0.5:  # 只考虑置信度大于0.5的结果
#             (x1, y1), (x2, y2), (x3, y3), (x4, y4) = bbox
#             x_min = min(x1, x2, x3, x4)
#             x_max = max(x1, x2, x3, x4)
#             y_min = min(y1, y2, y3, y4)
#             y_max = max(y1, y2, y3, y4)
#             plates.append((int(x_min), int(y_min), int(x_max), int(y_max)))
#     print(f"Detected plates: {plates}")
#     return plates

def detect_and_blur(frame):
    # frame = resize_frame(frame)
    faces = detect_faces_with_deepface(frame)
    if faces is not None:
        for (x1, y1, x2, y2) in faces:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            if x1 < 0: x1 = 0
            if y1 < 0: y1 = 0
            if x2 > frame.shape[1]: x2 = frame.shape[1]
            if y2 > frame.shape[0]: y2 = frame.shape[0]
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
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = None
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
            print(f"Reading frame {frame_count}")
            if frame is None or frame.size == 0:
                print(f"Empty frame at {frame_count}")
                continue
            blurred_frame = detect_and_blur(frame)
            if blurred_frame is None or blurred_frame.size == 0:
                print(f"Empty blurred frame at {frame_count}")
                continue
            print(f"Writing frame {frame_count}")
            out.write(blurred_frame)
    except Exception as e:
        print(f"Error during processing: {e}")
    finally:
        cap.release()
        out.release()
        print(f"Released video capture and writer resources.")
    print(f"Video processed and saved to: {output_path}")

def find_videos_in_folder(folder):
    video_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_files.append(os.path.join(root, file))
    return video_files

def main(input_folder):
    video_files = find_videos_in_folder(input_folder)
    for video_file in video_files:
        output_file = f"{os.path.splitext(video_file)[0]}_blurred.avi"
        process_video(video_file, output_file)

def download_and_process_video(url, output_path):
    ydl_opts = {
        'format': 'mp4',
        'outtmpl': 'downloaded_video.%(ext)s',
        'noplaylist': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    process_video('downloaded_video.mp4', output_path)
    # os.remove('downloaded_video.mp4')

def stream_video(url):
    download_and_process_video(url, 'stream_blurred_output.avi')

if __name__ == "__main__":
    input_path = "/home/yanhao/Code/blur/test_video_CN.mp4"
    output_path = "/home/yanhao/Code/blur/blur_video_CN.avi"
    process_video(input_path,output_path)
    # import argparse

    # parser = argparse.ArgumentParser(description='Process videos to blur faces and license plates.')
    # parser.add_argument('input_folder', nargs='?', default=None, help='Path to the folder containing videos.')
    # parser.add_argument('--stream',default="https://youtu.be/d59wPaenvzs?si=OEcvU4ZfWp5z84t-", type=str, help='URL of the video stream to process.')

    # args = parser.parse_args()

    # if args.stream:
    #     stream_video(args.stream)
    # elif args.input_folder:
    #     main(args.input_folder)
    # else:
    #     print("Please provide either an input folder or a stream URL.")