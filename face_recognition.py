import cv2
import os
import torch
from scipy.spatial.distance import cosine
from facenet_pytorch import InceptionResnetV1
from main import init_yolo_detector, detect_faces_in_image

def load_face_database():
    """Simulate loading a face database."""
    print("Loading face database...")
    database = {
        "Alice": torch.rand(512),
        "Bob": torch.rand(512),
    }
    print(f"Loaded {len(database)} identities into the database.")
    return database

def recognize_faces(cropped_faces, facenet, face_database):
    """Recognize faces from cropped face images."""
    recognized_names = []
    required_size = (160, 160)  # Facenet requires at least 160x160 input size

    for idx, face in enumerate(cropped_faces):
        print(f"Processing cropped face {idx + 1}...")
        print(f"Shape of cropped face {idx + 1}: {face.shape}")

        # 如果是灰度图，转换为 RGB
        if len(face.shape) == 2:
            face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)

        # 如果图像大小不足，调整到 Facenet 支持的大小
        if face.shape[0] < required_size[0] or face.shape[1] < required_size[1]:
            print(f"Resizing cropped face {idx + 1} to {required_size}")
            face = cv2.resize(face, required_size, interpolation=cv2.INTER_LINEAR)

        face_tensor = torch.tensor(face).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        embedding = facenet(face_tensor).detach().cpu().squeeze()
        name, max_similarity = "Unknown", 0.0
        for db_name, db_embedding in face_database.items():
            similarity = 1 - cosine(embedding.numpy(), db_embedding.numpy())
            print(f"Similarity with {db_name}: {similarity:.2f}")
            if similarity > 0.9 and similarity > max_similarity:
                name, max_similarity = db_name, similarity
        print(f"Face {idx + 1} recognized as: {name}")
        recognized_names.append(name)
    return recognized_names


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = init_yolo_detector(device=device)
    facenet = InceptionResnetV1(pretrained="vggface2").eval()

    # Detect faces in an image
    image_path = "./image/test.jpeg"
    print(f"Detecting faces in image: {image_path}")
    bboxes, _, annotated_image_path = detect_faces_in_image(model, image_path)

    print(f"Annotated image saved to: {annotated_image_path}")

    # Load database and recognize faces
    face_database = load_face_database()

    # Load cropped faces and handle missing files
    cropped_faces = []
    for idx, face_path in enumerate(["./image/cropped_image_0.jpg", "./image/cropped_image_1.jpg"]):
        if not os.path.exists(face_path):
            print(f"File not found: {face_path}")
            continue
        face = cv2.imread(face_path)
        if face is None:
            print(f"Failed to read image: {face_path}")
            continue
        cropped_faces.append(face)

    if not cropped_faces:
        print("No valid cropped faces found. Exiting.")
        exit()

    recognized_names = recognize_faces(cropped_faces, facenet, face_database)

    print("Recognition Results:")
    for name in recognized_names:
        print(f" - {name}")
