import sys
import argparse 

add_to_sys_path = "yoloface"
if add_to_sys_path not in sys.path:
    sys.path.append(add_to_sys_path)

from yoloface.face_detector import YoloDetector
import numpy as np
from PIL import Image
import cv2
import warnings
warnings.filterwarnings("ignore")

# args
parser = argparse.ArgumentParser(description="Face detection script with YOLO.")
parser.add_argument("--test_mode", action="store_true", help="Set this flag to disable webcam for face detection.")
parser.add_argument("--device", type=str,default='cpu', help="Device used for YOLO model, 'cpu' or 'cuda'")

args = parser.parse_args()

web_cam = False if args.test_mode else True
device = args.device

model = YoloDetector(target_size=None, device=device, min_face=90)

if not web_cam:
    # Open an image
    image = Image.open('./image/test.jpeg')

    # Resize the image
    new_size = (736, 720)  # (width, height)
    resized_image = image.resize(new_size)

    # Convert to NumPy array
    image_array = np.array(resized_image)

    bboxes,points = model.predict(image_array)

    bgr_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    # Annotate the image``
    for i, box in enumerate(bboxes[0]):
        x_min, y_min, x_max, y_max = box

        # crop the faces
        cropped_image = bgr_image[y_min:y_max, x_min:x_max].copy()

        # Draw the rectangle
        cv2.rectangle(bgr_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    
        cropped_img_path = f"image/cropped_image_{i}.jpg"
        cv2.imwrite(cropped_img_path, cropped_image)

    # Save or display the annotated image
    annotated_image_path = "image/annotated_image.jpg"
    cv2.imwrite(annotated_image_path, bgr_image)

else:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit()

    while True:
        # Capture a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Resize the frame to match the target size
        resized_frame = cv2.resize(frame, (736, 720))  # (width, height)

        # Convert BGR frame to RGB for the model
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

        # Predict faces in the frame
        bboxes, points = model.predict(rgb_frame)

        # Annotate the frame
        for box in bboxes[0]:
            x_min, y_min, x_max, y_max = box

            # crop the faces
            cropped_image = rgb_frame[y_min:y_max, x_min:x_max].copy()

            # Draw the rectangle
            cv2.rectangle(resized_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # crop faces (RGB)
            cv2.imshow("cropped face", cropped_image)

        # Display the annotated frame (BGR)
        cv2.imshow("Webcam Face Detection", resized_frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()