#video detection
import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import torchvision

# --- Configuration ---
MODEL_PATH = "models/faster_rcnn_trained_final_1.pth"
VIDEO_PATH = r"D:\srikar\Projects\cctv project\videos\LIVE ACCIDENT_ Truck transporting thinner crashes & blows up in tunnel, sets 11 vehicles on fire.mp4"  # Replace with your video path
OUTPUT_PATH = "output_video_1.mp4"
CONFIDENCE_THRESHOLD = 0.7  # Only consider detections with score > 0.7

# --- Load Model ---
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=4)  # 1 class + background
model.load_state_dict(torch.load(MODEL_PATH))
model.eval().cuda()  # Use .cpu() if no GPU available

# --- Video Setup ---
cap = cv2.VideoCapture(VIDEO_PATH)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Video writer for saving results
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

# --- Transform ---
transform = transforms.Compose([
    transforms.ToTensor(),
])

# --- Process Video Frame-by-Frame ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to PIL Image -> Tensor
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor_image = transform(pil_image).unsqueeze(0).cuda()  # Use .cpu() if no GPU

    # Run inference
    with torch.no_grad():
        outputs = model(tensor_image)[0]

    # Draw bounding boxes for accidents
    for box, score, label in zip(outputs["boxes"], outputs["scores"], outputs["labels"]):
        if label == 1 and score > CONFIDENCE_THRESHOLD:  # Only accidents with high confidence
            box = box.cpu().numpy().astype(int)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(frame, f"Accident: {score:.2f}", (box[0], box[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Write frame to output video
    out.write(frame)

    # Display (optional)
    cv2.imshow("Accident Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ Processed video saved to: {OUTPUT_PATH}")
