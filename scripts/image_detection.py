# ther are two codes in this you can run any as you wish check while running the code.

'''
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# --- Modify these paths ---
MODEL_PATH = "models/faster_rcnn_trained_final_1.pth"
IMAGE_DIR = "accident_detection.v2i.coco/test"
NUM_IMAGES = 20 # Number of images to visualize
CLASS_NAMES = {1: "accident"}  # Only show accidents

# --- Load model ---
from torchvision.models.detection import fasterrcnn_resnet50_fpn

model = fasterrcnn_resnet50_fpn(num_classes=4)  # 1 class + background
model.load_state_dict(torch.load(MODEL_PATH))
model.eval().cuda()

# --- Transform ---
transform = T.Compose([
    T.ToTensor(),
])

# --- Function to plot ---
def visualize_prediction(img_path):
    image = Image.open(img_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).cuda()
    with torch.no_grad():
        output = model(tensor)[0]

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)

    for box, score, label in zip(output["boxes"], output["scores"], output["labels"]):
        # ✅ Only show accidents (label=1) with high confidence
        if label != 1 or score < 0.7:  # Adjust threshold (0.7 is stricter)
            continue

        box = box.cpu().numpy()
        rect = patches.Rectangle(
            (box[0], box[1]), box[2] - box[0], box[3] - box[1],
            linewidth=2, edgecolor='green', facecolor='none'  # ✅ Green for accidents
        )
        ax.add_patch(rect)
        ax.text(
            box[0], box[1]-5, 
            f"Accident ({score:.2f})", 
            color='green', fontsize=12, backgroundcolor='white'
        )

    plt.axis('off')
    plt.show()

# --- Run on a few validation images ---
image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.png'))][:NUM_IMAGES]
for img_file in image_files:
    img_path = os.path.join(IMAGE_DIR, img_file)
    print(f"Visualizing: {img_path}")
    visualize_prediction(img_path)'''

#Run above or below anything is fine 
'''import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# --- Config ---
MODEL_PATH = "models/faster_rcnn_trained_final_1.pth"
IMAGE_DIR = "accident_detection.v2i.coco/valid"
NUM_IMAGES = 10  # Number of images to test
CONFIDENCE_THRESHOLD = 0.3  # Only show predictions with score > 0.7

# --- Load Model ---
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=4)  # 1 class + background
model.load_state_dict(torch.load(MODEL_PATH))
model.eval().cuda()

# --- Transform ---
transform = T.Compose([T.ToTensor()])

# --- Plot Predictions ---
def visualize_prediction(img_path):
    image = Image.open(img_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).cuda()
    
    with torch.no_grad():
        output = model(tensor)[0]

    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)

    for box, score, label in zip(output["boxes"], output["scores"], output["labels"]):
        if label != 1 or score < CONFIDENCE_THRESHOLD:  # Only accidents + high confidence
            continue

        box = box.cpu().numpy()
        rect = patches.Rectangle(
            (box[0], box[1]), box[2] - box[0], box[3] - box[1],
            linewidth=2, edgecolor="green", facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(
            box[0], box[1] - 10, 
            f"Accident ({score:.2f})", 
            color="white", fontsize=10, bbox=dict(facecolor="green", alpha=0.8)
        )

    plt.axis("off")
    plt.show()

# --- Run on sample images ---
image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith((".jpg", ".png"))][:NUM_IMAGES]
for img_file in image_files:
    img_path = os.path.join(IMAGE_DIR, img_file)
    print(f"🔍 Analyzing: {img_path}")
    visualize_prediction(img_path)'''
