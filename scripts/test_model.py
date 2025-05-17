import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# Load the pretrained Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# Modify the box predictor to match the number of classes in your dataset
in_features = model.roi_heads.box_predictor.cls_score.in_features
num_classes = 6  # Adjust this based on your dataset (e.g., Accident, No_Accident, Background)
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Move model to the appropriate device
model.to(device)
model.eval()  # Set to evaluation mode

# Dummy input for testing (1 image, 3 channels, 224x224 size)
dummy_input = torch.randn(1, 3, 224, 224).to(device)

# Run a forward pass
with torch.no_grad():
    output = model(dummy_input)

print("✅ Model loaded and tested successfully!")
print("Output:", output)
