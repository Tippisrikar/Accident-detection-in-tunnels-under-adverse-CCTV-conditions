import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
import os
import json
from PIL import Image, ImageDraw
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import time
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms as T  # ✅ Correct import

# Define the CustomDataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_info = []
        self.class_map = {}  # Maps category_id to class index starting from 1
        self.next_class_id = 1
        self.image_id_to_annotations = {}
        self.ALLOWED_CATEGORY_IDS = [1, 2, 5]  # accident, fire, smoke

        print(f"[DEBUG] Loading dataset from: {root}")
        ann_path = os.path.join(root, "annotations.coco.json")
        if not os.path.exists(ann_path):
            raise FileNotFoundError(f"[ERROR] Annotation file not found: {ann_path}")

        with open(ann_path, "r") as f:
            coco = json.load(f)

        # Preprocess annotation mapping
        for ann in coco["annotations"]:
            if ann["category_id"] in self.ALLOWED_CATEGORY_IDS:
                img_id = ann["image_id"]
                if img_id not in self.image_id_to_annotations:
                    self.image_id_to_annotations[img_id] = []
                self.image_id_to_annotations[img_id].append(ann)

        # Create class map
        for ann in coco["annotations"]:
            cid = ann["category_id"]
            if cid in self.ALLOWED_CATEGORY_IDS and cid not in self.class_map:
                self.class_map[cid] = self.next_class_id
                self.next_class_id += 1

        # Process image info
        for img in coco["images"]:
            img_path = os.path.join(root, img["file_name"])
            if os.path.exists(img_path):
                self.image_info.append({
                    "id": img["id"],
                    "file_name": img["file_name"],
                    "path": img_path
                })

        print(f"[DEBUG] Loaded {len(self.image_info)} valid images.")
        print(f"[DEBUG] Class map: {self.class_map}")

    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, idx):
        img_info = self.image_info[idx]
        img_path = os.path.join(self.root, img_info["file_name"])
        img = Image.open(img_path).convert("RGB")
        img_id = img_info["id"]
        annots = self.image_id_to_annotations.get(img_id, [])

        boxes, labels = [], []

        for ann in annots:
            category_id = ann["category_id"]
            if category_id not in self.class_map:
                continue
            bbox = ann["bbox"]
            xmin, ymin = bbox[0], bbox[1]
            xmax = xmin + bbox[2]
            ymax = ymin + bbox[3]
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_map[category_id])

        # Skip images without valid boxes
        if len(boxes) == 0:
            return None

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id])
        }

        if self.transform:
            img = self.transform(img)

        return img, target

# Define the get_transform function
def get_transform(train):
    transforms = [torchvision.transforms.ToTensor()]
    if train:
        transforms.append(torchvision.transforms.RandomHorizontalFlip(0.5))
    return torchvision.transforms.Compose(transforms)

# Define the collate_fn
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return [], []
    return tuple(zip(*batch))

# Define the get_model function
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="COCO_V1")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# Function to visualize and save predicted bounding boxes
def visualize_predictions(image, output, target, file_name):
    draw = ImageDraw.Draw(image)

    # Ground truth boxes (for comparison)
    for box in target["boxes"]:
        xmin, ymin, xmax, ymax = box
        draw.rectangle([xmin, ymin, xmax, ymax], outline="green", width=3)

    # Predicted boxes
    for box, score in zip(output["boxes"], output["scores"]):
        xmin, ymin, xmax, ymax = box
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
        draw.text((xmin, ymin), f"{score:.2f}", fill="red")

    # Save the image
    image.save(file_name)

# Define the evaluate function with COCO evaluation
def evaluate(model, data_loader, device, annotation_path, output_dir="predictions"):
    model.eval()
    model.to(device)

    print("[INFO] Starting evaluation...")

    coco_gt = COCO(annotation_path)
    coco_results = []

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            if not images:
                continue

            images = [img.to(device) for img in images]
            outputs = model(images)

            for target, output, img in zip(targets, outputs, images):
                image_id = int(target["image_id"].item())
                boxes = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
                labels = output["labels"].cpu().numpy()

                for box, score, label in zip(boxes, scores, labels):
                    xmin, ymin, xmax, ymax = box
                    coco_results.append({
                        "image_id": image_id,
                        "category_id": int(label),  # This assumes label == COCO category_id
                        "bbox": [float(xmin), float(ymin), float(xmax - xmin), float(ymax - ymin)],
                        "score": float(score)
                    })

                # ✅ Convert tensor to PIL image using torchvision.transforms.ToPILImage
                img = T.ToPILImage()(img.cpu())
                output_image_path = os.path.join(output_dir, f"pred_{image_id}.jpg")
                visualize_predictions(img, output, target, output_image_path)

    if not coco_results:
        print("[WARNING] No valid predictions found.")
        return

    # Save results to temporary file
    results_path = "coco_predictions.json"
    with open(results_path, "w") as f:
        json.dump(coco_results, f, indent=4)

    coco_dt = coco_gt.loadRes(results_path)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

if __name__ == "__main__":
    dataset_path = r"accident_detection.v2i.coco\valid"
    annotation_path = os.path.join(dataset_path, "annotations.coco.json")

    dataset = CustomDataset(dataset_path, transform=get_transform(train=False))
    data_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_classes = dataset.next_class_id
    model = get_model(num_classes)
    model.load_state_dict(torch.load(r"models\faster_rcnn_trained_final_1.pth"))
    model.to(device)

    # Evaluate the model and visualize predictions
    evaluate(model, data_loader, device, annotation_path)
