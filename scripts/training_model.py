import torch
import torchvision
import torchvision.transforms as T
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, Dataset
import os
import json
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing

# Set multiprocessing start method for Windows
multiprocessing.freeze_support()
torch.multiprocessing.set_start_method('spawn', force=True)

class CustomDataset(Dataset):
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

        # ✅ Skip images without valid boxes
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


def get_transform(train):
    transforms = [T.ToTensor()]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="COCO_V1")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def train(model, train_loader, optimizer, lr_scheduler, num_epochs=10):
    print(f"🚀 Training on device: {device}")
    epoch_losses = []
    epoch_accuracies = []

    for epoch in range(num_epochs):
        print(f"\n📢 Epoch [{epoch+1}/{num_epochs}]")
        model.train()

        total_loss, total_correct, total_samples = 0, 0, 0

        for batch in train_loader:
            if batch is None:
                continue

            images, targets = batch
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()

            with torch.no_grad():
                model.eval()
                outputs = model(images)
                model.train()

                for output, target in zip(outputs, targets):
                    if len(output['labels']) == 0 or len(target['labels']) == 0:
                        continue

                    pred_labels = output['labels'].cpu().numpy()
                    true_labels = target['labels'].cpu().numpy()
                    for true_label in true_labels:
                        if true_label > 0 and true_label in pred_labels:
                            total_correct += 1
                        if true_label > 0:
                            total_samples += 1

        lr_scheduler.step()
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * total_correct / total_samples if total_samples > 0 else 0.0

        epoch_losses.append(avg_loss)
        epoch_accuracies.append(accuracy)

        print(f"✅ Epoch [{epoch+1}] Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

    # Plotting loss and accuracy
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs+1), epoch_losses, marker='o')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs+1), epoch_accuracies, marker='o', color='green')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid()

    plt.tight_layout()
    plt.show()
    plt.savefig("training_metrics.png")
    print("📊 Saved training graph as training_metrics.png")

    return model

if __name__ == '__main__':
    dataset_path = r"accident_detection.v2i.coco\train"
    train_dataset = CustomDataset(dataset_path, transform=get_transform(train=True))

    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        if len(batch) == 0:
            return None
        return tuple(zip(*batch))

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_classes = train_dataset.next_class_id
    print(f"🔄 Number of classes: {num_classes} (including background)")

    model = get_model(num_classes).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    trained_model = train(model, train_loader, optimizer, lr_scheduler, num_epochs=10)

    torch.save(trained_model.state_dict(), "faster_rcnn_trained_final_1.pth")
    print("✅ Model saved as faster_rcnn_trained_final_1.pth")
