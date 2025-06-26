import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from emotion_dataset import FER2013Dataset
from emotion_model import EmotionEfficientNet
import os
import numpy as np
import pandas as pd

def compute_class_weights(csv_path, label_map, device):
    df = pd.read_csv(csv_path)
    counts = df['emotion'].map(label_map).value_counts().sort_index()
    weights = 1.0 / counts
    weights = weights / weights.sum()
    return torch.FloatTensor(weights.values).to(device)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    label_map = {
        "angry": 0,
        "disgust": 1,
        "fear": 2,
        "happy": 3,
        "sad": 4,
        "surprise": 5,
        "neutral": 6
    }

    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # EfficientNet expects 3 channels
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet mean/std
                             [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_dataset = FER2013Dataset("data/fer2013/train.csv", transform=train_transform)
    val_dataset = FER2013Dataset("data/fer2013/test.csv", transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    model = EmotionEfficientNet(num_classes=7).to(device)

    class_weights = compute_class_weights("data/fer2013/train.csv", label_map, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    best_acc = 0
    all_val_preds_final = []
    all_val_labels_final = []

    for epoch in range(1, 21):
        model.train()
        running_loss = 0
        total_batches = len(train_loader)
        print(f"Starting epoch {epoch}/20")
        for batch_idx, (images, labels) in enumerate(train_loader, start=1):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if batch_idx % 50 == 0 or batch_idx == total_batches:
                print(f"Epoch {epoch} - Batch {batch_idx}/{total_batches} - Loss: {loss.item():.4f}")

        avg_loss = running_loss / total_batches
        print(f"Epoch {epoch} completed. Average training loss: {avg_loss:.4f}")

        # Validation
        print("Starting validation...")
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
                all_probs.extend(probs)

        acc = accuracy_score(all_labels, all_preds)
        try:
            auc = roc_auc_score(
                np.eye(len(np.unique(all_labels)))[all_labels],
                np.array(all_probs),
                multi_class='ovr'
            )
        except Exception as e:
            print(f"Warning: Could not calculate AUC this epoch due to: {e}")
            auc = float('nan')

        print(f"Validation accuracy after epoch {epoch}: {acc:.4f}")
        print(f"Validation multiclass AUC after epoch {epoch}: {auc:.4f}")

        scheduler.step()

        if acc > best_acc:
            best_acc = acc
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/emotion_effnet_model.pt")
            print(f"Model saved with improved accuracy: {best_acc:.4f}")

        all_val_preds_final.extend(all_preds)
        all_val_labels_final.extend(all_labels)

    # Final reports
    print("\nFinal classification report on validation set:")
    print(classification_report(all_val_labels_final, all_val_preds_final, digits=4, zero_division=0))

    print("Final confusion matrix on validation set:")
    cm = confusion_matrix(all_val_labels_final, all_val_preds_final)
    print(cm)

if __name__ == "__main__":
    train()
