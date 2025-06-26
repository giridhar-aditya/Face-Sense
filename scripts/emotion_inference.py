import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pandas as pd
import numpy as np
from emotion_dataset import FER2013Dataset
from emotion_model import EmotionEfficientNet

def load_model(model_path, device):
    model = EmotionEfficientNet(num_classes=7)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def run_inference(model, dataloader, device):
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Load test dataset
    test_csv = "data/fer2013/test.csv"
    test_dataset = FER2013Dataset(test_csv, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # Load model
    model_path = "models/emotion_effnet_model.pt"
    model = load_model(model_path, device)

    # Run inference
    preds, labels, probs = run_inference(model, test_loader, device)

    # Print predictions for first 10 samples as example
    print("Sample predictions:", preds[:10])

    # Classification report
    print("\nClassification report:")
    print(classification_report(labels, preds, digits=4, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    print("\nConfusion matrix:")
    print(cm)

    # Compute multiclass AUC
    try:
        auc = roc_auc_score(
            np.eye(len(np.unique(labels)))[labels],
            probs,
            multi_class='ovr'
        )
        print(f"\nMulticlass AUC: {auc:.4f}")
    except Exception as e:
        print(f"Could not compute AUC: {e}")

if __name__ == "__main__":
    main()
