import torch
import cv2
import numpy as np
from torchvision import transforms
from emotion_model import EmotionEfficientNet  # import your model class
import torch.nn.functional as F

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionEfficientNet(num_classes=7).to(device)
model.load_state_dict(torch.load("models/emotion_effnet_model.pt", map_location=device))
model.eval()
print("✅ Model loaded")

# Define label map
label_map = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral"
}

# Preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open webcam")
    exit()

print("🎥 Webcam started. Press 'q' to quit.")

with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break

        # Convert to grayscale and detect face using HaarCascade
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi = frame[y:y+h, x:x+w]
            input_tensor = transform(roi).unsqueeze(0).to(device)
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            label = label_map[pred]

            # Draw box and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (255, 255, 255), 2)

        cv2.imshow("Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
