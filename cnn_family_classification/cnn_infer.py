import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# --------------------------------------------------
# Model architecture (MATCHES Yoshi training code)
# --------------------------------------------------
class AircraftCNN_Bigger(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        def block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3, 64),
            block(64, 128),
            block(128, 256),
            block(256, 512),
            block(512, 512),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.6),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)

# --------------------------------------------------
# Load model once (GLOBAL)
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CKPT_PATH = "cnn_family_classification/best_cnn_family_stage2_384.pt"
ckpt = torch.load(CKPT_PATH, map_location=device)

class_to_idx = ckpt["class_to_idx"]
idx_to_class = {v: k for k, v in class_to_idx.items()}
img_size = ckpt["img_size"]

model = AircraftCNN_Bigger(len(class_to_idx))
model.load_state_dict(ckpt["model_state_dict"])
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# --------------------------------------------------
# Inference function used by app.py
# --------------------------------------------------
def predict_aircraft_family(image: Image.Image):
    image = image.convert("RGB")
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        confidence = probs[0, pred_idx].item()

    return idx_to_class[pred_idx], confidence
