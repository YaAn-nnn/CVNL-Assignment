import torch
import torch.nn as nn
from torchvision import models, transforms

class FamilyPredictor:
    def __init__(self, ckpt_path: str, families_txt_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load family labels
        with open(families_txt_path, "r", encoding="utf-8") as f:
            families = [ln.strip() for ln in f if ln.strip()]
        self.id_to_family = {i: fam for i, fam in enumerate(families)}
        num_classes = len(families)

        # Build ConvNeXt Tiny (MUST match training)
        self.model = models.convnext_tiny(weights=None)
        in_features = self.model.classifier[2].in_features
        self.model.classifier[2] = nn.Linear(in_features, num_classes)
        self.model.to(self.device)

        # Load trained weights
        state = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

        # EXACT eval transforms you used
        self.transform = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])

    @torch.no_grad()
    def predict(self, pil_image):
        x = self.transform(pil_image.convert("RGB")).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        label = self.id_to_family[int(pred.item())]
        return label, float(conf.item())
