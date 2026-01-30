import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

class VisionPredictor:
    def __init__(self, ckpt_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the checkpoint using secure mode
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        self.classes = checkpoint['classes']
        
        # Rebuild ResNet-18 architecture
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, len(self.classes))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Use the same 448px resolution from your training
        self.transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, pil_image):
        image = pil_image.convert("RGB")
        image = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image)
            _, predicted = torch.max(outputs, 1)
            
        return self.classes[predicted.item()]