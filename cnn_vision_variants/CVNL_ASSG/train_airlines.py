import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ==========================================
# 1. local path configuration
# ==========================================
BASE_PATH = r'C:\Users\jayde\Downloads\CVNL_ASSG\fgvc-aircraft-2013b\data' 
IMAGES_DIR = os.path.join(BASE_PATH, 'images_448')
BEST_MODEL_PATH = os.path.join(BASE_PATH, "airline_model_best.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"SYSTEM STATUS: Training on {device.type.upper()}")

# ==========================================
# 2. dataset class
# ==========================================
class AirlineIDDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_labels = []
        with open(annotation_file, 'r') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    self.img_labels.append((parts[0], parts[1]))
        unique_labels = sorted(list(set([label for _, label in self.img_labels])))
        self.label_to_idx = {label: i for i, label in enumerate(unique_labels)}
        self.classes = unique_labels

    def __len__(self): return len(self.img_labels)

    def __getitem__(self, idx):
        img_id, label_name = self.img_labels[idx]
        img_path = os.path.join(self.root_dir, f"{img_id}.jpg")
        image = Image.open(img_path).convert('RGB')
        label = self.label_to_idx[label_name]
        if self.transform: image = self.transform(image)
        return image, label

# ==========================================
# 3. training engine
# ==========================================
def train_model():
    history = {'train_loss': [], 'val_acc': []}

    # added augmentation to force model to learn better
    my_transforms = transforms.Compose([
    # No need for Resize((448, 448)) anymore (resize_dataset_448.py does the work)
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

    train_dataset = AirlineIDDataset(IMAGES_DIR, os.path.join(BASE_PATH, 'images_variant_train.txt'), my_transforms)
    val_dataset = AirlineIDDataset(IMAGES_DIR, os.path.join(BASE_PATH, 'images_variant_val.txt'), my_transforms)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    model = models.resnet18(weights='DEFAULT')
    model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes)) 
    model = model.to(device)

    # lowered learning rate for finer adjustments (was originally 0.001)
    optimizer = optim.Adam(model.parameters(), lr=0.00000001) 
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # it monitors 'acc' and cuts LR by half if it doesn't improve for 2 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    # resume logic: loads the best score to prevent overwriting
    best_acc = 0.0
    if os.path.exists(BEST_MODEL_PATH):
        print("🔍 existing best model detected. loading previous record...")
        checkpoint = torch.load(BEST_MODEL_PATH, weights_only=True)
        best_acc = checkpoint.get('accuracy', 0.0)
        model.load_state_dict(checkpoint['model_state_dict']) 
        print(f"✅ current record to beat: {best_acc:.2f}%")

    # training loop
    epochs_without_improvement = 0
    for epoch in range(50): 
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        history['train_loss'].append(running_loss / len(train_loader))

        model.eval()
        correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                _, preds = torch.max(model(inputs), 1)
                correct += torch.sum(preds == labels.data)
        
        acc = 100 * correct.double() / len(val_dataset)
        history['val_acc'].append(acc.item())
        scheduler.step(acc)
        print(f"Epoch {epoch+1} Complete | Accuracy: {acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            epochs_without_improvement = 0
            torch.save({'model_state_dict': model.state_dict(), 'accuracy': acc.item(), 'classes': train_dataset.classes}, BEST_MODEL_PATH)
            print(f"✅ NEW BEST! Saved to: {BEST_MODEL_PATH}")
        else:
            epochs_without_improvement += 1

        # early stopping: stop if no improvement after 7 epochs
        if epochs_without_improvement >= 15:
            print(f"🛑 early stopping triggered after {epoch+1} epochs.")
            break

    # ==========================================
    # 4. visualizations for report
    # ==========================================
    # generate graphs and matrix automatically
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Val Accuracy', color='orange')
    plt.title('Validation Accuracy Curve')
    plt.legend()
    plt.savefig(os.path.join(BASE_PATH, 'training_curves.png'))
    plt.show()

    print(f"\n--- FINAL PROTOTYPE RESULTS ---")
    print(f"Best Accuracy Achieved: {best_acc:.2f}%")
    print(f"Saved File Path: {BEST_MODEL_PATH}")

if __name__ == "__main__":
    train_model()