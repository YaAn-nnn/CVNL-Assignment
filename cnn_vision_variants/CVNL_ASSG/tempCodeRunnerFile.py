
BEST_MODEL_PATH = os.path.join(BASE_PATH, "airline_model_best.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"SYSTEM STATUS: Training on {device.type.upper()}")

# ==========================================
# 2. dataset class
# ==========================================
class AirlineIDDataset(Dataset):