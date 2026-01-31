import os
from PIL import Image
from tqdm import tqdm


BASE_PATH = r'C:\Users\jayde\Downloads\CVNL_ASSG\fgvc-aircraft-2013b\data'
ORIGINAL_IMAGES = os.path.join(BASE_PATH, 'images')
NEW_IMAGES_DIR = os.path.join(BASE_PATH, 'images_448')


if not os.path.exists(NEW_IMAGES_DIR):
    os.makedirs(NEW_IMAGES_DIR)

def preprocess_dataset():
    image_files = [f for f in os.listdir(ORIGINAL_IMAGES) if f.endswith('.jpg')]
    print(f"🚀 Found {len(image_files)} images. Starting high-speed resize...")

    for filename in tqdm(image_files):
        img_path = os.path.join(ORIGINAL_IMAGES, filename)
        save_path = os.path.join(NEW_IMAGES_DIR, filename)
        
        if os.path.exists(save_path):
            continue
            
        try:
            with Image.open(img_path) as img:
                img_resized = img.resize((448, 448), Image.Resampling.LANCZOS)
                img_resized.convert('RGB').save(save_path, 'JPEG', quality=95)
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

    print(f"✅ DONE! All images are now in: {NEW_IMAGES_DIR}")

if __name__ == "__main__":
    preprocess_dataset()