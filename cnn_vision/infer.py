class VisionPredictor:
    def __init__(self, ckpt_path: str):
        self.ckpt_path = ckpt_path

    def predict(self, pil_image):
        return "CNN_NOT_READY"