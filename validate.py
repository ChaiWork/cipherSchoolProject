from ultralytics import YOLO
import torch

def validate_model():

    # Clear GPU cache
    torch.cuda.empty_cache()

    # Load trained model
    model = YOLO(
        r"load ur own model"
    )

    # Validate model
    metrics = model.val(
        data=r"load ur own config path",
        batch=1,
        imgsz=640,
        device=0,
        workers=0
    )

    # Print metrics
    print("Precision:", metrics.box.mp)
    print("Recall:", metrics.box.mr)
    print("mAP50:", metrics.box.map50)
    print("mAP50-95:", metrics.box.map)

# IMPORTANT FOR WINDOWS
if __name__ == "__main__":
    validate_model()