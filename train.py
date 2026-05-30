from ultralytics import YOLO

def train_model():
    model = YOLO("yolo26m.pt")

    results = model.train(
        data="ur own config path",

        epochs=200,
        imgsz=640,        # better balance than 576
        batch=12,         # safer for RTX 4030
        device=0,

        optimizer="AdamW",
        lr0=0.001,
        weight_decay=0.01,
        warmup_epochs=3,
        patience=40,
        workers=4,
        cache='disk',     # faster + stable

        amp=True,

        mosaic=0.5,
        mixup=0.0,
        copy_paste=0.0,

        # light realism augmentation (important)
        degrees=2,
        translate=0.05,
        scale=0.25,

        close_mosaic=5,

        project="license_plate_detection",
        name="yolo26m_balanced"
    )

if __name__ == "__main__":
    train_model()