from ultralytics import YOLO


def main():

    # Load pretrained YOLOv8 model
    model = YOLO("yolov8m.pt")

    # Train model
    model.train(
        data="../datasets/rickshaw_dataset/data.yaml",
        epochs=80,
        imgsz=640,
        batch=16,
        device=0,
        project="models",
        name="rickshaw_detector"
    )


if __name__ == "__main__":
    main()