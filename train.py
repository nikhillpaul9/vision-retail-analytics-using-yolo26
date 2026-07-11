from ultralytics import YOLO

model = YOLO("yolo26n.pt") 

# Train using custom dataset configuration
model.train(
    data="dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    optimizer="MuSGD",  # Uses YOLO26's modern hybrid optimizer architecture
    name="yolo26_shop_model"
)