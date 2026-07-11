import os
from ultralytics import YOLO

# Load the latest YOLO26 nano base model
base_model = YOLO("yolo26n.pt") 

# Training Set
# IMAGE_DIR = "./my_shop_dataset/train/images"
# LABEL_DIR = "./my_shop_dataset/train/labels"

# Validation Set 
IMAGE_DIR = "./my_shop_dataset/val/images"
LABEL_DIR = "./my_shop_dataset/val/labels"

# Loop through and append customer mappings programmatically
for img_name in os.listdir(IMAGE_DIR):
    if not img_name.endswith(('.jpg', '.jpeg', '.png')):
        continue
        
    img_path = os.path.join(IMAGE_DIR, img_name)
    label_path = os.path.join(LABEL_DIR, os.path.splitext(img_name)[0] + ".txt")
    
    shopkeeper_boxes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = list(map(float, line.strip().split()))
                if int(parts[0]) == 0: 
                    shopkeeper_boxes.append(parts[1:]) 
                    
    results = base_model(img_path, verbose=False)
    new_labels = []
    h_img, w_img = results[0].orig_shape
    
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        if cls_id == 0: # Base COCO class 0 is 'person'
            xyxy = box.xyxy[0].tolist()
            x_cnt = ((xyxy[0] + xyxy[2]) / 2) / w_img
            y_cnt = ((xyxy[1] + xyxy[3]) / 2) / h_img
            w_box = (xyxy[2] - xyxy[0]) / w_img
            h_box = (xyxy[3] - xyxy[1]) / h_img
            
            is_shopkeeper = False
            for sk_box in shopkeeper_boxes:
                if abs(x_cnt - sk_box[0]) < 0.05 and abs(y_cnt - sk_box[1]) < 0.05:
                    is_shopkeeper = True
                    break
            
            final_cls = 0 if is_shopkeeper else 1 
            new_labels.append(f"{final_cls} {x_cnt:.6f} {y_cnt:.6f} {w_box:.6f} {h_box:.6f}\n")
            
    with open(label_path, 'w') as f:
        f.writelines(new_labels)

print("Dataset successfully updated using YOLO26 targets!")