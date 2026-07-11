import json
from ultralytics import YOLO

def run_evaluation():
    print("Loading custom YOLO26 model...")
    # 1. Load the 'best.pt' weights from your training run
    model = YOLO("runs/detect/yolo26_shop_model/weights/best.pt")

    print("Running validation...")
    # 2. Run the validation process on the validation dataset
    metrics = model.val(
        data="dataset.yaml",
        split="val",                 # Evaluates on your val/images folder
        project="shop_metrics",      # Custom main folder for output
        name="yolo26_evaluation",    # Custom subfolder for this run
        save_json=True               # Saves standard COCO format predictions
    )

    # 3. Extract the core metrics from the returned object
    # Ultralytics metrics objects store bounding box metrics under the `.box` attribute
    performance_data = {
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "mean_precision": float(metrics.box.mp),
        "mean_recall": float(metrics.box.mr),
        # gets per-class mAP (Class 0: Shopkeeper, Class 1: Customer)
        "per_class_mAP50_95": metrics.box.maps.tolist() 
    }

    # 4. Save out to a custom JSON file for your records
    with open("shop_model_performance.json", "w") as f:
        json.dump(performance_data, f, indent=4)

    # 5. Print a quick summary to the terminal
    print("\n--- Evaluation Complete ---")
    print(f"Mean Precision: {performance_data['mean_precision']:.3f}")
    print(f"Mean Recall:    {performance_data['mean_recall']:.3f}")
    print(f"mAP@50:         {performance_data['mAP50']:.3f}")
    print(f"mAP@50-95:      {performance_data['mAP50_95']:.3f}")
    
    print("\nDetailed visual charts saved to: shop_metrics/yolo26_evaluation/")
    print("Core numerical metrics saved to: shop_model_performance.json")

if __name__ == "__main__":
    run_evaluation()