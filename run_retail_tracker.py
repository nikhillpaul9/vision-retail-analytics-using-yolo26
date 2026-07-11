import os
import time
import cv2
import numpy as np
from ultralytics import YOLO

def main():
    model_path = 'runs/detect/yolo26_shop_model/weights/best.pt'
    if not os.path.exists(model_path):
        print(f"Error: Custom weights not found at {model_path}.")
        return

    model = YOLO(model_path) 
    
    video_path = 'shop_footage.mp4' 
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    output_video_path = 'tracked_shop_analytics.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # --- NEW: Custom Post-Tracker Memory Manager ---
    # This bypasses the YOLO ID switching bug entirely
    yolo_to_true_id = {}
    true_id_profiles = {}
    next_true_customer_id = 1
    next_true_shopkeeper_id = 1
    
    # Class-specific memory thresholds
    THRESHOLDS = {
        'customer': {'max_lost_frames': 150, 'max_pixel_move': 200},  # 5 seconds memory
        'shopkeeper': {'max_lost_frames': 1800, 'max_pixel_move': 9999} # 60s memory, anywhere on screen for staff
    }

    def get_center(box):
        return ((box[0]+box[2])/2, (box[1]+box[3])/2)

    def dist(c1, c2):
        return ((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)**0.5
    # -----------------------------------------------

    customer_entry_frames = {}
    customer_time_spent = {}
    total_unique_customers = set()
    total_unique_shopkeepers = set()

    print("Processing video with Custom Post-Tracker Memory...")
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Increased conf=0.35 to kill noisy ghost detections
        results = model.track(
            frame, 
            persist=True, 
            tracker="botsort.yaml", 
            conf=0.35,  
            iou=0.4,
            verbose=False
        )
        
        current_customers_in_shop = set()
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            
            for box, yolo_id, class_id in zip(boxes, track_ids, class_ids):
                x1, y1, x2, y2 = map(int, box)
                label = model.names[class_id] 
                center = get_center(box)
                
                # --- Custom ID Matching Logic ---
                if yolo_id not in yolo_to_true_id:
                    matched_true_id = None
                    min_dist = float('inf')
                    
                    max_lost = THRESHOLDS.get(label, THRESHOLDS['customer'])['max_lost_frames']
                    max_move = THRESHOLDS.get(label, THRESHOLDS['customer'])['max_pixel_move']
                    
                    for true_id, profile in true_id_profiles.items():
                        # Match if it's the same class & they vanished within their specific threshold
                        if profile['class'] == label and (frame_count - profile['last_frame']) < max_lost:
                            d = dist(center, profile['last_center'])
                            if d < max_move and d < min_dist:
                                min_dist = d
                                matched_true_id = true_id
                    
                    if matched_true_id is not None:
                        yolo_to_true_id[yolo_id] = matched_true_id
                    else:
                        # Assign brand new stable ID
                        if label == 'customer':
                            yolo_to_true_id[yolo_id] = next_true_customer_id
                            next_true_customer_id += 1
                        else:
                            yolo_to_true_id[yolo_id] = next_true_shopkeeper_id
                            next_true_shopkeeper_id += 1
                            
                true_id = yolo_to_true_id[yolo_id]
                
                # Keep their profile updated with their newest location and timestamp
                true_id_profiles[true_id] = {
                    'class': label,
                    'last_frame': frame_count,
                    'last_center': center
                }
                # --------------------------------
                
                if label == 'customer':
                    current_customers_in_shop.add(true_id)
                    total_unique_customers.add(true_id)
                    
                    if true_id not in customer_entry_frames:
                        customer_entry_frames[true_id] = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    
                    frames_spent = cap.get(cv2.CAP_PROP_POS_FRAMES) - customer_entry_frames[true_id]
                    time_spent_sec = frames_spent / fps
                    customer_time_spent[true_id] = time_spent_sec
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Cust {true_id} | {time_spent_sec:.1f}s", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                
                elif label == 'shopkeeper':
                    total_unique_shopkeepers.add(true_id)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"Staff {true_id}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.putText(frame, f"Active Inside: {len(current_customers_in_shop)}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        # Dynamic active count (doesn't apply the 5-second rule live, but helps visualize live tracker IDs)
        cv2.putText(frame, f"Total Unique Count: {len(total_unique_customers)}", (20, 75), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        out.write(frame)
        
        cv2.imshow("Processing Feed...", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # 5. Clean up data: Filter out tiny ghost tracks (under 5 seconds)
    MIN_STAY_SECONDS = 5.0
    valid_customers = {cid: duration for cid, duration in customer_time_spent.items() if duration >= MIN_STAY_SECONDS}
    
    dwell_times = list(valid_customers.values())
    avg_time_spent = np.mean(dwell_times) if dwell_times else 0.0
    max_time_spent = np.max(dwell_times) if dwell_times else 0.0
    
    summary_txt_path = 'store_analytics_report.txt'
    with open(summary_txt_path, 'w') as f:
        f.write("=========================================\n")
        f.write("      RETAIL STORE VISITOR REPORT        \n")
        f.write(f"      Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=========================================\n\n")
        f.write(f"Total Unique Customers Counted: {len(valid_customers)}\n")
        f.write(f"Total Distinct Shopkeepers Identified: {len(total_unique_shopkeepers)}\n")
        f.write(f"Average Customer Dwell Time: {avg_time_spent:.2f} seconds\n")
        f.write(f"Maximum Single Customer Stay: {max_time_spent:.2f} seconds\n")
        f.write(f"*(Filtered out noisy tracks under {MIN_STAY_SECONDS}s)*\n\n")
        f.write("--- Detailed Breakdowns ---\n")
        if valid_customers:
            for cust_id, duration in valid_customers.items():
                f.write(f" -> Customer ID {cust_id}: Stayed for {duration:.1f} seconds\n")
        else:
            f.write(" No valid customer dwell-times logged.\n")
        f.write("\n=========================================\n")

if __name__ == "__main__":
    main()