# **🛒 Retail Store Visitor Analytics & Tracking System**

An end-to-end computer vision pipeline built with the latest YOLO framework to monitor, track, and analyze human traffic in a retail environment.

This project uses a custom-trained object detection model to distinguish between **Customers** and **Shopkeepers**, applies an advanced object tracking algorithm (BoT-SORT), and utilizes a **Custom Post-Tracker Memory Manager** to solve common ID-switching issues, generating highly accurate dwell times and footfall counts.

## **📸 Live Dashboard Preview**

<p align="center">
  <img src="vision_retail_shop_analytics_screenshot.png" alt="Retail Store Visitor Analytics Dashboard" width="100%">
</p>

<p align="center">
<b>Real-time customer and shopkeeper tracking with persistent IDs, dwell-time analytics, live occupancy, and unique visitor counting.</b>
</p>

**What you are seeing in the tracking output:**

* **🔴 Red Bounding Boxes:** Securely tracked **Staff/Shopkeepers** (e.g., Staff 1, Staff 2\) equipped with an extended memory buffer so they aren't lost when bending behind counters.  
* **🟢 Green Bounding Boxes:** Tracked **Customers** assigned with unique persistent IDs and real-time dwell-time counters (e.g., Cust 3 | 2.1s).  
* **📊 Analytics HUD (Top-Left):** A live text overlay displaying Active Inside (current physical occupancy) and Total Unique Count (overall cumulative footfall for the session).

## **✨ Features**

* **Custom Object Detection:** Fine-tuned YOLO model capable of distinctly identifying 'Customers' and 'Shopkeepers'.  
* **Advanced Re-Identification (ReID):** Integrates BoT-SORT tracking with a bespoke Python memory manager to handle occlusions (e.g., customers walking behind shelves) without inflating ID counts.  
* **Class Locking:** Prevents "flickering" classifications (e.g., a customer being briefly tagged as a shopkeeper) by locking in historical identity profiles.  
* **Noise Filtering:** Automatically filters out "ghost" tracks (shadows, reflections) by enforcing minimum dwell-time thresholds (e.g., \> 5 seconds).  
* **Live Video Dashboard:** Generates an MP4 output with live bounding boxes, unique IDs, dwell timers, and global store population counters.  
* **Automated Reporting:** Exports a detailed .txt analytics report summarizing total unique footfall, staff presence, and individual customer dwell times.

## **🏗️ Architecture & Pipeline**

1. **Data Annotation:** Initial shopkeeper bounding boxes were drawn using CVAT.  
2. **Auto-Labeling:** A hybrid auto-labeling script utilized a base YOLO model to dynamically find and label the remaining 'customers' in the training frames to prevent background-error penalization.  
3. **Training:** Transfer learning applied to train a custom weights file (best.pt).  
4. **Inference & Tracking:** Video footage is processed frame-by-frame. YOLO predictions are fed into BoT-SORT, and then strictly filtered through our custom yolo\_to\_true\_id dictionary logic to ensure absolute ID stability.

## **🚀 Installation**

1. Clone this repository:  
   git clone \[https://github.com/nikhillpaul9/vision-retail-analytics-using-yolo26.git]  
   cd retail-analytics-tracking

2. Install the required Python dependencies:  
   *(Requires Python 3.8+)*  
   pip install \-r requirements.txt

   *Core dependencies include ultralytics, opencv-python, and numpy.*  
3. Add your model and video files:  
   * Place your custom-trained YOLO weights at: runs/detect/yolo26\_shop\_model/weights/best.pt  
   * Place your source CCTV/Video footage in the root directory and name it shop\_footage.mp4.

## **💻 Usage**

Run the main tracker script from your terminal:

python run\_retail\_tracker.py

### **Outputs Generated:**

1. **tracked\_shop\_analytics.mp4**: A video file visualizing the live AI tracking. Shopkeepers are boxed in red, Customers in green.  
2. **store\_analytics\_report.txt**: A text summary of the session. Example output:  
   \=========================================  
         RETAIL STORE VISITOR REPORT          
         Generated on: 2026-07-11 12:00:00  
   \=========================================

   Total Unique Customers Counted: 10  
   Total Distinct Shopkeepers Identified: 2  
   Average Customer Dwell Time: 69.87 seconds  
   Maximum Single Customer Stay: 110.82 seconds  
   \*(Filtered noisy tracks \- Cust \< 5.0s | Staff \< 20.0s)\*

   \--- Detailed Breakdowns \---  
    \-\> Customer ID 1: Stayed for 110.8 seconds  
    \-\> Customer ID 2: Stayed for 97.8 seconds  
   ...

## **⚙️ Configuration (Tuning the Tracker)**

You can adjust the tracking memory and sensitivity inside run\_retail\_tracker.py to better fit the layout and camera angle of your specific store.

Locate the THRESHOLDS dictionary in the code:

THRESHOLDS \= {  
    'customer': {'max\_lost\_frames': 150, 'max\_pixel\_move': 200},    \# 5 seconds of memory buffer  
    'shopkeeper': {'max\_lost\_frames': 999999, 'max\_pixel\_move': 9999} \# Infinite memory  
}

* max\_lost\_frames: How many frames a person can be hidden from the camera before the tracker considers them "gone".  
* max\_pixel\_move: The maximum distance (in pixels) a person can move while hidden and still be recognized as the same person when they reappear.

You can also adjust the noise filters at the bottom of the script:

MIN\_CUST\_STAY \= 5.0   \# Ignore customers who exist for less than 5 seconds  
MIN\_STAFF\_STAY \= 20.0 \# Ignore staff who exist for less than 20 seconds

## **🤝 Contributing**

Contributions, issues, and feature requests are welcome\! Feel free to check the issues page.