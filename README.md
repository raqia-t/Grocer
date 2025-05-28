#**🛒 Grocery Item Detection & Counting with YOLOv8, by Shaheryar Shakeel, Raqia Tauqir, Ahmed Soban.
This project fine-tunes a YOLOv8 object detection model on a custom grocery dataset and includes video inference with real-time object counting and cost calculation.**

**📦 Features**

    Fine-tunes YOLOv8 (yolov8n.pt) on custom grocery dataset.

    Performs inference on single images and videos.

    Tracks individual objects and counts them using track IDs.

    Calculates a total cost for detected items.

    Outputs an annotated video with counts and pricing.


#**📂 Dataset (via Roboflow)**

The custom dataset includes three classes: ketchup, masala, and soap.

**🔽 Download from Roboflow**

Visit the dataset page:

    Roboflow Dataset URL

    Click Download Dataset > Choose YOLOv5 PyTorch format.

    Unzip and structure it like this in your Google Drive:

/MyDrive/

└── Data/
└── grocery_yolov8/

    ├── train/
    
    │   ├── images/
    
    │   └── labels/
    
    ├── valid/
    
    │   ├── images/
    
    │   └── labels/
    
    ├── test/
    
    │   ├── images/
    
    │   └── labels/
    
    └── grocery_yolov8/
    
        └── data.yaml 
        

The data.yaml file should look like:

train: ../train/images val: ../valid/images test: ../test/images

nc: 3 names: ['ketchup', 'masala', 'soap']

**⚙️ Installation**

#**📒 Requirements (Python or Jupyter)**

Install necessary libraries:

pip install ultralytics opencv-python

For Google Colab:

!pip install ultralytics from google.colab import drive drive.mount('/content/drive')

#**🚀 Running the Project**

**✅ Step 1: Fine-Tune YOLOv8**

Update the path to your data.yaml:

data_config = '/content/drive/MyDrive/Data/grocery_yolov8/grocery_yolov8/data.yaml'

Train the model:

from ultralytics import YOLO model = YOLO('yolov8n.pt')

results = model.train( data=data_config, epochs=57, imgsz=640, batch=8, device=0, # or 'cpu' workers=4 )

**✅ Step 2: Validate**

val_results = model.val() print("Validation results:", val_results)

**✅ Step 3: Inference on Images**

model = YOLO('/content/runs/detect/train2/weights/best.pt')

inference_results = model.predict( source='/content/drive/MyDrive/Data/grocery_yolov8/grocery_yolov8/test/images/sample.jpg', conf=0.8, iou=0.6, save=True, project='inference_results', name='detected' )

**✅ Step 4: Inference on Videos**

Edit the following Python script (video_infer.py) or run in Jupyter:

python video_infer.py

This will:


    Load your fine-tuned model.

    Run inference on soap_infer.mp4.

    Track and count objects.

    Overlay total cost.

    Save annotated output to output_infer.mp4.


#**💰 Item Pricing (Editable)**

The script includes cost logic:

item_costs = { "soap": 70, "ketchup": 250, "masala": 100 }

You can adjust the pricing in video_infer.py as needed. 🧪 Notes

    Best run on GPU (Google Colab or local CUDA-enabled machines).

    YOLOv8 is from Ultralytics.

    Roboflow license: CC BY 4.0.

#**📜 License**

This project uses the CC BY 4.0 license per the dataset from Roboflow. By Shaheryar Shakeel, Raqia Tauqir, Ahmed Soban. Feel free to fork and experiment, but give credit where due!


