# Training (inside the app)
Run the server WITHOUT --reload while installing or training:  `uvicorn main:app`
Open **Train Model** in the sidebar:
1. Install training packages (one click, large download).
2. Upload a YOLO-format dataset .zip (or enter Roboflow details). Get a licensed dataset from universe.roboflow.com ("insulator defect", "power line", "transmission tower") or label your own drone photos.
3. Start training. The log updates live. When it finishes, the model is installed automatically and its real mAP50 is shown. No restart needed.
A GPU is strongly recommended. Command-line alternative: `python train.py --data path/to/data.yaml`.
