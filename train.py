"""Training script (the web app's Train Model page runs this for you; you can also run it by hand).
  python train.py --data path/to/data.yaml --epochs 80
  python train.py --roboflow API_KEY WORKSPACE PROJECT VERSION
Saves models/best.pt and models/metrics.json (REAL metrics from your validation set)."""
import argparse, json, os, shutil, datetime
from pathlib import Path
p = argparse.ArgumentParser(); p.add_argument("--data"); p.add_argument("--roboflow", nargs=4, metavar=("KEY", "WORKSPACE", "PROJECT", "VERSION"))
p.add_argument("--rf", nargs=3, metavar=("WORKSPACE", "PROJECT", "VERSION"), help="key read from env ROBOFLOW_API_KEY")
p.add_argument("--epochs", type=int, default=60); p.add_argument("--imgsz", type=int, default=640); p.add_argument("--model", default="yolov8n.pt")
a = p.parse_args()
if a.roboflow or a.rf:
    from roboflow import Roboflow
    k, w, pr, v = a.roboflow if a.roboflow else (os.environ["ROBOFLOW_API_KEY"], *a.rf)
    a.data = Roboflow(api_key=k).workspace(w).project(pr).version(int(v)).download("yolov8").location + "/data.yaml"
if not a.data: p.error("give --data, --rf or --roboflow")
import yaml
def fix(y):   # make dataset paths absolute and correct (Roboflow zips use ../ paths)
    d = yaml.safe_load(open(y)); root = Path(y).parent.resolve(); d["path"] = str(root)
    for key, alts in (("train", ["train/images", "images/train"]), ("val", ["valid/images", "val/images", "images/val"])):
        for c in [str(d.get(key, "")).replace("../", "")] + alts:
            if c and (root / c).exists(): d[key] = c; break
    out = root / "data_fixed.yaml"; yaml.safe_dump(d, open(out, "w")); return str(out)
from ultralytics import YOLO
m = YOLO(a.model); m.train(data=fix(a.data), epochs=a.epochs, imgsz=a.imgsz, project="runs", name="powerline", exist_ok=True)
r = m.val(); Path("models").mkdir(exist_ok=True); shutil.copy(m.trainer.best, "models/best.pt")
json.dump(dict(map50=round(float(r.box.map50), 3), map50_95=round(float(r.box.map), 3), classes=list(m.names.values()), epochs=a.epochs,
               trained_at=datetime.datetime.now().isoformat(timespec="seconds")), open("models/metrics.json", "w"), indent=1)
print("DONE. Saved models/best.pt and models/metrics.json")
