"""Plug-in for a TRAINED YOLO model. Does nothing until models/best.pt exists (see TRAINING.md)."""
from pathlib import Path
W = Path(__file__).parent / "models" / "best.pt"
# model class name -> (fault type, severity if confident, recommended action). Edit to match your dataset classes.
CLASS_MAP = {
    "broken_wire": ("Broken/disconnected wire", "HIGH", "Urgent visual verification of the conductor."),
    "dropped_wire": ("Dropped wire", "CRITICAL", "Treat as hazard: escalate per safety procedure after human verification."),
    "insulator_damage": ("Wire/insulator damage", "HIGH", "Schedule prompt inspection of the insulator."),
    "tower_damage": ("Damaged tower component", "HIGH", "Structural assessment by qualified engineers."),
    "fallen_tower": ("Fallen/displaced tower", "CRITICAL", "Escalate per safety procedure after human verification."),
    "rust": ("Rust/corrosion", "MEDIUM", "Corrosion assessment."),
}
_model = None; _mt = None
def detect(path, conf=0.35):
    global _model, _mt
    if not W.exists(): return []
    mt = W.stat().st_mtime
    if _model is None or mt != _mt:   # reloads automatically after retraining
        from ultralytics import YOLO
        _model = YOLO(str(W)); _mt = mt
    out = []
    for r in _model(str(path), conf=conf, verbose=False):
        for b in r.boxes:
            name = _model.names[int(b.cls)]
            c = float(b.conf)
            t, sev, act = CLASS_MAP.get(name, (name.replace("_", " ").title() + " (model class)", "MEDIUM", "Verify visually; classify manually."))   # unknown classes still reported
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            sure = c >= 0.6   # low confidence is never escalated: it is marked as a possible defect at MEDIUM
            out.append(dict(type=t if sure else "Possible " + t.lower(), severity=sev if sure else "MEDIUM", box=[x1, y1, x2 - x1, y2 - y1],
                method=f"YOLO trained model (confidence {c:.2f})", action=act,
                description=f"Model detected '{name}' with score {c:.2f}. The score is the model's own estimate, not a guarantee; check accuracy on your own test set."))
    return out
