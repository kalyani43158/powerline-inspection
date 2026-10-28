import os, sys, json, uuid, threading, csv, io, shutil, zipfile, subprocess, importlib.util, urllib.request, datetime as dt
from pathlib import Path
import cv2, numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import create_engine, event, inspect, text, Column, Integer, String, Float, Text, DateTime, Boolean, ForeignKey, func
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

BASE = Path(__file__).parent; STORE = Path(os.environ.get("PL_STORE", BASE / "storage")); STORE.mkdir(exist_ok=True)
_env = BASE / ".env"
if _env.exists():
    for _l in _env.read_text().splitlines():
        if "=" in _l and not _l.strip().startswith("#"): _k, _v = _l.split("=", 1); os.environ.setdefault(_k.strip(), _v.strip())
MAX_MB, ALLOWED = 15, {".jpg", ".jpeg", ".png"}
# Configurable rules (edit here): severity from defect type + extent; alerts for these levels
RULES = {"rust_medium_area_ratio": 0.03, "alert_levels": ["CRITICAL", "HIGH"], "thermal_delta_medium": 15.0, "thermal_delta_high": 30.0, "sag_ratio": 0.06, "sag_severity": "MEDIUM", "gap_frac": 0.05, "gap_severity": "HIGH", "tilt_deg": 10.0, "tilt_severity": "MEDIUM"}
SETF = Path(os.environ.get("PL_SETTINGS", BASE / "settings.json"))
DSET = Path(os.environ.get("PL_DATASETS", BASE / "datasets"))
VERSION = "v13-2026-10-28"
SEV = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
LIMITS = "Image-based inspection cannot detect every fault. Thermal faults need thermal data. Results are heuristic candidates, not a trained ML model, and need qualified human verification before any maintenance or electrical-safety decision."

engine = create_engine(f"sqlite:///{os.environ.get('PL_DB', BASE/'app.db')}", connect_args={"check_same_thread": False})
Session = sessionmaker(engine); Base = declarative_base()
now = dt.datetime.utcnow

if SETF.exists(): RULES.update(json.loads(SETF.read_text()))

class Inspection(Base):
    __tablename__ = "inspections"
    id = Column(Integer, primary_key=True); code = Column(String, unique=True, index=True)
    name = Column(String); taken = Column(DateTime); drone_id = Column(String); tower = Column(String)
    location = Column(String); lat = Column(Float); lon = Column(Float); inspector = Column(String)
    notes = Column(Text); status = Column(String, default="completed"); created = Column(DateTime, default=now)
    images = relationship("Image", cascade="all,delete-orphan"); faults = relationship("Fault", cascade="all,delete-orphan")
class Image(Base):
    __tablename__ = "images"
    id = Column(Integer, primary_key=True); inspection_id = Column(Integer, ForeignKey("inspections.id"), index=True)
    filename = Column(String); original = Column(String); annotated = Column(String); quality = Column(String); result = Column(String)
class Fault(Base):
    __tablename__ = "faults"
    id = Column(Integer, primary_key=True); inspection_id = Column(Integer, ForeignKey("inspections.id"), index=True)
    image_id = Column(Integer, ForeignKey("images.id")); type = Column(String, index=True); severity = Column(String, index=True)
    description = Column(Text); method = Column(String); box = Column(String); action = Column(Text)
    review_status = Column(String, default="New", index=True); comment = Column(Text); created = Column(DateTime, default=now)
class Alert(Base):
    __tablename__ = "alerts"
    id = Column(Integer, primary_key=True); fault_id = Column(Integer, ForeignKey("faults.id", ondelete="CASCADE"))
    inspection_id = Column(Integer, ForeignKey("inspections.id")); severity = Column(String); message = Column(Text)
    ack = Column(Boolean, default=False); created = Column(DateTime, default=now); delivery = Column(String)
class Audit(Base):
    __tablename__ = "audit_logs"
    id = Column(Integer, primary_key=True); entity = Column(String); entity_id = Column(Integer); action = Column(String)
    old = Column(Text); new = Column(Text); at = Column(DateTime, default=now)
Base.metadata.create_all(engine)
if "delivery" not in [c["name"] for c in inspect(engine).get_columns("alerts")]:
    with engine.begin() as cx: cx.execute(text("ALTER TABLE alerts ADD COLUMN delivery VARCHAR"))

def send_external(msg):
    """Returns an honest delivery status. 'accepted' = the server answered OK; it does not prove a human read it."""
    res = []; host, to = os.environ.get("SMTP_HOST"), os.environ.get("ALERT_EMAIL_TO")
    if host and to:
        try:
            import smtplib; from email.message import EmailMessage
            e = EmailMessage(); e["Subject"] = "Power-line alert: " + msg[:60]; e["From"] = os.environ.get("SMTP_USER", to); e["To"] = to; e.set_content(msg)
            with smtplib.SMTP(host, int(os.environ.get("SMTP_PORT", 587)), timeout=8) as sm:
                sm.starttls()
                if os.environ.get("SMTP_USER"): sm.login(os.environ["SMTP_USER"], os.environ.get("SMTP_PASS", ""))
                sm.send_message(e)
            res.append("Email accepted by SMTP server")
        except Exception as x: res.append(f"Email FAILED: {x}")
    url = os.environ.get("WEBHOOK_URL")
    if url:
        try:
            rq = urllib.request.Request(url, data=json.dumps({"text": msg}).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(rq, timeout=8) as r: res.append(f"Webhook accepted (HTTP {r.status})")
        except Exception as x: res.append(f"Webhook FAILED: {x}")
    return "; ".join(res) or "In-app only (no external channel configured)"
def deliver(aid, msg):
    st = send_external(msg); s = Session(); a = s.get(Alert, aid)
    if a: a.delivery = st; s.commit()
    s.close()
@event.listens_for(Alert, "after_insert")
def _queue(mapper, conn, a):
    from sqlalchemy.orm import object_session
    object_session(a).info.setdefault("pend", []).append((a.id, a.message))
@event.listens_for(Session, "after_commit")
def _send(s):
    for i, t in s.info.pop("pend", []): threading.Thread(target=deliver, args=(i, t), daemon=True).start()


def analyze(path: Path, out: Path):
    img = cv2.imread(str(path))
    if img is None: return dict(status="failed", quality="corrupt", faults=[], summary="Unreadable image. Manual inspection required.")
    h, w = img.shape[:2]; g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur, bright = cv2.Laplacian(g, cv2.CV_64F).var(), float(g.mean())
    q = "too_small" if min(h, w) < 200 else "too_dark" if bright < 35 else "blurry" if blur < 30 else "ok"
    if q != "ok":
        cv2.imwrite(str(out), img)
        return dict(status="unable_to_determine", quality=q, faults=[], summary="Unable to determine: manual inspection required.")
    hsv = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2HSV)
    mask = cv2.morphologyEx(cv2.inRange(hsv, (5, 120, 60), (20, 255, 200)), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    faults = []
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:5]:
        ratio = cv2.contourArea(c) / (h * w)
        if ratio < 0.002: continue
        x, y, bw, bh = cv2.boundingRect(c)
        sev = "MEDIUM" if ratio >= RULES["rust_medium_area_ratio"] else "LOW"
        faults.append(dict(type="Possible rust/corrosion (candidate)", severity=sev, box=[x, y, bw, bh], method="Heuristic HSV colour threshold (not a trained model)",
            description=f"Rust-coloured region covering {ratio*100:.1f}% of the frame. Colour alone can be a false positive (soil, paint, lighting). Severity set by configurable rule: area >= {RULES['rust_medium_area_ratio']*100:.0f}% is MEDIUM, else LOW.",
            action="Verify visually; schedule corrosion assessment if confirmed."))
        cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 165, 255), 3)
        cv2.putText(img, f"{sev} rust?", (x, max(20, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    try:
        from detectors import detect_geometry
        for d in detect_geometry(g, RULES):
            x, y, bw, bh = d["box"]; faults.append(d)
            cv2.rectangle(img, (x, y), (x + bw, y + bh), (255, 80, 0), 3)
            cv2.putText(img, d["severity"] + " " + d["type"][9:30], (x, max(20, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 80, 0), 2)
    except Exception as e: print("Geometry detectors skipped:", e)
    try:
        from detector_yolo import detect  # returns [] when models/best.pt is absent
        for d in detect(path):
            x, y, bw, bh = d["box"]; faults.append(d)
            cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 0, 255), 3)
            cv2.putText(img, d["severity"] + " " + d["type"][:22], (x, max(20, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    except Exception as e: print("Trained model skipped:", e)
    cv2.imwrite(str(out), img)
    s = f"{len(faults)} candidate region(s) flagged for human review." if faults else "No visible fault detected by the available heuristics."
    return dict(status="analyzed", quality=q, faults=faults, summary=s)

app = FastAPI(title="AI-Based Power-Line Inspection with a Drone")
app.mount("/files", StaticFiles(directory=STORE), name="files")
def db(): return Session()
def iso(d): return d.isoformat() if d else None
def audit(s, ent, eid, act, old, new): s.add(Audit(entity=ent, entity_id=eid, action=act, old=str(old), new=str(new)))
def f_out(f): return dict(id=f.id, inspection_id=f.inspection_id, image_id=f.image_id, type=f.type, severity=f.severity, description=f.description, method=f.method,
    box=json.loads(f.box), action=f.action, review_status=f.review_status, comment=f.comment, created=iso(f.created))
def i_out(i, full=False):
    d = dict(id=i.id, code=i.code, name=i.name, taken=iso(i.taken), location=i.location, tower=i.tower, lat=i.lat, lon=i.lon, inspector=i.inspector, drone_id=i.drone_id,
        notes=i.notes, status=i.status, n_images=len(i.images), n_faults=len(i.faults),
        top=next((x for x in ["CRITICAL", "HIGH", "MEDIUM", "LOW"] if any(f.severity == x for f in i.faults)), "None"))
    if full:
        d["images"] = [dict(id=m.id, filename=m.filename, original=f"/files/{m.original}", annotated=f"/files/{m.annotated}", quality=m.quality, result=m.result) for m in i.images]
        d["faults"] = [f_out(f) for f in i.faults]
    return d

@app.get("/api/health")
def health(): return {"status": "ok", "version": VERSION}

@app.post("/api/inspections")
async def create(name: str = Form(...), taken: str = Form(""), drone_id: str = Form(""), tower: str = Form(""), location: str = Form(""),
                 lat: str = Form(""), lon: str = Form(""), inspector: str = Form(""), notes: str = Form(""), files: list[UploadFile] = File(...)):
    for u in files:
        if Path(u.filename or "").suffix.lower() not in ALLOWED: raise HTTPException(400, f"{u.filename}: only JPG/JPEG/PNG allowed")
    s = db()
    try:
        ins = Inspection(code="INS-" + uuid.uuid4().hex[:8].upper(), name=name, drone_id=drone_id, tower=tower, location=location, inspector=inspector, notes=notes,
            taken=dt.datetime.fromisoformat(taken) if taken else now(), lat=float(lat) if lat else None, lon=float(lon) if lon else None)
        s.add(ins); s.flush()
        for u in files:
            data = await u.read()
            if len(data) > MAX_MB * 1024 * 1024: raise HTTPException(400, f"{u.filename} exceeds {MAX_MB} MB")
            if cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) is None: raise HTTPException(400, f"{u.filename} is not a valid image")
            ext = Path(u.filename).suffix.lower(); base = uuid.uuid4().hex
            (STORE / f"{base}{ext}").write_bytes(data)
            r = analyze(STORE / f"{base}{ext}", STORE / f"{base}_annotated.png")
            im = Image(inspection_id=ins.id, filename=Path(u.filename).name, original=f"{base}{ext}", annotated=f"{base}_annotated.png", quality=r["quality"], result=r["summary"])
            s.add(im); s.flush()
            for f in r["faults"]:
                ft = Fault(inspection_id=ins.id, image_id=im.id, type=f["type"], severity=f["severity"], description=f["description"], method=f["method"], box=json.dumps(f["box"]), action=f["action"])
                s.add(ft); s.flush()
                if ft.severity in RULES["alert_levels"]: s.add(Alert(fault_id=ft.id, inspection_id=ins.id, severity=ft.severity, message=f"{ft.severity} finding: {ft.type}"))
        audit(s, "inspection", ins.id, "create", "", ins.code); s.commit(); s.refresh(ins)
        return i_out(ins, True)
    except Exception:
        s.rollback(); raise
    finally: s.close()

@app.get("/api/inspections")
def list_ins(q: str = "", severity: str = "", status: str = ""):
    s = db(); rows = s.query(Inspection).order_by(Inspection.created.desc()).all()
    out = [i_out(i) for i in rows if (not q or q.lower() in (i.name + i.code + (i.location or "")).lower()) and (not severity or i_out(i)["top"] == severity)]
    s.close(); return out

@app.get("/api/inspections/{iid}")
def get_ins(iid: int):
    s = db(); i = s.get(Inspection, iid)
    if not i: raise HTTPException(404, "Not found")
    r = i_out(i, True); s.close(); return r

@app.delete("/api/inspections/{iid}")
def del_ins(iid: int):
    s = db(); i = s.get(Inspection, iid)
    if not i: raise HTTPException(404, "Not found")
    s.query(Alert).filter(Alert.inspection_id == iid).delete(); audit(s, "inspection", iid, "delete", i.code, "")
    s.delete(i); s.commit(); s.close(); return {"deleted": iid}

@app.get("/api/dashboard")
def dashboard():
    s = db(); sev = dict(s.query(Fault.severity, func.count()).group_by(Fault.severity).all())
    types = dict(s.query(Fault.type, func.count()).group_by(Fault.type).all())
    days = s.query(func.date(Inspection.created), func.count()).group_by(func.date(Inspection.created)).order_by(func.date(Inspection.created)).all()
    r = dict(inspections=s.query(Inspection).count(), images=s.query(Image).count(), faults=s.query(Fault).count(),
        severity={k: sev.get(k, 0) for k in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]}, types=types, trend=[dict(day=str(d), n=n) for d, n in days],
        open_alerts=s.query(Alert).filter(Alert.ack == False).count(),
        recent=[i_out(i) for i in s.query(Inspection).order_by(Inspection.created.desc()).limit(6)])
    s.close(); return r

@app.get("/api/faults")
def faults(severity: str = "", review_status: str = ""):
    s = db(); q = s.query(Fault)
    if severity: q = q.filter(Fault.severity == severity)
    if review_status: q = q.filter(Fault.review_status == review_status)
    out = [f_out(f) for f in q.order_by(Fault.created.desc())]; s.close(); return out

class Review(BaseModel):
    status: str | None = None; comment: str | None = None; severity: str | None = None
@app.post("/api/faults/{fid}/review")
def review(fid: int, b: Review):
    s = db(); f = s.get(Fault, fid)
    if not f: raise HTTPException(404, "Not found")
    if b.status:
        if b.status not in ["New", "Under Review", "Verified", "False Positive", "Resolved"]: raise HTTPException(400, "Invalid status")
        audit(s, "fault", fid, "status", f.review_status, b.status); f.review_status = b.status
    if b.severity and b.severity != f.severity:
        if b.severity not in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]: raise HTTPException(400, "Invalid severity")
        audit(s, "fault", fid, "severity", f.severity, b.severity); f.severity = b.severity
        if b.severity in RULES["alert_levels"]: s.add(Alert(fault_id=f.id, inspection_id=f.inspection_id, severity=b.severity, message=f"{b.severity} (reviewer-set): {f.type}"))
    if b.comment is not None: f.comment = b.comment
    s.commit(); r = f_out(f); s.close(); return r

@app.get("/api/alerts")
def alerts():
    s = db(); out = [dict(id=a.id, fault_id=a.fault_id, inspection_id=a.inspection_id, severity=a.severity, message=a.message, ack=a.ack, created=iso(a.created), delivery=a.delivery or "Sending...") for a in s.query(Alert).order_by(Alert.created.desc())]
    s.close(); return out
@app.patch("/api/alerts/{aid}/acknowledge")
def ack(aid: int):
    s = db(); a = s.get(Alert, aid)
    if not a: raise HTTPException(404, "Not found")
    a.ack = True; s.commit(); s.close(); return {"ok": True}

@app.get("/api/hardware/status")
def hw(): return {"connected": False, "message": "Not Connected. Using manual upload mode. No live sensor data is shown.", "last_image": None, "last_reading": None, "gps": "Not Connected", "thermal": "Not Connected"}

@app.get("/api/audit")
def audit_list():
    s = db(); out = [dict(entity=a.entity, id=a.entity_id, action=a.action, old=a.old, new=a.new, at=iso(a.at)) for a in s.query(Audit).order_by(Audit.id.desc()).limit(50)]
    s.close(); return out

@app.get("/api/inspections/{iid}/report")
def report(iid: int):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RI
    s = db(); i = s.get(Inspection, iid)
    if not i: raise HTTPException(404, "Not found")
    st = getSampleStyleSheet(); rid = "RPT-" + uuid.uuid4().hex[:8].upper(); path = STORE / f"{rid}.pdf"
    P = lambda t, k="BodyText": Paragraph(t, st[k]); el = [P("AI-Based Power-Line Inspection with a Drone", "Title"), P("Inspection Report", "Heading2")]
    el.append(P(f"Report ID: {rid}<br/>Inspection ID: {i.code} ({i.name})<br/>Date: {i.taken:%Y-%m-%d %H:%M}<br/>Inspector: {i.inspector or 'N/A'} | Drone: {i.drone_id or 'N/A'}<br/>Location: {i.location or 'N/A'} | Tower/line: {i.tower or 'N/A'}<br/>GPS: {f'{i.lat}, {i.lon}' if i.lat is not None and i.lon is not None else 'Not provided'}<br/>Images analyzed: {len(i.images)}<br/>Detection method: Heuristic computer vision (candidate detection, not a trained ML model)"))
    c = {k: sum(f.severity == k for f in i.faults) for k in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]}
    el += [Spacer(1, 8), P("Summary of findings", "Heading3"), P(f"Total: {len(i.faults)} | Critical: {c['CRITICAL']} | High: {c['HIGH']} | Medium: {c['MEDIUM']} | Low: {c['LOW']}")]
    for m in i.images:
        el += [P(f"Image: {m.filename} (quality: {m.quality}). {m.result}", "Heading4")]
        try: el.append(RI(str(STORE / m.annotated), width=300, height=200, kind="proportional"))
        except Exception: pass
    if i.faults:
        rows = [["ID", "Type", "Severity", "Review", "Explanation / action"]] + [[str(f.id), P(f.type), f.severity, f.review_status, P(f.description + " Action: " + f.action)] for f in i.faults]
        t = Table(rows, colWidths=[25, 90, 55, 65, 270]); t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0b1f3a")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), .4, colors.grey), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        el += [Spacer(1, 8), t]
    el += [Spacer(1, 10), P("Limitations", "Heading3"), P(LIMITS), P(f"Generated: {now():%Y-%m-%d %H:%M} UTC")]
    SimpleDocTemplate(str(path), pagesize=A4).build(el); s.close()
    return FileResponse(path, media_type="application/pdf", filename=f"{i.code}_report.pdf")

class Thermal(BaseModel):
    temperatures: list[list[float]]
@app.post("/api/inspections/{iid}/thermal")
def thermal(iid: int, b: Thermal):
    """Rule-based hotspot check on real thermal-sensor data (e.g. MLX90640 24x32 grid in deg C)."""
    try: a = np.array(b.temperatures, dtype=float)
    except ValueError: raise HTTPException(400, "Rows must have equal length")
    if a.ndim != 2 or a.size < 16: raise HTTPException(400, "Need a 2D temperature grid")
    s = db()
    if not s.get(Inspection, iid): raise HTTPException(404, "Not found")
    d = float(a.max() - np.median(a)); y, x = np.unravel_index(a.argmax(), a.shape)
    sev = "HIGH" if d >= RULES["thermal_delta_high"] else "MEDIUM" if d >= RULES["thermal_delta_medium"] else None
    if not sev: s.close(); return {"result": f"No thermal abnormality by rule (hottest point {d:.1f} C above frame median)."}
    f = Fault(inspection_id=iid, type="Thermal abnormality (candidate)", severity=sev, method="Rule-based thermal threshold on sensor data (not a trained model)",
        description=f"Hottest point {a.max():.1f} C is {d:.1f} C above the frame median ({np.median(a):.1f} C) at grid cell row {y}, col {x}. Thresholds are configurable and not validated against any utility standard; load, emissivity and distance affect readings.",
        box=json.dumps([int(x), int(y), 1, 1]), action="Verify with a calibrated thermal camera and qualified personnel.")
    s.add(f); s.flush()
    if sev in RULES["alert_levels"]: s.add(Alert(fault_id=f.id, inspection_id=iid, severity=sev, message=f"{sev} thermal candidate"))
    audit(s, "fault", f.id, "create", "", "thermal"); s.commit(); r = f_out(f); s.close(); return {"result": "Thermal candidate recorded", "fault": r}

@app.get("/api/settings")
def get_settings(): return RULES
@app.patch("/api/settings")
def set_settings(b: dict):
    for k, v in b.items():
        if k not in RULES: raise HTTPException(400, f"Unknown setting: {k}")
        try:
            if isinstance(RULES[k], float): b[k] = float(v)
        except (TypeError, ValueError): raise HTTPException(400, f"{k} must be a number")
        if k.endswith("_severity") and v not in SEV: raise HTTPException(400, f"{k} must be one of {SEV}")
        if k == "alert_levels" and not (isinstance(v, list) and all(x in SEV for x in v)): raise HTTPException(400, f"alert_levels must be a list of {SEV}")
    RULES.update(b); SETF.write_text(json.dumps(RULES)); return RULES
@app.post("/api/alerts/test")
def test_alert(): return {"result": send_external("Test notification from the Power-Line Inspection app")}
@app.get("/api/model")
def model_status():
    mf = BASE / "models" / "metrics.json"
    return {"installed": (BASE / "models" / "best.pt").exists(), "metrics": json.loads(mf.read_text()) if mf.exists() else None}
@app.get("/api/export/faults.csv")
def export_csv():
    s = db(); buf = io.StringIO(); w = csv.writer(buf); w.writerow(["id", "inspection_id", "type", "severity", "review_status", "method", "created"])
    for f in s.query(Fault).order_by(Fault.id): w.writerow([f.id, f.inspection_id, f.type, f.severity, f.review_status, f.method, f.created])
    s.close(); return Response(buf.getvalue(), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=faults.csv"})

JOB = {"proc": None, "name": "", "rc": None}
LOG = BASE / "models" / "train.log"
def _have(mod): return importlib.util.find_spec(mod) is not None
def _job(name, cmd, env=None):
    p = JOB["proc"]
    if p and p.poll() is None: raise HTTPException(409, "A job is already running. Wait or press Stop.")
    LOG.parent.mkdir(exist_ok=True)
    JOB.update(proc=subprocess.Popen(cmd, cwd=BASE, stdout=open(LOG, "w"), stderr=subprocess.STDOUT, env={**os.environ, **(env or {})}), name=name)
@app.get("/api/train/status")
def train_status():
    p = JOB["proc"]; run = bool(p and p.poll() is None)
    log = LOG.read_text(errors="ignore").replace("\r", "\n").splitlines()[-25:] if LOG.exists() and p else []
    return dict(running=run, job=JOB["name"], exit_code=None if (run or not p) else p.returncode, log=log,
                ultralytics=_have("ultralytics"), roboflow=_have("roboflow"), model=model_status())
@app.post("/api/train/install")
def train_install(): _job("install", [sys.executable, "-m", "pip", "install", "ultralytics", "roboflow"]); return {"ok": True}
@app.post("/api/train/dataset")
async def train_dataset(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".zip"): raise HTTPException(400, "Upload a .zip file")
    dest = DSET / uuid.uuid4().hex[:8]; dest.mkdir(parents=True); z = dest / "upload.zip"
    def fail(msg): shutil.rmtree(dest, ignore_errors=True); raise HTTPException(400, msg)
    with open(z, "wb") as f:
        while chunk := await file.read(1 << 20): f.write(chunk)
    try:
        with zipfile.ZipFile(z) as zf:
            if any(not (dest / n).resolve().is_relative_to(dest.resolve()) for n in zf.namelist()): fail("Unsafe path inside zip")
            zf.extractall(dest)
    except zipfile.BadZipFile: fail("Not a valid zip file")
    z.unlink(); ys = list(dest.rglob("data.yaml"))
    if not ys: fail("No data.yaml found. The zip must be a YOLO-format dataset.")
    return {"data_yaml": str(ys[0])}
class TrainReq(BaseModel):
    data_yaml: str | None = None; rf_key: str | None = None; rf_workspace: str | None = None; rf_project: str | None = None
    rf_version: int = 1; epochs: int = 50; imgsz: int = 640
@app.post("/api/train/start")
def train_start(b: TrainReq):
    if not _have("ultralytics"): raise HTTPException(400, "Training packages are not installed. Click 'Install training packages' first.")
    cmd = [sys.executable, "train.py", "--epochs", str(max(1, min(b.epochs, 300))), "--imgsz", str(max(32, min(b.imgsz, 1280)))]; env = {}
    if b.data_yaml:
        y = Path(b.data_yaml).resolve()
        if not y.exists() or not y.is_relative_to(DSET.resolve()): raise HTTPException(400, "Use a dataset uploaded through this page")
        cmd += ["--data", str(y)]
    elif b.rf_key and b.rf_workspace and b.rf_project:
        if not _have("roboflow"): raise HTTPException(400, "Roboflow package missing. Click 'Install training packages'.")
        cmd += ["--rf", b.rf_workspace, b.rf_project, str(b.rf_version)]; env["ROBOFLOW_API_KEY"] = b.rf_key
    else: raise HTTPException(400, "Upload a dataset zip or fill in the Roboflow details")
    _job("train", cmd, env); return {"ok": True}
@app.post("/api/train/stop")
def train_stop():
    p = JOB["proc"]
    if p and p.poll() is None: p.terminate()
    return {"ok": True}

app.mount("/", StaticFiles(directory=BASE / "static", html=True), name="ui")
