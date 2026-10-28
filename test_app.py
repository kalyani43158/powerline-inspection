import os, tempfile, subprocess, sys
from pathlib import Path
tmp = tempfile.mkdtemp(); os.environ["PL_DB"] = f"{tmp}/t.db"; os.environ["PL_SETTINGS"] = f"{tmp}/s.json"; os.environ["PL_DATASETS"] = f"{tmp}/ds"; os.environ["PL_STORE"] = f"{tmp}/store"
HERE = Path(__file__).parent
if not (HERE / "sample_images").exists(): subprocess.run([sys.executable, str(HERE / "make_samples.py")], check=True)
from fastapi.testclient import TestClient
from main import app
c = TestClient(app); S = HERE / "sample_images"

def run(name, fname=None, data=None):
    data = data if data is not None else (S / fname).read_bytes()
    r = c.post("/api/inspections", data={"name": name, "location": "Test"}, files=[("files", (fname or "x.png", data, "image/png"))])
    return r

def test_health(): assert c.get("/api/health").json()["status"] == "ok"
def test_rust_large():
    r = run("rust big", "1_rust_large_MEDIUM.png").json(); f = r["faults"]
    assert f and f[0]["severity"] == "MEDIUM" and "candidate" in f[0]["type"].lower()
    assert c.get(r["images"][0]["annotated"]).status_code == 200
def test_rust_small():
    f = run("rust small", "2_rust_small_LOW.png").json()["faults"]
    assert f and f[0]["severity"] == "LOW"
def test_clean_has_no_fault():
    r = run("clean", "3_clean_no_fault.png").json()
    assert r["n_faults"] == 0 and "No visible fault" in r["images"][0]["result"]
def test_unusable_images():
    for fn, q in [("4_blurry_unable.png", "blurry"), ("5_dark_unable.png", "too_dark"), ("6_tiny_unable.png", "too_small")]:
        r = run(fn, fn).json(); assert r["images"][0]["quality"] == q and r["n_faults"] == 0 and "Unable" in r["images"][0]["result"]
def test_validation():
    assert run("bad ext", "a.gif", b"GIF89a").status_code == 400
    assert run("corrupt", "a.png", b"not an image").status_code == 400
def test_review_audit_alert():
    f = run("rev", "1_rust_large_MEDIUM.png").json()["faults"][0]
    assert c.post(f"/api/faults/{f['id']}/review", json={"status": "Verified", "comment": "ok"}).json()["review_status"] == "Verified"
    assert c.post(f"/api/faults/{f['id']}/review", json={"status": "Bogus"}).status_code == 400
    n = len(c.get("/api/alerts").json()); c.post(f"/api/faults/{f['id']}/review", json={"severity": "HIGH"})
    al = c.get("/api/alerts").json(); assert len(al) == n + 1
    assert c.patch(f"/api/alerts/{al[0]['id']}/acknowledge").json()["ok"]
    assert any(a["action"] == "severity" for a in c.get("/api/audit").json())
def test_dashboard_report_hardware_delete():
    d = c.get("/api/dashboard").json(); assert d["inspections"] > 0 and sum(d["severity"].values()) == d["faults"]
    i = c.get("/api/inspections").json()[0]; r = c.get(f"/api/inspections/{i['id']}/report")
    assert r.status_code == 200 and r.content[:4] == b"%PDF"
    assert c.get("/api/hardware/status").json()["connected"] is False
    assert c.delete(f"/api/inspections/{i['id']}").status_code == 200 and c.get(f"/api/inspections/{i['id']}").status_code == 404

def test_thermal_rule():
    iid = run("th", "3_clean_no_fault.png").json()["id"]
    grid = [[30.0] * 32 for _ in range(24)]; grid[5][7] = 85.0
    r = c.post(f"/api/inspections/{iid}/thermal", json={"temperatures": grid}).json()
    assert r["fault"]["severity"] == "HIGH" and "not a trained model" in r["fault"]["method"]
    assert "No thermal" in c.post(f"/api/inspections/{iid}/thermal", json={"temperatures": [[30.0] * 32] * 24}).json()["result"]
    assert c.post(f"/api/inspections/{iid}/thermal", json={"temperatures": [[1.0]]}).status_code == 400

def types(fn): return [f["type"] for f in run(fn, fn).json()["faults"]]
def test_geometry_detectors():
    assert any("sag" in t for t in types("7_wire_sag.png"))
    assert any("broken" in t for t in types("8_broken_wire.png"))
    assert any("tilted" in t for t in types("9_tilted_tower.png"))
    for fn in ["3_clean_no_fault.png", "1_rust_large_MEDIUM.png"]:
        assert not any(k in t for t in types(fn) for k in ["sag", "broken", "tilted"]), fn

import time
def test_notifications_settings_export():
    f = run("n", "1_rust_large_MEDIUM.png").json()["faults"][0]
    c.post(f"/api/faults/{f['id']}/review", json={"severity": "HIGH"}); time.sleep(1.5)
    assert c.get("/api/alerts").json()[0]["delivery"].startswith("In-app only")
    assert "In-app only" in c.post("/api/alerts/test").json()["result"]
    assert c.patch("/api/settings", json={"sag_ratio": 0.08}).json()["sag_ratio"] == 0.08
    c.patch("/api/settings", json={"sag_ratio": 0.06})
    assert c.patch("/api/settings", json={"nope": 1}).status_code == 400
    assert c.get("/api/model").json()["installed"] is False
    assert c.get("/api/export/faults.csv").text.startswith("id,")

import io, zipfile
def _zip(name, body):
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z: z.writestr(name, body)
    return b.getvalue()
def test_training_page_api():
    s = c.get("/api/train/status").json(); assert s["running"] is False and "ultralytics" in s
    assert c.post("/api/train/start", json={"epochs": 1}).status_code == 400
    assert c.post("/api/train/dataset", files={"file": ("d.zip", _zip("../evil.txt", "x"))}).status_code == 400
    assert c.post("/api/train/dataset", files={"file": ("d.zip", _zip("a.txt", "x"))}).status_code == 400
    r = c.post("/api/train/dataset", files={"file": ("d.zip", _zip("data.yaml", "names: [a]\n"))})
    assert r.status_code == 200 and r.json()["data_yaml"].endswith("data.yaml")

import cv2
def _variants(fn):
    img = cv2.imread(str(S / fn)); yield img
    for q in (90, 75, 50, 30): yield cv2.imdecode(cv2.imencode(".jpg", img, [1, q])[1], 1)
    yield cv2.imdecode(cv2.imencode(".jpg", cv2.resize(img, None, fx=.6, fy=.6), [1, 70])[1], 1)
def test_robust_to_jpeg_and_resize():
    from detectors import detect_geometry
    import main
    exp = {"7_wire_sag.png": "sag", "8_broken_wire.png": "broken", "10_user_broken_wire.png": "broken", "9_tilted_tower.png": "tilted",
           "3_clean_no_fault.png": None, "1_rust_large_MEDIUM.png": None}
    for fn, want in exp.items():
        for v in _variants(fn):
            got = [d["type"] for d in detect_geometry(cv2.cvtColor(v, cv2.COLOR_BGR2GRAY), main.RULES)]
            assert (any(want in g for g in got) if want else not got), (fn, got)

def test_all_samples_give_expected_verdict():
    exp = {"1_rust_large_MEDIUM.png": ("ok", ["rust"]), "2_rust_small_LOW.png": ("ok", ["rust"]), "3_clean_no_fault.png": ("ok", []),
           "4_blurry_unable.png": ("blurry", []), "5_dark_unable.png": ("too_dark", []), "6_tiny_unable.png": ("too_small", []),
           "7_wire_sag.png": ("ok", ["sag"]), "8_broken_wire.png": ("ok", ["broken"]), "9_tilted_tower.png": ("ok", ["tilted"]),
           "10_user_broken_wire.png": ("ok", ["broken"])}
    for fn, (q, kinds) in exp.items():
        r = run(fn, fn).json(); got = [f["type"].lower() for f in r["faults"]]
        assert r["images"][0]["quality"] == q and len(got) == len(kinds) and all(any(k in g for g in got) for k in kinds), (fn, got)
    assert c.get("/api/health").json()["version"].startswith("v")
