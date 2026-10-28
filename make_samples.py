"""Generates test images in sample_images/ so you can upload them in the UI."""
import cv2, numpy as np
from pathlib import Path
out = Path(__file__).parent / "sample_images"; out.mkdir(exist_ok=True)
rng = np.random.default_rng(1)
def base(w=1280, h=720):
    img = np.zeros((h, w, 3), np.uint8); img[:] = (200, 170, 130)          # sky (BGR)
    cv2.line(img, (0, 200), (w, 260), (40, 40, 40), 4); cv2.line(img, (0, 230), (w, 290), (40, 40, 40), 4)
    cv2.rectangle(img, (600, 150), (640, h), (90, 90, 90), -1)                # tower mast
    return img
def noisy(img): return np.clip(img.astype(int) + rng.normal(0, 10, img.shape), 0, 255).astype(np.uint8)
rust = (30, 80, 170)                                                          # BGR rust colour
big = base(); cv2.rectangle(big, (560, 300), (780, 520), rust, -1)
small = base(); cv2.circle(small, (640, 400), 35, rust, -1)
clean = base()
cv2.imwrite(str(out / "1_rust_large_MEDIUM.png"), noisy(big))
cv2.imwrite(str(out / "2_rust_small_LOW.png"), noisy(small))
cv2.imwrite(str(out / "3_clean_no_fault.png"), noisy(clean))
cv2.imwrite(str(out / "4_blurry_unable.png"), cv2.GaussianBlur(noisy(big), (61, 61), 0))
cv2.imwrite(str(out / "5_dark_unable.png"), (noisy(big) * 0.05).astype(np.uint8))
cv2.imwrite(str(out / "6_tiny_unable.png"), noisy(big)[:80, :120])
def scene(): img = np.zeros((720, 1280, 3), np.uint8); img[:] = (200, 170, 130); return img
sag = scene()
pts = np.array([[x, 200 + 120 * 4 * ((x - 60) / 1160) * (1 - (x - 60) / 1160)] for x in range(60, 1221, 8)], np.int32)
cv2.polylines(sag, [pts], False, (40, 40, 40), 4); cv2.rectangle(sag, (40, 190), (70, 290), (90, 90, 90), -1); cv2.rectangle(sag, (1210, 190), (1240, 290), (90, 90, 90), -1)
brk = scene(); cv2.line(brk, (0, 300), (500, 320), (40, 40, 40), 4); cv2.line(brk, (700, 328), (1280, 351), (40, 40, 40), 4)
tilt = scene(); cv2.line(tilt, (400, 700), (618, 100), (90, 90, 90), 20)
cv2.imwrite(str(out / "7_wire_sag.png"), noisy(sag)); cv2.imwrite(str(out / "8_broken_wire.png"), noisy(brk)); cv2.imwrite(str(out / "9_tilted_tower.png"), noisy(tilt))
print("Samples created in", out)
