"""Geometry heuristics (OpenCV only, no training). Output = CANDIDATES for human review."""
import math, cv2, numpy as np
def _edges(g): return cv2.Canny(cv2.GaussianBlur(g, (5, 5), 0), 50, 150)
def _segs(e, minlen, thr):
    L = cv2.HoughLinesP(e, 1, np.pi / 180, thr, minLineLength=minlen, maxLineGap=6)
    return [] if L is None else [tuple(int(v) for v in l[0]) for l in L]

def _wire_mask(g):
    """Thin dark OR bright structures (wires) via black-hat/top-hat. Robust to JPEG noise."""
    gb = cv2.GaussianBlur(g, (3, 3), 0); k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    m = cv2.max(cv2.morphologyEx(gb, cv2.MORPH_BLACKHAT, k), cv2.morphologyEx(gb, cv2.MORPH_TOPHAT, k))
    return cv2.morphologyEx((m > 35).astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

def _pieces(mask, w):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask); out = []
    for i in range(1, n):
        if max(st[i][2], st[i][3]) < 0.08 * w or st[i][4] < 30: continue
        ys, xs = np.nonzero(lab == i)
        vx, vy, x0, y0 = cv2.fitLine(np.column_stack([xs, ys]).astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        if vx < 0: vx, vy = -vx, -vy
        t = (xs - x0) * vx + (ys - y0) * vy; length = t.max() - t.min(); perp = np.abs(-(xs - x0) * vy + (ys - y0) * vx)
        if length < 0.08 * w or perp.max() > max(12, 0.04 * length) or len(xs) / length > 12: continue   # must be long and thin
        out.append(dict(vx=vx, vy=vy, x0=x0, y0=y0, t0=float(t.min()), t1=float(t.max()), ang=math.degrees(math.atan2(vy, vx))))
    return out

def find_sag(g, mask, R):
    h, w = g.shape; out = []
    n, lab, st, _ = cv2.connectedComponentsWithStats(cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2))
    for i in range(1, n):
        x, y, bw, bh, _ = st[i]
        if bw < 0.4 * w or bh > 0.35 * w: continue
        ys, xs = np.nonzero((lab == i) & (mask > 0))
        if len(xs) < 100: continue
        keep = np.ones(len(xs), bool)
        for _ in range(3):
            p = np.polyfit(xs[keep], ys[keep], 2); keep = np.abs(np.polyval(p, xs) - ys) < 8
            if keep.sum() < 50: break
        x0, x1 = xs[keep].min(), xs[keep].max(); xx = np.linspace(x0, x1, 100); yy = np.polyval(p, xx)
        dev = np.abs(yy - (yy[0] + (yy[-1] - yy[0]) * (xx - x0) / (x1 - x0))).max(); ratio = dev / (x1 - x0)
        if ratio >= R["sag_ratio"]:
            out.append(dict(type="Possible abnormal wire sag (candidate)", severity=R["sag_severity"], box=[int(x0), int(yy.min()), int(x1 - x0), int(dev) + 5],
                method="Heuristic curve fit (not a trained model)", action="Compare with design sag/clearance data for this span.",
                description=f"Wire dips {ratio*100:.1f}% of its visible span below the straight line between its ends (rule: >= {R['sag_ratio']*100:.0f}%). Normal wires also sag, and camera angle distorts this; true abnormality needs span/clearance reference data."))
    return out

def find_gaps(g, mask, R):
    h, w = g.shape; out = []; dm = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    P = [p for p in _pieces(mask, w) if abs(p["ang"]) < 35]
    for i, u in enumerate(P):
        for v in P[i + 1:]:
            lx = lambda p: p["x0"] + p["vx"] * p["t0"]
            a, b = (u, v) if lx(u) < lx(v) else (v, u)
            ax, ay = a["x0"] + a["vx"] * a["t1"], a["y0"] + a["vy"] * a["t1"]          # right end of left piece
            bx, by = b["x0"] + b["vx"] * b["t0"], b["y0"] + b["vy"] * b["t0"]          # left end of right piece
            gap = bx - ax
            if not (R["gap_frac"] * w < gap < 0.5 * w) or abs(a["ang"] - b["ang"]) > 4: continue
            if abs(-(bx - a["x0"]) * a["vy"] + (by - a["y0"]) * a["vx"]) > max(12, 0.03 * w): continue   # not on the same line
            xs = np.linspace(ax + 8, bx - 8, max(5, int(gap / 4))); ys = ay + (by - ay) * (xs - ax) / (bx - ax)
            if dm[np.clip(ys.astype(int), 0, h - 1), xs.astype(int)].mean() > 0.1: continue           # something spans the gap
            out.append(dict(type="Possible broken/disconnected wire (candidate)", severity=R["gap_severity"], box=[int(ax), int((ay + by) / 2) - 15, int(gap), 30],
                method="Heuristic collinear-gap test (not a trained model)", action="Urgent visual verification of the conductor by qualified staff.",
                description=f"A wire piece ends and a collinear piece resumes {int(gap)}px later with nothing between. Occlusion by poles, foliage or motion blur gives the same pattern."))
    return out

def find_tilt(g, e, R):
    h, w = g.shape; best = None
    for x1, y1, x2, y2 in _segs(e, h // 4, 80):
        ang = math.degrees(math.atan2(abs(x2 - x1), abs(y2 - y1)))
        if R["tilt_deg"] <= ang <= 45 and (best is None or ang > best[0]): best = (ang, x1, y1, x2, y2)
    if not best: return []
    a, x1, y1, x2, y2 = best
    return [dict(type="Possible tilted/displaced tower (candidate)", severity=R["tilt_severity"], box=[min(x1, x2), min(y1, y2), abs(x2 - x1) + 4, abs(y2 - y1)],
        method="Heuristic long-line angle test (not a trained model)", action="Check tower verticality with a survey or a level photo.",
        description=f"A long structural line leans {a:.0f} degrees from vertical (rule: >= {R['tilt_deg']:.0f}). Camera tilt and perspective cause the same effect.")]

def detect_geometry(g, R):
    e = _edges(g); mk = _wire_mask(g); return find_sag(g, mk, R) + find_gaps(g, mk, R) + find_tilt(g, e, R)
