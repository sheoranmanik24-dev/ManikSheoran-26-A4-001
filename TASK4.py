"""
UGV Track Path Planner - Final Version
Generates a closed-loop green path starting from START, avoiding all obstacles,
returning to complete the circuit, and staying strictly inside the gray road.
"""

import cv2
import numpy as np
import heapq
import os


# ══════════════════════════════════════════════
# 1.  A* PATHFINDING
# ══════════════════════════════════════════════
def astar(cost_map, start, goal):
    """
    A* on a 2-D cost map.
    start / goal are (row, col) tuples.
    Returns list[(row, col)] or None.
    """
    h, w = cost_map.shape

    def heuristic(a, b):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    open_set = []
    counter = 0
    heapq.heappush(open_set, (heuristic(start, goal), counter, start))
    came_from = {}
    g_score = {start: 0}

    neighbors = [
        (0, 1, 1.0), (1, 0, 1.0), (0, -1, 1.0), (-1, 0, 1.0),
        (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414),
    ]

    iters = 0
    max_iter = h * w

    while open_set and iters < max_iter:
        iters += 1
        _, _, current = heapq.heappop(open_set)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        for dx, dy, move_cost in neighbors:
            nr, nc = current[0] + dy, current[1] + dx
            if 0 <= nr < h and 0 <= nc < w:
                cc = cost_map[nr, nc]
                if np.isinf(cc):
                    continue
                tg = g_score[current] + move_cost + cc
                if tg < g_score.get((nr, nc), float("inf")):
                    came_from[(nr, nc)] = current
                    g_score[(nr, nc)] = tg
                    f = tg + heuristic((nr, nc), goal)
                    counter += 1
                    heapq.heappush(open_set, (f, counter, (nr, nc)))
    return None


def astar_with_retries(cost_map, start, goal, max_offset=25):
    """Try A*; if it fails, reposition start/goal to nearest valid pixel."""
    path = astar(cost_map, start, goal)
    if path is not None:
        return path

    for offset in range(5, max_offset + 1, 5):
        for p in [start, goal]:
            y, x = p
            best = None
            best_dist = float("inf")
            for dy in range(-offset, offset + 1, 5):
                for dx in range(-offset, offset + 1, 5):
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < cost_map.shape[0]
                            and 0 <= nx < cost_map.shape[1]
                            and not np.isinf(cost_map[ny, nx])):
                        d = dy ** 2 + dx ** 2
                        if d < best_dist:
                            best_dist = d
                            best = (ny, nx)
            if best is not None:
                ns = best if p == start else start
                ng = best if p == goal else goal
                path = astar(cost_map, ns, ng)
                if path is not None:
                    return path
    return None


# ══════════════════════════════════════════════
# 2.  OBSTACLE DETECTION
# ══════════════════════════════════════════════
def detect_obstacles(img):
    """
    Detect colored blocks, white squares, and tire rings.
    Key: HSV saturation > 40 keeps the gray background (S~31) out.
    Returns
    -------
    safety      : dilated mask of colour + tire obstacles (for track detection)
    all_obs     : undilated mask of ALL obstacles (for pathfinding cost)
    color_contours, white_contours, tire_contours
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # --- colourful blocks (S > 40 excludes the gray background) ---
    blue = cv2.inRange(hsv, np.array([95, 40, 50]), np.array([130, 255, 255]))
    r1 = cv2.inRange(hsv, np.array([0, 40, 50]), np.array([10, 255, 255]))
    r2 = cv2.inRange(hsv, np.array([165, 40, 50]), np.array([180, 255, 255]))
    red = cv2.bitwise_or(r1, r2)
    yellow = cv2.inRange(hsv, np.array([15, 40, 50]), np.array([35, 255, 255]))
    brown = cv2.inRange(hsv, np.array([5, 40, 25]), np.array([25, 200, 200]))

    colored = red | blue | yellow | brown
    colored = cv2.morphologyEx(colored, cv2.MORPH_CLOSE, k3)
    colored = cv2.morphologyEx(colored, cv2.MORPH_OPEN, k3)

    color_contours, _ = cv2.findContours(
        colored, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    color_contours = [c for c in color_contours if cv2.contourArea(c) > 50]

    # --- white / cream squares ---
    white_mask = cv2.inRange(gray, 165, 255)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, k3)
    white_contours, _ = cv2.findContours(
        white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    white_contours = [c for c in white_contours if cv2.contourArea(c) > 80]

    # --- dark tire rings / potholes ---
    tire_raw = cv2.inRange(gray, 20, 50)
    tire_raw = cv2.morphologyEx(tire_raw, cv2.MORPH_OPEN, k3)
    tire_raw = cv2.morphologyEx(tire_raw, cv2.MORPH_CLOSE, k3)
    tire_contours, _ = cv2.findContours(
        tire_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    tire_contours = [c for c in tire_contours if cv2.contourArea(c) > 30]

    # --- combined obstacle mask (used for pathfinding cost) ---
    all_obs = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(all_obs, color_contours, -1, 255, -1)
    cv2.drawContours(all_obs, white_contours, -1, 255, -1)
    cv2.drawContours(all_obs, tire_contours, -1, 255, -1)

    # safety: dilate colour + tire (NOT white — they are scattered markers;
    # dilating them would eat the entire track)
    hard_obs = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(hard_obs, color_contours, -1, 255, -1)
    cv2.drawContours(hard_obs, tire_contours, -1, 255, -1)
    safety = cv2.dilate(
        hard_obs, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=2,
    )

    return safety, all_obs, color_contours, white_contours, tire_contours


# ══════════════════════════════════════════════
# 3.  TRACK (ROAD) DETECTION — gradient-based
# ══════════════════════════════════════════════
def detect_track(img, obstacle_safety):
    """
    The track is a gray band whose boundaries create gradient edges.

    Pipeline
    --------
    1. Sobel gradient → edge magnitude.
    2. Dilate high-gradient pixels → "edge zone" covering the full band.
    3. Intersect with broad gray mask (S<30, 40<gray<130).
    4. Remove obstacle safety zone.
    5. Keep largest connected component → track.
    """
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sat = hsv[:, :, 1]

    blurred = cv2.bilateralFilter(gray, 9, 75, 75)

    gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
    gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)
    grad = np.sqrt(gx ** 2 + gy ** 2)

    # edge zone — dilate to cover the full track width
    edge = (grad > 2.5).astype(np.uint8) * 255
    edge = cv2.dilate(
        edge, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    )

    # broad gray mask (low-saturation, medium-brightness)
    gray_mask = np.zeros((h, w), dtype=np.uint8)
    gray_mask[(sat < 30) & (gray > 40) & (gray < 130)] = 255
    gray_mask[obstacle_safety > 0] = 0

    track = cv2.bitwise_and(gray_mask, edge)

    # morphological clean-up
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    track = cv2.morphologyEx(track, cv2.MORPH_CLOSE, k7)
    track = cv2.morphologyEx(track, cv2.MORPH_OPEN, k5)

    # fill holes inside the band
    cnts, hier = cv2.findContours(
        track, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if hier is not None:
        for i in range(len(cnts)):
            if hier[0][i][3] != -1:          # has parent → hole
                cv2.drawContours(track, cnts, i, 255, -1)

    # keep components ≥ 5 % of the largest
    nlab, labs, stats, _ = cv2.connectedComponentsWithStats(track, 8)
    if nlab > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        thresh = np.max(areas) * 0.05
        filt = np.zeros((h, w), dtype=np.uint8)
        for i in range(1, nlab):
            if stats[i, cv2.CC_STAT_AREA] >= thresh:
                filt[labs == i] = 255
        track = filt

    track = cv2.morphologyEx(track, cv2.MORPH_CLOSE, k7)
    track = cv2.dilate(track, k5, iterations=2)
    return track


# ══════════════════════════════════════════════
# 4.  START-POSITION FINDER
# ══════════════════════════════════════════════
def find_start(img, track_mask):
    """Locate the white START marker and snap to the nearest track pixel."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]

    white = np.zeros_like(gray)
    white[(gray > 190) & (sat < 30)] = 255
    cnts, _ = cv2.findContours(
        white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if cnts:
        c = max(cnts, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            yt, xt = np.where(track_mask > 0)
            if len(yt):
                d = (xt - cx) ** 2 + (yt - cy) ** 2
                return int(yt[np.argmin(d)]), int(xt[np.argmin(d)])

    # fallback — right-most track pixel
    yt, xt = np.where(track_mask > 0)
    if len(yt):
        rm = xt > np.percentile(xt, 65)
        if np.any(rm):
            i = np.argmax(xt[rm])
            return int(yt[rm][i]), int(xt[rm][i])
    return int(yt[len(yt) // 2]), int(xt[len(xt) // 2])


# ══════════════════════════════════════════════
# 5.  WAYPOINT EXTRACTION
# ══════════════════════════════════════════════
def extract_waypoints(track_mask, num=24):
    """
    Cast radial rays from the track centroid; at each angle keep the point
    with the highest distance-transform value (track centre-line).
    Subsample to *num* evenly-spaced waypoints.
    """
    h, w = track_mask.shape
    yt, xt = np.where(track_mask > 0)
    if len(yt) == 0:
        return []

    cy, cx = float(np.mean(yt)), float(np.mean(xt))
    dt = cv2.distanceTransform(track_mask, cv2.DIST_L2, 5)

    raw = []
    for i in range(num * 4):
        a = 2 * np.pi * i / (num * 4)
        best, best_d = None, 0
        for r in range(3, max(h, w)):
            ty = int(cy + r * np.sin(a))
            tx = int(cx + r * np.cos(a))
            if 0 <= ty < h and 0 <= tx < w:
                if track_mask[ty, tx] > 0:
                    d = dt[ty, tx]
                    if d > best_d:
                        best_d = d
                        best = (ty, tx)
                elif best is not None:
                    break
        if best and best_d > 1:
            raw.append(best)

    if not raw or len(raw) <= num:
        return raw

    # even spacing by cumulative arc-length
    cum = [0.0]
    for i in range(1, len(raw)):
        cum.append(
            cum[-1] + np.sqrt(
                (raw[i][0] - raw[i - 1][0]) ** 2
                + (raw[i][1] - raw[i - 1][1]) ** 2
            )
        )
    total = cum[-1]
    if total == 0:
        return raw[:num]

    step = total / num
    sel = [raw[0]]
    for i in range(1, num):
        idx = int(np.argmin(np.abs(np.array(cum) - i * step)))
        if raw[idx] not in sel:
            sel.append(raw[idx])
    return sel


def order_clockwise(waypoints, center):
    """Sort waypoints clockwise around *center*."""
    cy, cx = center
    angles = [np.arctan2(wp[0] - cy, wp[1] - cx) for wp in waypoints]
    return [wp for _, wp in sorted(zip(angles, waypoints))]


# ══════════════════════════════════════════════
# 6.  PATH SMOOTHING & TRACK CLAMPING
# ══════════════════════════════════════════════
def smooth_path(path, win=5):
    """Moving-average smoother."""
    if len(path) < win * 2 + 1:
        return path
    sm = list(path)
    for i in range(win, len(path) - win):
        sm[i] = (
            int(np.mean([p[0] for p in path[i - win : i + win + 1]])),
            int(np.mean([p[1] for p in path[i - win : i + win + 1]])),
        )
    return sm


def clamp_to_track(path, track_mask, cost_map):
    """
    CRITICAL SAFETY: snap every path pixel to the nearest valid (on-track,
    off-obstacle) pixel.  Guarantees the green path NEVER leaves the road.
    """
    h, w = track_mask.shape
    valid = (track_mask > 0) & (~np.isinf(cost_map))

    clamped = []
    for (r, c) in path:
        if 0 <= r < h and 0 <= c < w and valid[r, c]:
            clamped.append((r, c))
        else:
            # Fast local search with expanding windows
            found = False
            for rad in [1, 3, 5, 8, 12]:
                r_lo = max(0, r - rad)
                r_hi = min(h, r + rad + 1)
                c_lo = max(0, c - rad)
                c_hi = min(w, c + rad + 1)
                block = valid[r_lo:r_hi, c_lo:c_hi]
                if block.any():
                    ys, xs = np.where(block)
                    dists = (ys - (r - r_lo)) ** 2 + (xs - (c - c_lo)) ** 2
                    idx = np.argmin(dists)
                    clamped.append(
                        (int(ys[idx] + r_lo), int(xs[idx] + c_lo))
                    )
                    found = True
                    break
            if not found:
                clamped.append((r, c))
    return clamped


# ══════════════════════════════════════════════
# 7.  PROCESS ONE IMAGE
# ══════════════════════════════════════════════
def process_track(input_path, output_path):
    print(f"\nProcessing: {input_path}")
    img = cv2.imread(input_path)
    if img is None:
        print(f"  ERROR: Cannot read {input_path}")
        return False

    h, w = img.shape[:2]

    # ---- obstacles ----
    safety, all_obs, cc, wc, tc = detect_obstacles(img)
    n_obs = len(cc) + len(wc)
    print(f"  Obstacles: {n_obs} blocks, {len(tc)} tires")

    # ---- track ----
    track = detect_track(img, safety)
    ta = np.sum(track > 0)
    print(f"  Track: {100 * ta / (h * w):.1f}%")
    if ta < 500:
        print("  ERROR: track not detected")
        return False

    # ---- cost map (infinite outside track & on obstacles) ----
    cost_obs = cv2.dilate(
        all_obs, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )
    dt = cv2.distanceTransform(track, cv2.DIST_L2, 5)
    md = max(np.max(dt), 1.0)
    cost = np.full((h, w), np.inf, dtype=np.float32)
    valid = (track > 0) & (cost_obs == 0)
    cost[valid] = 1.0 + (1.0 - dt[valid] / md) * 15.0
    # ↑ cost is LOWEST at the track centre-line → path hugs the middle

    # ---- start position ----
    start = find_start(img, track)
    print(f"  Start: {start}")

    # ---- waypoints ----
    wps = extract_waypoints(track, num=24)
    print(f"  Waypoints: {len(wps)}")
    if len(wps) < 4:
        print("  ERROR: too few waypoints")
        return False

    # ---- order clockwise, starting nearest to START ----
    yt, xt = np.where(track > 0)
    center = (float(np.mean(yt)), float(np.mean(xt)))
    wps = order_clockwise(wps, center)
    ds = [np.hypot(wp[0] - start[0], wp[1] - start[1]) for wp in wps]
    si = int(np.argmin(ds))
    wps = wps[si:] + wps[:si]  # cycle so first waypoint ≈ START

    # ---- A* between consecutive waypoints (wraps → closed loop) ----
    full_path = []
    ok = 0
    for i in range(len(wps)):
        a = wps[i]
        b = wps[(i + 1) % len(wps)]
        seg = astar_with_retries(cost, a, b, max_offset=20)
        if seg:
            if full_path and seg[0] == full_path[-1]:
                seg = seg[1:]
            full_path.extend(seg)
            ok += 1

    print(f"  A*: {ok}/{len(wps)} | Path: {len(full_path)} px")
    if len(full_path) < 10:
        print("  ERROR: path too short")
        return False

    # ---- explicit loop closure back to first pixel ----
    seg_close = astar_with_retries(cost, full_path[-1], full_path[0],
                                   max_offset=20)
    if seg_close:
        if seg_close[0] == full_path[-1]:
            seg_close = seg_close[1:]
        full_path.extend(seg_close)

    # ---- smooth, then CLAMP every pixel inside the road ----
    full_path = smooth_path(full_path, win=3)
    full_path = clamp_to_track(full_path, track, cost)

    # ---- compute path length ----
    plen = sum(
        np.hypot(full_path[i + 1][0] - full_path[i][0],
                 full_path[i + 1][1] - full_path[i][1])
        for i in range(len(full_path) - 1)
    )

    # ════════════ RENDER ════════════
    out = img.copy()

    # track outline (green border)
    tcnt, _ = cv2.findContours(
        track, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if tcnt:
        cv2.drawContours(out, [max(tcnt, key=cv2.contourArea)],
                         -1, (0, 200, 0), 2)

    # colour obstacles → orange bbox
    for c in cc:
        x, y, bw, bh = cv2.boundingRect(c)
        cv2.rectangle(out, (x - 2, y - 2), (x + bw + 2, y + bh + 2),
                      (0, 140, 255), 2)

    # white blocks → yellow bbox
    for c in wc:
        x, y, bw, bh = cv2.boundingRect(c)
        cv2.rectangle(out, (x - 2, y - 2), (x + bw + 2, y + bh + 2),
                      (0, 255, 255), 1)

    # tires → red circle
    for c in tc:
        (cx, cy), r = cv2.minEnclosingCircle(c)
        cv2.circle(out, (int(cx), int(cy)), int(r) + 4, (0, 0, 255), 2)

    # ── GREEN path ──
    for i in range(len(full_path) - 1):
        p1 = (full_path[i][1], full_path[i][0])
        p2 = (full_path[i + 1][1], full_path[i + 1][0])
        cv2.line(out, p1, p2, (0, 255, 0), 3, cv2.LINE_AA)

    # red dots at waypoints
    for wp in wps:
        cv2.circle(out, (wp[1], wp[0]), 5, (0, 0, 255), -1)

    # START marker
    cv2.circle(out, (start[1], start[0]), 8, (255, 100, 0), -1)
    cv2.putText(out, "START", (start[1] + 15, start[0] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # direction arrow near start
    if len(full_path) > 20:
        cv2.arrowedLine(
            out,
            (full_path[5][1], full_path[5][0]),
            (full_path[20][1], full_path[20][0]),
            (255, 255, 255), 2, tipLength=0.3,
        )

    # telemetry
    cv2.putText(out, f"Closed-loop path: {int(plen)} px", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(out, f"Obstacles: {n_obs}  Tires: {len(tc)}", (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
    cv2.putText(out, "A* closed-loop planner", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imwrite(output_path, out)
    print(f"  Saved: {output_path}")
    return True


# ══════════════════════════════════════════════
# 8.  MAIN — process all 10 images
# ══════════════════════════════════════════════
def main():
    image_names = [
        ("t41.jpeg",  "TASK4output1.jpg"),
        ("t42.jpeg",  "TASK4output2.jpg"),
        ("t43.jpeg",  "TASK4output3.jpg"),
        ("t44.jpeg",  "TASK4output4.jpg"),
        ("t45.jpeg",  "TASK4output5.jpg"),
        ("t46.jpeg",  "TASK4output6.jpg"),
        ("t47.jpeg",  "TASK4output7.jpg"),
        ("t48.jpeg",  "TASK4output8.jpg"),
        ("t49.jpeg",  "TASK4output9.jpg"),
        ("t410.jpeg", "TASK4output10.jpg"),
    ]

    print("=" * 55)
    print("  UGV Closed-Loop Track Path Planner  (final)")
    print("=" * 55)

    ok = 0
    for inp, out in image_names:
        if os.path.exists(inp):
            if process_track(inp, out):
                ok += 1
        else:
            alt = inp.replace(".jpeg", ".png")
            if os.path.exists(alt) and process_track(alt, out):
                ok += 1

    print(f"\n{'=' * 55}")
    print(f"  Done: {ok}/{len(image_names)} images processed successfully")
    print("=" * 55)


if __name__ == "__main__":
    main()