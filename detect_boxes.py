"""
Capturează o portiune a ecranului, detecteaza cadrane verzi și numara
cate APARITII DISTINCTE au existat pana la oprire.
Functionează pe RPi5; necesita mss + OpenCV.

Rulare:
  python3 detect_boxes.py --left 100 --top 100 --width 640 --height 480 --save out.mp4
"""

import argparse
import time
from collections import deque

import cv2
import numpy as np
from mss import mss


# ---------- Config detecție ----------
# Verde in HSV
HSV_LOWER = np.array([35, 60, 60], dtype=np.uint8)
HSV_UPPER = np.array([85, 255, 255], dtype=np.uint8)

# Filtre pentru zgomot / forma
MIN_AREA = 150       # area minimă a conturului (pixeli)
MIN_ASPECT = 0.5      # ratio w/h minim
MAX_ASPECT = 2.0      # ratio w/h maxim
MIN_SOLIDITY = 0.6   # area/convexHullArea: cat de „plin” e conturul (patrate => aproape 1.0)

# Tracking (asignare cutii intre cadre)
IOU_MATCH_THRESH = 0.3
MAX_MISSED_FRAMES = 8  # dupa cate cadre uitam un ID daca nu-l mai vedem

# --------------------------------------------------------


def iou(a, b):
    """IoU între două boxuri [x1,y1,x2,y2]."""
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    union = area_a + area_b - inter + 1e-6
    return inter / union


def detect_green_boxes(bgr):
    """Returnează lista de boxuri (x1,y1,x2,y2) pentru cadrane verzi in imagine."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
    # elim. zgomot
    k = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / (h + 1e-6)

        # „soliditate”
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull) + 1e-6
        solidity = area / hull_area

        if MIN_ASPECT <= aspect <= MAX_ASPECT and solidity >= MIN_SOLIDITY:
            boxes.append([x, y, x + w, y + h])
    return boxes


def main():
    ap = argparse.ArgumentParser(description="Numara cadranele verzi dintr-o zona a ecranului.")
    ap.add_argument("--left", type=int, required=True, help="Coordonata X a coltului stanga-sus")
    ap.add_argument("--top", type=int, required=True, help="Coordonata Y a coltului stanga-sus")
    ap.add_argument("--width", type=int, required=True, help="Latimea zonei capturate")
    ap.add_argument("--height", type=int, required=True, help="Inaltimea zonei capturate")
    ap.add_argument("--fps", type=int, default=20, help="FPS tinta pentru captura/inregistrare")
    ap.add_argument("--save", type=str, default="", help="Daca e setat, salveaza video in acest fisier (ex. out.mp4)")
    args = ap.parse_args()

    monitor = {"left": args.left, "top": args.top, "width": args.width, "height": args.height}
    sct = mss()

    # VideoWriter
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, args.fps, (args.width, args.height))

    # Tracker minimal: dict id -> {box, last_seen}
    next_id = 1
    tracks = {}
    total_unique = 0

    # FPS overlay
    times = deque(maxlen=30)

    print("[INFO] Portiune capturata:", monitor)
    print("[INFO] Apasa 'q' pentru stop.")
    try:
        while True:
            t0 = time.time()
            frame = np.array(sct.grab(monitor))  # RGBA
            bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # detectare boxuri verzi
            boxes = detect_green_boxes(bgr)

            # asociere cu trackere existente
            assigned = set()
            for tid, obj in list(tracks.items()):
                obj["missed"] += 1

            for box in boxes:
                # cauta cel mai bun match după IoU
                best_iou, best_tid = 0.0, None
                for tid, obj in tracks.items():
                    i = iou(box, obj["box"])
                    if i > best_iou:
                        best_iou, best_tid = i, tid
                if best_iou >= IOU_MATCH_THRESH:
                    # actualizeaza track existent
                    tracks[best_tid]["box"] = box
                    tracks[best_tid]["missed"] = 0
                    assigned.add(best_tid)
                else:
                    # obiect NOU -> aloca ID, creste total_unique
                    tracks[next_id] = {"box": box, "missed": 0}
                    total_unique += 1
                    assigned.add(next_id)
                    next_id += 1

            # elimina track-urile pierdute
            to_del = [tid for tid, obj in tracks.items() if obj["missed"] > MAX_MISSED_FRAMES]
            for tid in to_del:
                del tracks[tid]

            # desen
            for tid, obj in tracks.items():
                x1, y1, x2, y2 = map(int, obj["box"])
                cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(bgr, f"ID {tid}", (x1, max(0, y1 - 7)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

            # overlay info
            times.append(time.time())
            fps_now = 0.0
            if len(times) >= 2:
                dt = (times[-1] - times[0]) / max(1, (len(times) - 1))
                fps_now = 1.0 / dt if dt > 0 else 0.0

            cv2.putText(bgr, f"Unique green tiles: {total_unique}", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(bgr, f"FPS: {fps_now:.1f}", (10, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.imshow("Screen region (press 'q' to stop)", bgr)
            if writer is not None:
                writer.write(bgr)

            # respecta FPS tinta
            delay = max(1, int(1000 / max(1, args.fps)))
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break

    finally:
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        sct.close()
        print("\n====================")
        print(f"Numar final cadrane verzi DISTINCTE: {total_unique}")
        print("====================")


if __name__ == "__main__":
    main()
