#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detectează obiectele roșii de pe o porțiune a ecranului și le numără
fără a le repeta (folosește un tracker simplu bazat pe IoU).

Rulare exemplu:
  python3 count_red_objects.py --left 100 --top 100 --width 640 --height 480 --save out.mp4
"""

import argparse
import time
from collections import deque

import cv2
import numpy as np
from mss import mss

# ---------------- Configurație ----------------
MIN_AREA = 120         # minimă suprafață contur
IOU_MATCH_THRESH = 0.3
MAX_MISSED_FRAMES = 20  # câte cadre poate lipsi un obiect până e uitat
# ------------------------------------------------


def iou(a, b):
    """Returnează IoU + distanța între centre."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    union = area_a + area_b - inter + 1e-6
    iou_val = inter / union if union > 0 else 0.0

    # distanța dintre centre
    cx_a, cy_a = (a[0]+a[2])/2, (a[1]+a[3])/2
    cx_b, cy_b = (b[0]+b[2])/2, (b[1]+b[3])/2
    dist = np.hypot(cx_a - cx_b, cy_a - cy_b)
    return iou_val, dist




def detect_red_objects(bgr):
    """Returnează lista de boxuri [x1,y1,x2,y2] pentru obiecte roșii."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # roșu = 2 intervale pe H (aproape de 0 și aproape de 180)
    lower_red1 = np.array([0, 80, 60], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([160, 80, 60], dtype=np.uint8)
    upper_red2 = np.array([179, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # curățare zgomot
    k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        boxes.append([x, y, x + w, y + h])
    return boxes


def main():
    ap = argparse.ArgumentParser(description="Numără obiectele roșii de pe ecran.")
    ap.add_argument("--left", type=int, required=True, help="X colț stânga-sus")
    ap.add_argument("--top", type=int, required=True, help="Y colț stânga-sus")
    ap.add_argument("--width", type=int, required=True, help="Lățime zonă")
    ap.add_argument("--height", type=int, required=True, help="Înălțime zonă")
    ap.add_argument("--fps", type=int, default=20, help="FPS captură")
    ap.add_argument("--save", type=str, default="", help="Salvează video (opțional)")
    args = ap.parse_args()

    monitor = {"left": args.left, "top": args.top, "width": args.width, "height": args.height}
    sct = mss()

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, args.fps, (args.width, args.height))

    next_id = 1
    tracks = {}
    total_unique = 0
    times = deque(maxlen=30)

    print("[INFO] Capturăm regiunea:", monitor)
    print("[INFO] Apasă 'q' pentru stop.")
    try:
        while True:
            frame = np.array(sct.grab(monitor))  # RGBA
            bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            detections = detect_red_objects(bgr)

            # marchează toate track-urile ca „missed”
            for tid in list(tracks.keys()):
                tracks[tid]["missed"] += 1

            # asocieri detecții -> track existente
            DIST_THRESH = 1000  # distanța maximă între centre ca să fie același obiect

            for det in detections:
                best_score, best_tid = 0.0, None
                for tid, obj in tracks.items():
                    iou_val, dist = iou(det, obj["box"])
                    # scor compozit: mai mare = mai similar
                    score = iou_val - (dist / DIST_THRESH) * 0.5
                    if dist < DIST_THRESH and score > best_score:
                        best_score, best_tid = score, tid

                if best_tid is not None and best_score > 0:
                    # update pe track existent
                    tracks[best_tid]["box"] = det
                    tracks[best_tid]["missed"] = 0
                else:
                    # nou obiect
                    tracks[next_id] = {"box": det, "missed": 0}
                    total_unique += 1
                    next_id += 1


            # curățare track-uri pierdute
            to_del = [tid for tid, obj in tracks.items() if obj["missed"] > MAX_MISSED_FRAMES]
            for tid in to_del:
                del tracks[tid]

            # desen și contorizare
            for tid, obj in tracks.items():
                x1, y1, x2, y2 = map(int, obj["box"])
                cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(bgr, f"ID {tid}", (x1, max(15, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            times.append(time.time())
            if len(times) >= 2:
                fps_now = len(times) / (times[-1] - times[0])
            else:
                fps_now = 0

            cv2.putText(bgr, f"Obiecte rosii unice: {total_unique}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(bgr, f"FPS: {fps_now:.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.imshow("Detectie obiecte rosii", bgr)
            if writer is not None:
                writer.write(bgr)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        sct.close()
        print("\n====================")
        print(f"Număr final obiecte roșii DISTINCTE: {total_unique}")
        print("====================")


if __name__ == "__main__":
    main()
