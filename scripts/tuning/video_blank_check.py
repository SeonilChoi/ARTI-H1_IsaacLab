"""Count blank (flat) frames in an mp4: python video_blank_check.py <file.mp4>"""
import sys, cv2, numpy as np
cap = cv2.VideoCapture(sys.argv[1]); n = blank = 0
while True:
    ok, f = cap.read()
    if not ok: break
    n += 1; blank += int(f.std() < 1.0)
print(f"{sys.argv[1]}: {n} frames, {blank} blank ({100.0 * blank / max(n, 1):.1f}%)")
