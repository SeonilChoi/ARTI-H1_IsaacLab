"""Count blank (flat) frames in an mp4: python video_blank_check.py <file.mp4>"""
import sys, cv2, numpy as np


def frames(path):
    cap = cv2.VideoCapture(path)
    ok, f = cap.read()
    if not ok:
        # The Isaac Lab 3.x venv's OpenCV has no FFmpeg backend and reads no frames; decode with imageio-ffmpeg instead.
        import imageio.v2 as iio
        yield from iio.get_reader(path)
        return
    while ok:
        yield f
        ok, f = cap.read()


n = blank = 0
for f in frames(sys.argv[1]):
    n += 1; blank += int(np.asarray(f).std() < 1.0)
print(f"{sys.argv[1]}: {n} frames, {blank} blank ({100.0 * blank / max(n, 1):.1f}%)")
