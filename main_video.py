

import cv2
import numpy as np

# ── Video Input (0 = webcam OR give video path) ──
# cap = cv2.VideoCapture(0)  
cap = cv2.VideoCapture("video.mp4")  # use this for file

def t(img):
    return cv2.resize(img, (400, 300))

def label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0,0), (len(text)*11+10, 28), (0,0,0), -1)
    cv2.putText(out, text, (5,20), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255,255,255), 1)
    return out

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.resize(frame, (800, 600))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # ── Canny ──
    edges        = cv2.Canny(blur, 50, 150)
    edge_dilated = cv2.dilate(edges, np.ones((15,15), np.uint8))

    # ── Harris ──
    harris = cv2.cornerHarris(np.float32(gray), 4, 3, 0.04)
    harris = cv2.dilate(harris, None)
    harris_mask = (harris > 0.01 * harris.max()) & (edge_dilated > 0)
    coords = np.argwhere(harris_mask)

    # ── Shi-Tomasi ──
    shi_pts = cv2.goodFeaturesToTrack(gray, maxCorners=100,
                                      qualityLevel=0.01,
                                      minDistance=10, blockSize=5)

    shi_corners = []
    if shi_pts is not None:
        for c in shi_pts:
            x, y = c.ravel().astype(int)
            if edge_dilated[y, x] > 0:
                shi_corners.append((x, y))

    # ── Panels ──
    p1 = label(t(image), "Original")
    p2 = label(t(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)), "Canny Edges")

    harris_out = image.copy()
    for y, x in coords[::6]:   # ↓ reduce density for video
        cv2.circle(harris_out, (x,y), 2, (0,0,255), -1)
    p3 = label(t(harris_out), "Harris")

    shi_out = image.copy()
    for (x,y) in shi_corners:
        cv2.circle(shi_out, (x,y), 4, (0,255,255), -1)
    p4 = label(t(shi_out), "Shi-Tomasi")

    combined = image.copy()
    for y, x in coords[::6]:
        cv2.circle(combined, (x,y), 2, (0,0,255), -1)
    for (x,y) in shi_corners:
        cv2.circle(combined, (x,y), 4, (0,255,255), 1)
    p5 = label(t(combined), "Combined")

    empty = np.zeros_like(p1)

    grid = np.vstack([
        np.hstack([p1, p2, p3]),
        np.hstack([p4, p5, empty])
    ])

    cv2.imshow("Real-Time Corner Detection", grid)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()