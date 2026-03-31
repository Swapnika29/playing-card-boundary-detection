import cv2
import numpy as np
import os

image_path = "image.jpeg"   # change this
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError("Could not load image.")

os.makedirs("experiment_outputs", exist_ok=True)


def t(img):
    return cv2.resize(img, (400, 300))


def label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (len(text) * 11 + 10, 28), (0, 0, 0), -1)
    cv2.putText(out, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1)
    return out


def process_frame(image, blur_kernel=(5, 5), canny_low=50, canny_high=150):
    image = cv2.resize(image, (800, 600))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if blur_kernel is not None:
        blur = cv2.GaussianBlur(gray, blur_kernel, 0)
    else:
        blur = gray.copy()

    edges = cv2.Canny(blur, canny_low, canny_high)
    edge_dilated = cv2.dilate(edges, np.ones((15, 15), np.uint8))

    # Harris
    harris = cv2.cornerHarris(np.float32(gray), 4, 3, 0.04)
    harris = cv2.dilate(harris, None)
    harris_mask = (harris > 0.01 * harris.max()) & (edge_dilated > 0)
    coords = np.argwhere(harris_mask)

    # Shi-Tomasi
    shi_pts = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
    shi_corners = []
    if shi_pts is not None:
        for c in shi_pts:
            x, y = c.ravel().astype(int)
            if 0 <= y < edge_dilated.shape[0] and 0 <= x < edge_dilated.shape[1]:
                if edge_dilated[y, x] > 0:
                    shi_corners.append((x, y))

    # Panels
    p1 = label(t(image), "Original")
    p2 = label(t(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)),
               f"Canny ({canny_low},{canny_high})")

    harris_out = image.copy()
    for y, x in coords[::6]:
        cv2.circle(harris_out, (x, y), 2, (0, 0, 255), -1)
    p3 = label(t(harris_out), "Harris")

    shi_out = image.copy()
    for (x, y) in shi_corners:
        cv2.circle(shi_out, (x, y), 4, (0, 255, 255), -1)
    p4 = label(t(shi_out), "Shi-Tomasi")

    combined = image.copy()
    for y, x in coords[::6]:
        cv2.circle(combined, (x, y), 2, (0, 0, 255), -1)
    for (x, y) in shi_corners:
        cv2.circle(combined, (x, y), 4, (0, 255, 255), 1)

    blur_text = "No Blur" if blur_kernel is None else f"Blur {blur_kernel[0]}x{blur_kernel[1]}"
    p5 = label(t(combined), blur_text)

    empty = np.zeros_like(p1)

    grid = np.vstack([
        np.hstack([p1, p2, p3]),
        np.hstack([p4, p5, empty])
    ])

    return grid, len(coords), len(shi_corners), np.count_nonzero(edges)


# ----------------------------
# Experiment A: threshold tuning
# ----------------------------
threshold_settings = [
    (30, 100),
    (50, 150),
    (80, 200),
    (100, 250)
]

for low, high in threshold_settings:
    grid, harris_count, shi_count, edge_pixels = process_frame(
        image, blur_kernel=(5, 5), canny_low=low, canny_high=high
    )
    save_path = f"experiment_outputs/threshold_{low}_{high}.png"
    cv2.imwrite(save_path, grid)
    print(f"Saved {save_path} | Harris={harris_count}, Shi={shi_count}, EdgePixels={edge_pixels}")


# ----------------------------
# Experiment B: blur tuning
# ----------------------------
blur_settings = [
    None,
    (5, 5),
    (9, 9)
]

for blur_kernel in blur_settings:
    grid, harris_count, shi_count, edge_pixels = process_frame(
        image, blur_kernel=blur_kernel, canny_low=50, canny_high=150
    )

    if blur_kernel is None:
        save_path = "experiment_outputs/blur_none.png"
    else:
        save_path = f"experiment_outputs/blur_{blur_kernel[0]}x{blur_kernel[1]}.png"

    cv2.imwrite(save_path, grid)
    print(f"Saved {save_path} | Harris={harris_count}, Shi={shi_count}, EdgePixels={edge_pixels}")