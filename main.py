import cv2
import numpy as np

# Load image
image = cv2.imread("image2.jpeg")  # change path
image = cv2.resize(image, (800, 600))
output = image.copy()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Adaptive threshold (better than Canny for your case)
thresh = cv2.adaptiveThreshold(
    blur, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    11, 2
)

# Morphological closing (fills gaps)
kernel = np.ones((5, 5), np.uint8)
morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# Find contours
contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort by area
contours = sorted(contours, key=cv2.contourArea, reverse=True)

card_contour = None

for cnt in contours:
    area = cv2.contourArea(cnt)
    
    if area < 60000:
        continue

    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h

    if not (0.5 < aspect_ratio < 0.9):
        continue

    epsilon = 0.02 * cv2.arcLength(cnt, True)
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h

    if not (0.6 < aspect_ratio < 0.9):
        continue

    approx = cv2.approxPolyDP(cnt, epsilon, True)

    if len(approx) == 4:
        card_contour = approx
        break

# Draw edges (from morph mask)
edges = cv2.Canny(morph, 50, 150)

# Draw results
cv2.drawContours(output, contours, -1, (0, 255, 0), 1)

if card_contour is not None:
    cv2.drawContours(output, [card_contour], -1, (255, 0, 0), 3)

    # Draw corners
    for point in card_contour:
        x, y = point[0]
        cv2.circle(output, (x, y), 8, (0, 0, 255), -1)

# Show results
cv2.imshow("Threshold", thresh)
cv2.imshow("Edges", edges)
cv2.imshow("Final Output", output)

cv2.waitKey(0)
cv2.destroyAllWindows()



