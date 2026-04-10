import cv2
import numpy as np
import os

def detect_lanes(image):
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image}")
    else:
        img = image.copy()

    output = img.copy()
    h, w = img.shape[:2]

    # ROI: mostly floor region
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    polygon = np.array([[
        (int(0.18 * w), int(0.42 * h)),
        (int(0.82 * w), int(0.42 * h)),
        (w, h),
        (0, h)
    ]], dtype=np.int32)
    cv2.fillPoly(roi_mask, polygon, 255)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([15, 40, 80])
    upper_yellow = np.array([45, 255, 255])

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask = cv2.bitwise_and(mask, roi_mask)

    kernel_open = np.ones((3, 3), np.uint8)
    kernel_close = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 12:
            continue

        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw < 4 or ch < 3:
            continue

        aspect_ratio = cw / float(ch)
        if aspect_ratio < 0.7 or aspect_ratio > 4.5:
            continue

        cx = x + cw / 2.0
        cy = y + ch / 2.0

        # Stronger center filtering:
        # real markers are near the image center, not near the left tape edge
        if abs(cx - w / 2.0) > 0.10 * w:
            continue

        candidates.append((cnt, cx, cy, area))

    # Optional: keep only contours forming a vertical chain
    if len(candidates) >= 3:
        pts = np.array([[c[1], c[2]] for c in candidates], dtype=np.float32)

        # Fit x = m*y + b because markers align roughly vertically
        ys = pts[:, 1]
        xs = pts[:, 0]
        m, b = np.polyfit(ys, xs, 1)

        filtered = []
        for cnt, cx, cy, area in candidates:
            expected_x = m * cy + b
            if abs(cx - expected_x) < 0.06 * w:
                filtered.append(cnt)
    else:
        filtered = [c[0] for c in candidates]

    for cnt in filtered:
        cv2.drawContours(output, [cnt], -1, (0, 255, 0), 2)
        overlay = output.copy()
        cv2.drawContours(overlay, [cnt], -1, (0, 255, 0), -1)
        cv2.addWeighted(overlay, 0.25, output, 0.75, 0, output)

    return output


def main():
    input_path = "../resource/lane.png"
    output_path = "lane_detected.jpg"

    if not os.path.exists(input_path):
        print(f"Input not found: {os.path.abspath(input_path)}")
        return

    result = detect_lanes(input_path)
    cv2.imwrite(output_path, result)
    print(f"Saved result to {output_path}")


if __name__ == "__main__":
    main()
