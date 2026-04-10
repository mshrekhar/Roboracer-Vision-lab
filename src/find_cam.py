import cv2

def scan_cameras():
    print("Scanning indices 0 through 10...")
    for i in range(11):
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if cap.isOpened():
            w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print(f"Index {i}: OPENED ({int(w)}x{int(h)})")
            cap.release()
        else:
            print(f"Index {i}: Failed")

if __name__ == "__main__":
    scan_cameras()