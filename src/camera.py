import cv2
import time

def main():
    # Opening with V4L2
    cap = cv2.VideoCapture(4, cv2.CAP_V4L2)
    
    # 1. Apply hardware constraints
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # 2. Warm up the sensor (Crucial for RealSense buffer stability)
    print("Pre-heating sensor and syncing clocks...")
    for _ in range(60):
        cap.read()

    # 3. Query actual hardware state for verification
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    requested_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"\n--- Hardware Verification ---")
    if actual_w == 960 and actual_h == 540:
        print(f"RESOLUTION: {actual_w}x{actual_h} [PASSED]")
    else:
        print(f"RESOLUTION: {actual_w}x{actual_h} [FAILED - Defaulting]")

    print(f"REQUESTED FPS: {requested_fps} Hz")
    print("-----------------------------\n")

    # 4. Benchmark Frequency
    print("Starting 10-second high-speed benchmark...")
    frames = 0
    start_time = time.time()
    
    while (time.time() - start_time) < 10.0:
        ret, _ = cap.read()
        if ret:
            frames += 1
            
    total_time = time.time() - start_time
    actual_hz = frames / total_time

    print(f"\n--- Benchmark Results ---")
    print(f"Total Frames Captured: {frames}")
    print(f"Elapsed Time:          {total_time:.2f}s")
    print(f"Measured Frequency:    {actual_hz:.2f} Hz")
    
    # 5. Final Status Prints for Submission
    print("\n--- Lab Part II Submission Status ---")
    if actual_hz >= 59.5 and actual_w == 960:
        print("RESULT: 60Hz @ 960x540 ACHIEVED SUCCESSFULLY")
    elif actual_hz < 59.5:
        print(f"RESULT: Frequency Gap detected ({actual_hz:.2f} Hz). Lower 'exposure_time_absolute' via v4l2-ctl.")
    else:
        print("RESULT: Resolution Mismatch. Check USB 3.0 connection.")

    cap.release()

if __name__ == "__main__":
    main()