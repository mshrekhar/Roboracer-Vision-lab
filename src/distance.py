import cv2
import numpy as np

def get_cone_corner(img_path):
    img = cv2.imread(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Red color range (two masks needed because red wraps around 0 and 180)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Get the largest contour (the cone)
    largest_cnt = max(contours, key=cv2.contourArea)

    # Find the lower-right corner: Maximize U (x) and V (y)
    # Note: Lab Note 2 says x,y are different from row/col. 
    # In OpenCV: u = col (x), v = row (y)
    u_max = 0
    v_max = 0
    for pt in largest_cnt:
        curr_u, curr_v = pt[0]
        # We want the corner that is furthest right AND lowest down
        if (curr_u + curr_v) > (u_max + v_max):
            u_max, v_max = curr_u, curr_v

    return u_max, v_max

def main():
    # Load K from your previous calibration
    data = np.load('camera_params.npz')
    K = data['K']
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    # 1. Calculate H from cone_x40cm.png
    u40, v40 = get_cone_corner('../resource/cone_x40cm.png')
    x_ref = 400.0 # mm
    H = ((v40 - cy) * x_ref) / fy
    print(f"Detected 40cm cone at pixel: ({u40}, {v40})")
    print(f"Calculated H: {H:.2f} mm")

    # 2. Calculate Distance for cone_unknown.png
    u_unk, v_unk = get_cone_corner('../resource/cone_unknown.png')
    
    # Projection Math
    x_car = (fy * H) / (v_unk - cy)
    y_car = (u_unk - cx) * x_car / fx

    print(f"\n--- Unknown Cone Results ---")
    print(f"Detected Corner: ({u_unk}, {v_unk})")
    print(f"Forward (x_car): {x_car/10:.2f} cm")
    print(f"Lateral (y_car): {y_car/10:.2f} cm")

if __name__ == "__main__":
    main()