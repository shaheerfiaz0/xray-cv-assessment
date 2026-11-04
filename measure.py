import cv2
import numpy as np
import os

def measure_object(image_path, ref_known_width_mm=50.0):
    if not os.path.exists("calibration_data.npz"):
        print("[ERROR] 'calibration_data.npz' not found! Run calibrate.py first.")
        return
    
    calib = np.load("calibration_data.npz")
    mtx, dist = calib['mtx'], calib['dist']

    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Could not read image at {image_path}")
        return

    h, w = img.shape[:2]

    # 1. Undistort image without aggressive cropping
    newcameramtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
    undistorted = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # 2. Preprocessing & Edge Detection
    gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    edges = cv2.Canny(blurred, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)

    # 3. Find Contours from Edges
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("[WARNING] No objects found via edge detection.")
        return

    print(f"[INFO] Found {len(contours)} contours.")

    # 4. Filter out contours touching image borders
    valid_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        if 1500 < area < (w * h * 0.75):
            x, y, cw, ch = cv2.boundingRect(c)
            if x > 15 and y > 15 and (x + cw) < (w - 15) and (y + ch) < (h - 15):
                valid_contours.append(c)

    if not valid_contours:
        print("[WARNING] No inner object contours found. Using largest valid contour.")
        valid_contours = [c for c in contours if cv2.contourArea(c) < (w * h * 0.9)]
        if not valid_contours:
            print("[ERROR] Failed to isolate object.")
            return
        c = max(valid_contours, key=cv2.contourArea)
    else:
        c = max(valid_contours, key=cv2.contourArea)

    # 5. Measurement Calculations
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = box.astype(np.intp) # Fixed NumPy int0 compatibility issue

    pixel_width = rect[1][0]
    pixel_height = rect[1][1]

    if pixel_width == 0 or pixel_height == 0:
        print("[ERROR] Invalid bounding box dimensions.")
        return

    mm_per_pixel = ref_known_width_mm / max(pixel_width, pixel_height)
    real_width_mm = pixel_width * mm_per_pixel
    real_height_mm = pixel_height * mm_per_pixel

    print(f"\n--- Measurement Results ---")
    print(f"Pixel Width: {pixel_width:.2f} px, Height: {pixel_height:.2f} px")
    print(f"Calculated Scale: {mm_per_pixel:.4f} mm/px")
    print(f"Target Width: {real_width_mm:.2f} mm")
    print(f"Target Height: {real_height_mm:.2f} mm")

    # 6. Draw results
    cv2.drawContours(undistorted, [box], 0, (0, 255, 0), 3)
    cv2.putText(undistorted, f"{real_width_mm:.1f}x{real_height_mm:.1f}mm", 
                (int(rect[0][0]), int(rect[0][1])), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Measurement Result", undistorted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_img = r"IMG_20261104_163720_870.jpg.jpeg"
    if os.path.exists(test_img):
        measure_object(test_img)
    else:
        print(f"[ERROR] Test image not found at: {test_img}")