import numpy as np
import cv2
import glob
import os

def calibrate_camera(images_folder="calib_images", square_size=25.0):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Valid chessboard inner corner sizes (both dimensions must be > 2)
    possible_sizes = [
        (7, 5), (5, 7), 
        (8, 6), (6, 8), 
        (9, 6), (6, 9), 
        (7, 7), (8, 8),
        (4, 11), (11, 4),
        (3, 8), (8, 3)
    ]

    images = []
    for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']:
        images.extend(glob.glob(os.path.join(images_folder, ext)))
    images = list(set(images))

    if not images:
        print(f"[ERROR] No calibration images found in '{images_folder}' folder!")
        return None, None

    print(f"[INFO] Found {len(images)} images for calibration.")

    detected_size = None
    gray_shape = None

    # Find the correct chessboard size using the first successful image
    for size in possible_sizes:
        for fname in images[:5]: # Test first 5 images
            img = cv2.imread(fname)
            if img is None: continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_shape = gray.shape[::-1]
            try:
                ret, _ = cv2.findChessboardCorners(gray, size, None)
                if ret:
                    detected_size = size
                    break
            except Exception:
                continue
        if detected_size:
            break

    if not detected_size:
        print("[ERROR] Could not automatically detect chessboard size. Please check your pattern image grid.")
        return None, None

    print(f"[SUCCESS] Detected Chessboard Inner Corner Size: {detected_size}")

    objp = np.zeros((detected_size[0] * detected_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:detected_size[0], 0:detected_size[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = [] 
    imgpoints = [] 

    for fname in images:
        img = cv2.imread(fname)
        if img is None: continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, detected_size, None)

        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            print(f"[OK] Corners found in: {os.path.basename(fname)}")
        else:
            print(f"[WARNING] Corners NOT found in: {os.path.basename(fname)}")

    if len(objpoints) == 0:
        print("[ERROR] Could not find chessboard corners in any images.")
        return None, None

    # Perform camera calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_shape, None, None)

    print("\n[INFO] Camera Calibration Successful!")
    print(f"Camera Matrix (K):\n{mtx}")
    print(f"Distortion Coefficients:\n{dist}")

    # Save calibration parameters
    np.savez("calibration_data.npz", mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
    print("[INFO] Calibration parameters saved to 'calibration_data.npz'")

    return mtx, dist

if __name__ == "__main__":
    os.makedirs("calib_images", exist_ok=True)
    calibrate_camera()