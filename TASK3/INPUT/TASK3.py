import cv2
import numpy as np
import os

def detect_obstacles_and_potholes(image_path, output_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Skipping: {image_path} not found.")
        return

    h, w, _ = img.shape
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    output_img = img.copy()

    # -------------------------------------------------------------
    # 1. POTHOLE DETECTION (White Circular Blobs)
    # -------------------------------------------------------------
    # Isolate bright white regions in HSV
    lower_white = np.array([0, 0, 220], dtype=np.uint8)
    upper_white = np.array([180, 25, 255], dtype=np.uint8)
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # Filter out long solid white lane boundary lines
    pothole_contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    pothole_count = 0
    for cnt in pothole_contours:
        area = cv2.contourArea(cnt)
        if 50 < area < 10000:  # Area range for potholes
            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect_ratio = float(bw) / bh
            
            # Potholes are compact oval/circular blobs (aspect ratio close to 1.0)
            if 0.35 <= aspect_ratio <= 2.8 and max(bw, bh) < int(w * 0.25):
                pothole_count += 1
                # Draw Cyan Bounding Box for Potholes
                cv2.rectangle(output_img, (x, y), (x + bw, y + bh), (255, 255, 0), 2)
                coord_text = f"Pothole: ({x},{y})"
                cv2.putText(output_img, coord_text, (x, max(15, y - 5)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

    # -------------------------------------------------------------
    # 2. OBSTACLE DETECTION (Yellow, Blue, Green Cylinders & Crates)
    # -------------------------------------------------------------
    # HSV threshold masks for yellow, green, and blue obstacle colors
    mask_yellow = cv2.inRange(hsv, np.array([15, 100, 100]), np.array([35, 255, 255]))
    mask_green  = cv2.inRange(hsv, np.array([36, 100, 100]), np.array([85, 255, 255]))
    mask_blue   = cv2.inRange(hsv, np.array([90, 100, 100]), np.array([130, 255, 255]))

    # Merge all obstacle color masks
    obstacle_mask = cv2.bitwise_or(mask_yellow, mask_green)
    obstacle_mask = cv2.bitwise_or(obstacle_mask, mask_blue)

    # Clean up minor shadow pixels
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    obstacle_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_OPEN, kernel)

    obstacle_contours, _ = cv2.findContours(obstacle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    obstacle_count = 0
    for cnt in obstacle_contours:
        area = cv2.contourArea(cnt)
        if area > 120:  # Ignore trivial color noise
            x, y, bw, bh = cv2.boundingRect(cnt)
            obstacle_count += 1
            # Draw Red Bounding Box for Obstacles
            cv2.rectangle(output_img, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
            coord_text = f"Obstacle: ({x},{y})"
            cv2.putText(output_img, coord_text, (x, max(15, y - 5)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    # -------------------------------------------------------------
    # 3. OVERLAY TOTAL COUNTS & SAVE
    # -------------------------------------------------------------
    summary_text_1 = f"Total Potholes: {pothole_count}"
    summary_text_2 = f"Total Obstacles: {obstacle_count}"

    # Background banner for scannable counts
    cv2.rectangle(output_img, (10, 10), (280, 65), (0, 0, 0), -1)
    cv2.putText(output_img, summary_text_1, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
    cv2.putText(output_img, summary_text_2, (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    # Save output to file to avoid VS Code window crashes
    cv2.imwrite(output_path, output_img)
    print(f"Processed {image_path} -> Found: {pothole_count} Potholes, {obstacle_count} Obstacles. Saved as {output_path}")

# Run pipeline across all 10 target images (t31 through t310)
for i in range(1, 11):
    # Checks common image extensions
    for ext in ['.png', '.jpg', '.jpeg']:
        input_file = f"t3{i}{ext}"
        if os.path.exists(input_file):
            output_file = f"task3_output_t3{i}.png"
            detect_obstacles_and_potholes(input_file, output_file)
            break