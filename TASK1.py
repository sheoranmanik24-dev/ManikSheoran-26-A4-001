import cv2
import numpy as np
import os

def preprocess_image(image):
    # Convert to HLS and HSV color spaces to reliably detect white and yellow lanes
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Yellow lane color threshold
    lower_yellow = np.array([10, 40, 80], dtype=np.uint8)
    upper_yellow = np.array([40, 255, 255], dtype=np.uint8)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # White lane color threshold (HLS light channel isolates bright whites in shadows/tunnels)
    lower_white = np.array([0, 160, 0], dtype=np.uint8)
    upper_white = np.array([255, 255, 255], dtype=np.uint8)
    white_mask = cv2.inRange(hls, lower_white, upper_white)

    # High-intensity fallback threshold for dark/night frames
    _, adapt_thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Combine color masks
    combined_color = cv2.bitwise_or(yellow_mask, white_mask)
    combined_color = cv2.bitwise_or(combined_color, adapt_thresh)

    # Edge detection
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 120)

    # Combine edges with color boundaries
    return cv2.bitwise_or(edges, combined_color)

def region_of_interest(image):
    height, width = image.shape[:2]
    
    # Flexible trapezoidal ROI mask focused on lower 55% of the frame
    polygons = np.array([
        [
            (int(width * 0.02), height),
            (int(width * 0.98), height),
            (int(width * 0.58), int(height * 0.45)),
            (int(width * 0.42), int(height * 0.45))
        ]
    ], dtype=np.int32)
    
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    return cv2.bitwise_and(image, mask)

def draw_lanes(image, lines):
    overlay = np.zeros_like(image)
    if lines is None:
        return overlay
    
    # Safe 2D array reshape to prevent array shape unpack errors
    lines_reshaped = lines.reshape(-1, 4)
    height, width = image.shape[:2]
    midpoint = width / 2
    
    left_points = []
    right_points = []
    valid_lines = []

    for x1, y1, x2, y2 in lines_reshaped:
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        
        # Filter horizontal horizontal edges (road borders/refinement noise)
        if abs(slope) < 0.2:
            continue

        valid_lines.append((x1, y1, x2, y2))
        
        # Categorize points by slope direction and position
        if slope < 0 and x1 < midpoint + width * 0.15:
            left_points.append((x1, y1))
            left_points.append((x2, y2))
        elif slope > 0 and x1 > midpoint - width * 0.15:
            right_points.append((x1, y1))
            right_points.append((x2, y2))

    def fit_line(pts):
        if len(pts) < 2:
            return None
        pts = np.array(pts)
        [vx, vy, x, y] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
        y1 = height
        y2 = int(height * 0.52)
        if abs(vy[0]) < 1e-6:
            return None
        x1 = int(x[0] + (y1 - y[0]) * vx[0] / vy[0])
        x2 = int(x[0] + (y2 - y[0]) * vx[0] / vy[0])
        return (x1, y1, x2, y2)

    left_line = fit_line(left_points)
    right_line = fit_line(right_points)

    # Highlight green polygon lane area if both lines are fit
    if left_line is not None and right_line is not None:
        pts = np.array([[
            [left_line[0], left_line[1]],
            [left_line[2], left_line[3]],
            [right_line[2], right_line[3]],
            [right_line[0], right_line[1]]
        ]], dtype=np.int32)
        cv2.fillPoly(overlay, pts, (0, 255, 0))
        cv2.line(overlay, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (255, 0, 0), 8)
        cv2.line(overlay, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (255, 0, 0), 8)
    else:
        # Direct segment rendering fallback for curves and sharp angles
        for x1, y1, x2, y2 in valid_lines:
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 5)

    return overlay

def process_single_image(image_path, output_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Skipping (File not found): {image_path}")
        return

    processed_edges = preprocess_image(image)
    cropped_edges = region_of_interest(processed_edges)

    lines = cv2.HoughLinesP(
        cropped_edges,
        rho=1,
        theta=np.pi/180,
        threshold=25,
        minLineLength=25,
        maxLineGap=120
    )

    overlay = draw_lanes(image, lines)
    result = cv2.addWeighted(image, 0.8, overlay, 0.5, 0)
    
    cv2.imwrite(output_path, result)
    print(f"Successfully processed: {image_path} -> {output_path}")

# List of image filenames to run
image_files = [
    "1.jpg", "1.png", "2.jpg", "2.png", "3.jpg", "3.png",
    "4.jpeg", "4.jpg", "5.jpeg", "5.jpg", "6.jpeg", "6.jpg",
    "7.jpeg", "7.jpg", "8.jpeg", "8.jpg", "9.jpeg", "9.jpg", 
    "10.jpeg", "10.jpg"
]

# Run detection loop
for img_file in image_files:
    if os.path.exists(img_file):
        output_name = f"result_{os.path.splitext(img_file)[0]}.jpg"
        process_single_image(img_file, output_name)