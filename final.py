import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

def main():
    # 1. Load YOLOv8 Segmentation Model
    print("Loading YOLO model...")
    model = YOLO("yolov8l-seg.pt") 

    # 2. Initialize Intel RealSense Pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Enable Depth and Color streams (640x480 resolution at 30 FPS)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start the camera stream
    profile = pipeline.start(config)

    # Get depth scale ( 0.001 meters, meaning 1 unit = 1 millimeter)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    # 3. Setup Alignment and Filters
    # Align depth stream to color stream
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Initialize RealSense post-processing filters
    spatial_filter = rs.spatial_filter()
    hole_filling_filter = rs.hole_filling_filter()

    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not aligned_depth_frame or not color_frame:
                continue

            # Apply filters to clean up the depth data
            filtered_depth = spatial_filter.process(aligned_depth_frame)
            filtered_depth = hole_filling_filter.process(filtered_depth)

            # Convert images to numpy arrays
            depth_image = np.asanyarray(filtered_depth.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # 4. Run YOLOv8 Segmentation on the Color Image
            results = model(color_image, stream=True, verbose=False)

            for r in results:
                # Check if any objects were detected and masks were generated
                if r.masks is not None:
                    masks = r.masks.xy  # Polygon coordinates of the masks
                    boxes = r.boxes.xyxy # Bounding box coordinates
                    classes = r.boxes.cls # Class IDs

                    for i, mask_points in enumerate(masks):
                        # Convert polygon points to integer format for OpenCV
                        pts = np.array(mask_points, dtype=np.int32)

                        # Create a blank black image matching the depth image dimensions
                        blank_mask = np.zeros(depth_image.shape, dtype=np.uint8)

                        # Draw the YOLO mask as a solid white shape (255) on the blank image
                        cv2.fillPoly(blank_mask, [pts], 255)

                        # 5. The Trimmed Mask Strategy (Depth Calculation)
                        # Extract all depth pixel values where our mask is white
                        object_depths = depth_image[blank_mask == 255]

                        # Filter out invalid '0' readings caused by dead pixels or reflections
                        valid_depths = object_depths[object_depths > 0]

                        if len(valid_depths) > 0:
                            # Calculate the median of the valid depth pixels
                            median_depth_raw = np.median(valid_depths)
                            
                            # Convert raw depth units to meters
                            depth_meters = median_depth_raw * depth_scale

                            # --- Drawing Results for Visualization ---
                            class_name = model.names[int(classes[i])]
                            
                            # Draw bounding box
                            x1, y1, x2, y2 = map(int, boxes[i])
                            cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            # Draw object contour (the mask outline)
                            cv2.polylines(color_image, [pts], True, (255, 0, 0), 2)

                            # Put text with Class Name and Depth
                            label = f"{class_name}: {depth_meters:.2f}m"
                            cv2.putText(color_image, label, (x1, y1 - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Show the final image
            cv2.imshow('RealSense + YOLOv8 Seg Depth', color_image)

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()