import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8m.pt")

# Start RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

profile = pipeline.start(config)

# Get depth scale
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

print("Depth Scale:", depth_scale)

try:
    while True:

        # Wait for frames
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Convert frames to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        frame = np.asanyarray(color_frame.get_data())

        # Run YOLO detection
        results = model(frame)

        for r in results:

            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()

            for box, cls in zip(boxes, classes):

                x1, y1, x2, y2 = map(int, box)

                # Shrink bounding box to avoid background
                x_start = int(x1 + 0.25 * (x2 - x1))
                x_end   = int(x2 - 0.25 * (x2 - x1))

                y_start = int(y1 + 0.25 * (y2 - y1))
                y_end   = int(y2 - 0.25 * (y2 - y1))

                # Extract depth region
                depth_region = depth_image[y_start:y_end, x_start:x_end]

                # Flatten to list
                depth_values = depth_region.flatten()

                # Remove invalid depth values
                depth_values = depth_values[depth_values > 0]

                if len(depth_values) == 0:
                    continue

                # Compute median depth
                distance = np.median(depth_values) * depth_scale

                label = model.names[int(cls)]

                text = f"{label} {distance:.2f}m"

                # Draw bounding box
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

                # Draw label
                cv2.putText(frame,
                            text,
                            (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0,255,0),
                            2)

        cv2.imshow("Object Detection + Depth", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:

    pipeline.stop()
    cv2.destroyAllWindows()