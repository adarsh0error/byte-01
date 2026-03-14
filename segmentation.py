import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

# Load segmentation model
model = YOLO("yolov8m-seg.pt")

# RealSense pipeline
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

        # Get frames
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Convert to numpy
        depth_image = np.asanyarray(depth_frame.get_data())
        frame = np.asanyarray(color_frame.get_data())

        # Run YOLO segmentation
        results = model(frame, conf=0.5)

        for r in results:
            if r.masks is None:
                continue

            boxes = r.boxes.xyxy.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            masks = r.masks.data.cpu().numpy()

            for box, cls, mask in zip(boxes, classes, masks):

                x1, y1, x2, y2 = map(int, box)

                # Resize mask to match frame
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

                # Convert mask to boolean
                mask_pixels = mask > 0.5

                # Extract depth values using mask
                depth_values = depth_image[mask_pixels]

                # Remove invalid depth
                depth_values = depth_values[(depth_values > 0) & (depth_values < 5000)]

                if len(depth_values) == 0:
                    continue

                # Median depth
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

                # Draw segmentation mask overlay
                colored_mask = np.zeros_like(frame)
                colored_mask[mask_pixels] = (0,255,0)

                frame = cv2.addWeighted(frame, 1.0, colored_mask, 0.3, 0)

        cv2.imshow("YOLOv8 Segmentation + Depth", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:

    pipeline.stop()
    cv2.destroyAllWindows()