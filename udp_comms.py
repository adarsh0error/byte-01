import socket
import json

class TargetTracker:
    def __init__(self, pi_ip, port=5005):
        self.pi_ip = pi_ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"📡 UDP Transmitter initialized. Target Pi: {self.pi_ip}:{self.port}")

    def send_target_error(self, face_left, face_right, frame_width=640):
        # 1. Find the center of the face box
        face_center_x = (face_left + face_right) / 2
        
        # 2. Calculate the error from the center of the camera frame
        # If face is left of center, error is negative. Right of center, positive.
        error_x = face_center_x - (frame_width / 2)
        
        # 3. Package and send
        packet = json.dumps({"error_x": round(error_x, 2)}).encode('utf-8')
        try:
            self.sock.sendto(packet, (self.pi_ip, self.port))
            return error_x
        except Exception as e:
            print(f"Network error: {e}")
            return 0.0