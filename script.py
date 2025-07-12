import cv2
import numpy as np
import threading
import json
from websocket_server import WebsocketServer
import math
import os

# List to hold connected clients
clients = []

# WebSocket handlers
def new_client(client, server):
    print(f"Client {client['id']} connected")
    clients.append(client)

def client_left(client, server):
    print(f"Client {client['id']} disconnected")
    if client in clients:
        clients.remove(client)

def run_ws_server():
    global ws_server
    ws_server = WebsocketServer(host='0.0.0.0', port=9000)
    ws_server.set_fn_new_client(new_client)
    ws_server.set_fn_client_left(client_left)
    ws_server.run_forever()

# Start WebSocket server in a new thread
ws_thread = threading.Thread(target=run_ws_server)
ws_thread.daemon = True
ws_thread.start()

# OpenCV Camera Feed + Circle Detection
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("http://192.168.2.30:8080/video")

# Load the Earth image with transparency
image_path = "D:/AR/images/earth.jpeg"  # update path if needed
earth = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
if earth is None:
    print(f"ERROR: Earth image not loaded from '{image_path}'. Check the file path.")
    exit()

# Parameters for consolidating circles
MERGE_CENTER_DIST_THRESHOLD = 30
MERGE_RADIUS_DIFF_THRESHOLD = 15

def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    return rotated

def overlay_image_alpha(img, img_overlay, pos, overlay_size):
    x, y = pos
    overlay = cv2.resize(img_overlay, overlay_size, interpolation=cv2.INTER_AREA)

    if overlay.shape[2] < 4:
        print("WARNING: Overlay image doesn't have alpha channel.")
        return

    b, g, r, a = cv2.split(overlay)
    alpha_mask = a / 255.0

    h, w = overlay.shape[:2]
    for c in range(0, 3):
        img[y:y+h, x:x+w, c] = (
            (1. - alpha_mask) * img[y:y+h, x:x+w, c] +
            alpha_mask * overlay[:, :, c]
        )

rotation_angle = 0

while True:
    ret, image = cap.read()
    if not ret:
        continue

    image_width, image_height = 640, 480
    image = cv2.resize(image, (image_width, image_height))

    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(grey, 15)

    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, 1, 30,
        param1=100, param2=40, minRadius=10, maxRadius=200
    )

    consolidated_circles = []

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for detected_circle in circles[0, :]:
            x_det, y_det, r_det = detected_circle[0], detected_circle[1], detected_circle[2]
            merged = False
            for i, existing_circle in enumerate(consolidated_circles):
                x_exist, y_exist, r_exist = existing_circle
                center_distance = dist((x_det, y_det), (x_exist, y_exist))
                radius_difference = abs(r_det - r_exist)
                if center_distance < MERGE_CENTER_DIST_THRESHOLD and \
                   radius_difference < MERGE_RADIUS_DIFF_THRESHOLD:
                    consolidated_circles[i] = (
                        int((x_exist + x_det) / 2),
                        int((y_exist + y_det) / 2),
                        int((r_exist + r_det) / 2)
                    )
                    merged = True
                    break
            if not merged:
                consolidated_circles.append((x_det, y_det, r_det))

    circles_to_send = []

    rotation_angle = (rotation_angle + 2) % 360
    rotated_earth = rotate_image(earth, rotation_angle)

    for x, y, r in consolidated_circles:
        print(f"Consolidated circle: X={x}, Y={y}, R={r}")
        logo_size = (int(r * 2), int(r * 2))
        top_left = (int(x - logo_size[0] / 2), int(y - logo_size[1] / 2))

        if (
            0 <= top_left[0] and top_left[0] + logo_size[0] <= image.shape[1] and
            0 <= top_left[1] and top_left[1] + logo_size[1] <= image.shape[0]
        ):
            overlay_image_alpha(image, rotated_earth, top_left, logo_size)
        else:
            print("WARNING: Logo position out of bounds, skipping overlay.")

        circles_to_send.append({
            'x': x / image_width,
            'y': y / image_height,
            'radius': r / image_width
        })

    if clients:
        try:
            msg = json.dumps(circles_to_send)
            for client in clients:
                ws_server.send_message(client, msg)
        except Exception as e:
            print("Error sending message:", e)

    cv2.imshow("Detection", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
