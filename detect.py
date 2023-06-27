from ultralytics import YOLO
import time
from udp_sender import UDPSender
from kalman import Kalman
from centroidtracker import CentroidTracker
import cv2
import numpy as np

UDP_PORT = 5053

CAM_INDEX = 0
USE_HIGHRES = False

CAM_WIDTH = 1280 if USE_HIGHRES else 640
CAM_HEIGHT = 720 if USE_HIGHRES else 360

udp_sender = UDPSender(port=UDP_PORT)

ct = CentroidTracker(horizon=900, z_mult=30000)

pTime = 0


def show_fps(image):
    global pTime
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(image, f"{fps:.1f}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)


cv2.namedWindow("Color frame")

def run_using_yolo():
    model = YOLO("yolov8n-face.pt")
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)


    global cam_state, chosen_id
    cam_state = "IDLE"

    kalman_x = Kalman(CAM_WIDTH, 2.5, 30, 1.0, CAM_WIDTH)
    kalman_y = Kalman(CAM_HEIGHT, 2.5, 30, 1.0, CAM_HEIGHT)

    while True:
        success, color_frame = cap.read()

        results = model(color_frame)

        rects = []

        boxes = results[0].boxes

        if boxes:
            for box in boxes:
                top_left_x = int(box.xyxy.tolist()[0][0])
                top_left_y = int(box.xyxy.tolist()[0][1])
                bottom_right_x = int(box.xyxy.tolist()[0][2])
                bottom_right_y = int(box.xyxy.tolist()[0][3])

                #cv2.rectangle(color_frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (50,200,129),2)

                rects.append(xyxy_to_xywh([top_left_x, top_left_y, bottom_right_x, bottom_right_y]))
                print(box.xyxy)


        trackable_objects = ct.update(rects)

        if cam_state == "IDLE":
            chosen_id = choose_person(trackable_objects)
            if chosen_id != -1:
                cam_state = "STALK"

        elif cam_state == "STALK":
            if chosen_id in trackable_objects.keys():

                to = trackable_objects[chosen_id]

                x = to.rect[0] + to.rect[2] / 2
                y = to.rect[1] + to.rect[3] / 2

                cv2.rectangle(color_frame, to.rect, (255, 0, 255))

                x = int(kalman_x.filter(x))
                y = int(kalman_y.filter(y))

                cv2.circle(color_frame, (int(x), int(y)), 4, (0, 0, 255))

                distance = 100

                print(f'{x},{y},{distance:.0f}')
                udp_sender.send(f'{x},{y},{distance:.0f}')
                #udp_sender.send(f'{CAM_WIDTH/2},{CAM_HEIGHT/2},500')

            else:
                cam_state = "IDLE"


        show_fps(color_frame)
        cv2.imshow("Color frame", color_frame)
        key = cv2.waitKey(1)
        if key == 27:
            break

def best_to(area1, area2):
    return area1 > area2

def xyxy_to_xywh(xyxy):
    """
    Convert XYXY format (x,y top left and x,y bottom right) to XYWH format (x,y center point and width, height).
    :param xyxy: [X1, Y1, X2, Y2]
    :return: [X, Y, W, H]
    """
    if np.array(xyxy).ndim > 1 or len(xyxy) > 4:
        raise ValueError('xyxy format: [x1, y1, x2, y2]')
    x_temp = (xyxy[0] + xyxy[2]) / 2
    y_temp = (xyxy[1] + xyxy[3]) / 2
    w_temp = abs(xyxy[0] - xyxy[2])
    h_temp = abs(xyxy[1] - xyxy[3])
    #return np.array([int(x_temp), int(y_temp), int(w_temp), int(h_temp)])
    return np.array([int(xyxy[0]), int(xyxy[1]), int(w_temp), int(h_temp)])


def choose_person(trackable_objects):
    result = -1
    larger_area = 0

    for to_id, to in trackable_objects.items():

        sx = to.rect[0] - to.rect[2]
        sy = to.rect[1] - to.rect[3]
        area = sx * sy

        if best_to(area, larger_area):
            larger_area = area
            result = to_id

    return result

if __name__ == "__main__":
    run_using_yolo()




