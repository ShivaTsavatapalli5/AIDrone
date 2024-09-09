from dronekit import connect, VehicleMode
import time
import argparse
import serial
import os
import cv2
import numpy as np
import importlib.util

def connectMyCopter():
    parser = argparse.ArgumentParser(description='Commands to connect to a drone')
    parser.add_argument('--connect', help="Connection string (e.g., 'udp:127.0.0.1:14550')")
    args = parser.parse_args()

    connection_string = args.connect
    baud_rate = 57600
    if not connection_string:
        raise ValueError("Connection string not provided. Use the --connect argument.")
    
    print("\nConnecting to vehicle on: %s" % connection_string)
    vehicle = connect(connection_string, baud=baud_rate, wait_ready=True)
    return vehicle

def get_gps_location(vehicle):
    gps_location = vehicle.location.global_frame
    return gps_location

def initialize_serial_connection(port='/dev/ttyAMA0', baudrate=9600):
    try:
        ser = serial.Serial(port, baudrate=baudrate, timeout=1)
        time.sleep(2)
        return ser
    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")
        return None

def send_sms(ser, phone_number, message):
    try:
        if ser is None:
            print("Serial connection is not initialized.")
            return

        ser.write(b'AT+CMGF=1\r\n')
        time.sleep(1)
        ser.write(f'AT+CMGS="{phone_number}"\r\n'.encode('utf-8'))
        time.sleep(1)
        ser.write(f'{message}\r\n'.encode('utf-8'))
        ser.write(bytes([26]))  # CTRL+Z to send the message
        response = ser.read_all().decode('utf-8', errors='replace')
        print(f"Response after sending message: {response}")
        return
    except serial.SerialException as e:
        print(f"Serial Error: {e}")
    except Exception as e:
        print(f"Error: {e}")

def capture_frame(video_capture):
    ret, frame = video_capture.read()
    if not ret:
        raise RuntimeError("Failed to capture frame")
    return frame


vehicle = connectMyCopter()
ser = initialize_serial_connection()

MODEL_NAME = '/home/dronepi/shiva_det/tensorflowliteedgetpu/Sample_TFlite_model'
GRAPH_NAME = 'detect.tflite'
LABELMAP_NAME = 'labelmap.txt'
min_conf_threshold = 0.5
resolution = '1280x720'
use_TPU = True

resW, resH = resolution.split('x')
imW, imH = int(resW), int(resH)

pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

if use_TPU and GRAPH_NAME == 'detect.tflite':
    GRAPH_NAME = 'edgetpu.tflite'

CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

if labels[0] == '???':
    del(labels[0])

if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                                experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)
input_mean = 127.5
input_std = 127.5

outname = output_details[0]['name']
if 'StatefulPartitionedCall' in outname:
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else:
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

frame_rate_calc = 1
freq = cv2.getTickFrequency()
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
video_capture.set(3, imW)
video_capture.set(4, imH)
time.sleep(1)

while True:
    t1 = cv2.getTickCount()
    frame1 = capture_frame(video_capture)
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]

    detected_person = False

    for i in range(len(scores)):
        if min_conf_threshold < scores[i] <= 1.0:
            class_index = int(classes[i])
            if 0 <= class_index < len(labels):
                ymin = int(max(1, (boxes[i][0] * imH)))
                xmin = int(max(1, (boxes[i][1] * imW)))
                ymax = int(min(imH, (boxes[i][2] * imH)))
                xmax = int(min(imW, (boxes[i][3] * imW)))
                
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
                object_name = labels[class_index]
                label = '%s: %d%%' % (object_name, int(scores[i] * 100))
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                label_ymin = max(ymin, labelSize[1] + 10)
                cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10), (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255), cv2.FILLED)
                cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                if object_name.lower() == 'person':
                    detected_person = True
            else:
                print(f"Class index {class_index} out of range for labels list.")

    if detected_person:
        gps_location = get_gps_location(vehicle)
        Latitude = gps_location.lat
        Longitude = gps_location.lon
        print(f"Latitude: {Latitude}, Longitude: {Longitude}, Altitude: {gps_location.alt}")
        send_sms(ser, '6309038588', f"https://www.google.com/maps?q={Latitude},{Longitude}")
        time.sleep(1)

    cv2.putText(frame, 'FPS: {0:.2f}'.format(frame_rate_calc), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Object detector', frame)

    t2 = cv2.getTickCount()
    time1 = (t2 - t1) / freq
    frame_rate_calc = 1 / time1

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
video_capture.release()
if ser:
    ser.close()