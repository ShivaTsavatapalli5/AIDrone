import os
import cv2
import numpy as np
import time
from threading import Thread
import importlib.util
from dronekit import connect, VehicleMode, LocationGlobalRelative
import argparse
import geopy.distance
import serial

# Define VideoStream class to handle streaming of video from webcam in a separate processing thread
class VideoStream:
    """Camera object that controls video streaming from the webcam"""
    def __init__(self, resolution=(640, 480), framerate=30):
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3, resolution[0])
        ret = self.stream.set(4, resolution[1])
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                self.stream.release()
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

# Load TensorFlow Lite model
def load_model(use_TPU=True):
    MODEL_NAME = '/home/dronepi/shiva_det/tensorflowliteedgetpu/Sample_TFlite_model'
    GRAPH_NAME = 'detect.tflite'
    LABELMAP_NAME = 'labelmap.txt'
    min_conf_threshold = 0.5
    resolution = '1280x720'
    
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

    return interpreter, input_details, output_details, width, height, floating_model, input_mean, input_std, boxes_idx, classes_idx, scores_idx, labels

# Drone functions
def connectMyCopter():
    parser = argparse.ArgumentParser(description='commands')
    parser.add_argument('--connect')
    args = parser.parse_args()

    connection_string = args.connect
    baud_rate = 57600
    print("\nConnecting to vehicle on: %s" % connection_string)
    vehicle = connect(connection_string, baud=baud_rate, wait_ready=True)
    return vehicle

def arm_and_takeoff(vehicle, aTargetAltitude):
    print("Basic pre-arm checks")
    while not vehicle.is_armable:
        print(" Waiting for vehicle to initialise...")
        time.sleep(1)

    print("Arming motors")
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True
    time.sleep(3)

    print("Taking off!")
    vehicle.simple_takeoff(aTargetAltitude)

    while True:
        print(" Altitude: ", vehicle.location.global_relative_frame.alt)
        if vehicle.location.global_relative_frame.alt >= aTargetAltitude * 0.95:
            print("Reached target altitude")
            break
        time.sleep(1)

def get_distance(cord1, cord2):
    return (geopy.distance.geodesic(cord1, cord2).km) * 1000        

def goto_location(vehicle, to_lat, to_long):
    print(" Global Location (relative altitude): %s" % vehicle.location.global_relative_frame)
    curr_lat = vehicle.location.global_relative_frame.lat
    curr_lon = vehicle.location.global_relative_frame.lon
    curr_alt = vehicle.location.global_relative_frame.alt

    to_alt = curr_alt
    to_point = LocationGlobalRelative(to_lat, to_long, to_alt)
    vehicle.simple_goto(to_point, groundspeed=1)
    
    to_cord = (to_lat, to_long)
    while True:
        curr_lat = vehicle.location.global_relative_frame.lat
        curr_lon = vehicle.location.global_relative_frame.lon
        curr_cord = (curr_lat, curr_lon)
        print("Current location: {}".format(curr_cord))
        distance = get_distance(curr_cord, to_cord)
        print("Distance remaining: {}".format(distance))
        if distance <= 2:
            print("Reached within 2 meters of target location...")
            break
        time.sleep(1)

def send_sms(phone_number, message):
    # Initialize serial connection to SIM900A
    ser = serial.Serial('/dev/ttyAMA0', baudrate=9600, timeout=1)
    
    # Wait for SIM900A to initialize
    time.sleep(2)
    
    # Send AT command to check communication
    ser.write(b'AT\r')
    time.sleep(1)
    response = ser.read(10)
    print("Response:", response.decode())

    # Set SMS to text mode
    ser.write(b'AT+CMGF=1\r')
    time.sleep(1)
    response = ser.read(10)
    print("Response:", response.decode())

    # Send SMS
    ser.write(b'AT+CMGS="' + phone_number.encode() + b'"\r')
    time.sleep(1)
    ser.write(message.encode() + b'\r')
    time.sleep(1)
    ser.write(b'\x1A')  # Send Ctrl+Z to send the message
    time.sleep(3)
    response = ser.read(100)
    print("Response:", response.decode())

    # Close serial connection
    ser.close()

interpreter, input_details, output_details, width, height, floating_model, input_mean, input_std, boxes_idx, classes_idx, scores_idx, labels = load_model()
videostream = VideoStream(resolution=(width, height), framerate=30).start()
time.sleep(1)

    # Initialize drone
vehicle = connectMyCopter()
time.sleep(2)
ht = 7
arm_and_takeoff(vehicle, ht)

coordinates_list = [
    {'latitude': 16.5656374, 'longitude': 81.5216573},  # Example coordinates
    {'latitude': 16.5656255, 'longitude': 81.5217049},  # Example coordinates
    {'latitude': 16.5656727, 'longitude': 81.5217254},  # Example coordinates
    {'latitude': 16.5656901, 'longitude': 81.5216664},  # Example coordinates
      # Example coordinates
    #Add more coordinates as needed
]

phone_number = '+916309038588'  # Replace with recipient's phone number

for coord in coordinates_list:
    latitude = coord['latitude']
    longitude = coord['longitude']
    print(f"Navigating to latitude: {latitude}, longitude: {longitude}")
    goto_location(vehicle, latitude, longitude)
    
    # Check for person detection
    while True:
        frame1 = videostream.read()
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

        person_detected = False
        for i in range(len(scores)):
            if min_conf_threshold < scores[i] <= 1.0:
                class_index = int(classes[i])
                if 0 <= class_index < len(labels) and labels[class_index] == 'person':
                    person_detected = True
                    break
        
        if person_detected:
            time.sleep(5)
            print("Person detected. Sending SMS with current coordinates.")
            curr_lat = vehicle.location.global_relative_frame.lat
            curr_lon = vehicle.location.global_relative_frame.lon
            message = f"Person detected at latitude: {curr_lat}, longitude: {curr_lon}"
            send_sms(phone_number, message)
            print("SMS sent.")
            time.sleep(3)
            break

    time.sleep(3)

print("Returning to Launch")
vehicle.mode = VehicleMode("LAND")
time.sleep(10)

cv2.destroyAllWindows()
videostream.stop()

