import struct
import socket
import time
import uuid
import threading
import datetime
import serial
import pynmea2
import requests
import cv2
import numpy as np

import json
import websocket
import websockets
import asyncio

#import matplotlib.pyplot as plt
import math
#from geopy.distance import geodesic

# Define the packet format using struct
packet_format = struct.Struct('!d d h h H H I')
#packet_format = struct.Struct('!f f h h H H I')


# Define the IP address and port number for broadcasting
broadcast_address = '192.168.1.255'
port = 5000

# Create a UDP socket for broadcasting and receiving messages
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
sock.bind(('0.0.0.0', port))

# Initialize global variables
global_latitude = 0.0
global_longitude = 0.0
global_speed_mps = 10.0

accel = 9.81 #hardcoded
gps_data1 = []
gps_data2 = []

#Store data in arrays

def store_gps_data(gps_array, latitude, longitude):
    if len(gps_array) < 20:
        gps_array.append((latitude, longitude))
        print(f"[{time.strftime('%H:%M:%S')}] Recent GPS point added to the array")
        #print(f"[{time.strftime('%H:%M:%S')}] ", gps_array)
        
    else:
        gps_array.pop(0)
        gps_array.append((latitude, longitude))
        print(f"[{time.strftime('%H:%M:%S')}] Recent GPS point added to the array")
        #print(f"[{time.strftime('%H:%M:%S')}] ", gps_array)


#-------------------------------------------------------------------------------------------

def kalman_filter(gps_data_array):

  class KalmanFilter:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

    def predict(self, coordX, coordY):
        ''' This function estimates the position of the object'''
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()

        x, y = int(predicted[0]), int(predicted[1])
        return x, y


  # Kalman Filter
  kf = KalmanFilter()
  path = []

  predicted = (0, 0)
  for pt in gps_data_array:
      predicted = kf.predict(pt[0], pt[1])

  path.append((predicted[1], predicted[0]))

  for i in range(15):
      predicted = kf.predict(predicted[0], predicted[1])
      path.append(predicted)

  return path


def transform_gps_data(gps_data):
    gps_data_new = []
    for lat, lon in gps_data:
        lat_new = float((lat - 6.07) * 1000000000000000)
        lon_new = float((lon - 80.19) * 1000000000000000)
        gps_data_new.append((lat_new, lon_new))
    return gps_data_new


def haversine(coord1, coord2):
    lat1, lon1 = coord1
    lat2, lon2 = coord2[0]
    radius_of_earth = 6371  # in kilometers

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) *
         math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)

    c = 2 * math.asin(math.sqrt(a))

    distance = radius_of_earth * c #in km
    distance_m = radius_of_earth * c * 1000 #in m
    return distance_m


def get_intersections(x0, y0, r0, x1, y1, r1):
    # circle 1: (x0, y0), radius r0
    # circle 2: (x1, y1), radius r1

    d=math.sqrt((x1-x0)**2 + (y1-y0)**2)

    # non intersecting
    if d > r0 + r1 :
        return None
    # One circle within other
    if d < abs(r0-r1):
        return None
    # coincident circles
    if d == 0 and r0 == r1:
        return None
    else:
        a=(r0**2-r1**2+d**2)/(2*d)
        h=math.sqrt(r0**2-a**2)
        x2=x0+a*(x1-x0)/d
        y2=y0+a*(y1-y0)/d
        x3=x2+h*(y1-y0)/d
        y3=y2-h*(x1-x0)/d

        x4=x2-h*(y1-y0)/d
        y4=y2+h*(x1-x0)/d

        return (x3, y3, x4, y4)


def reverse_transform_gps_data(nearest_intersection):
    lat, lon = nearest_intersection

    lat_new = round(((lat - 1) / 1000000000000000) + 6.07)
    lon_new = round((lon / 1000000000000000) + 80.19)
    nearest_intersection = (lat_new, lon_new)
    return nearest_intersection



def get_distance_to_interset(intersections, last_gps_data_new):
    if intersections is not None:
        # Extract the intersection coordinates
        i_x3, i_y3, i_x4, i_y4 = intersections

        # Calculate the squared distances from the last predicted point to both intersections
        distance_squared_1 = (last_gps_data_new[1] - i_x3) ** 2 + (last_gps_data_new[0] - i_y3) ** 2
        distance_squared_2 = (last_gps_data_new[1] - i_x4) ** 2 + (last_gps_data_new[0] - i_y4) ** 2

        # Find the intersection closest to the last predicted point
        if distance_squared_1 < distance_squared_2:
            nearest_intersection = (i_y3,i_x3)
        else:
            nearest_intersection = (i_y4, i_x4)

        #print("Nearest Intersection:",nearest_intersection)
        # Calculate the distance between the nearest intersection and the last predicted point
        reversed_nearest_intersection = reverse_transform_gps_data(nearest_intersection)
        # print(reversed_nearest_intersection)
        # distance = geodesic((reversed_nearest_intersection[0], reversed_nearest_intersection[1]), (gps_data1[-1][0], gps_data1[-1][1])).meters
        distance = geodesic(gps_data1[-1], reversed_nearest_intersection).meters

        # distance = haversine(gps_data1[-1], reversed_nearest_intersection)
        print("Distance in meters:", distance)


        return nearest_intersection, distance
    else:
        return None, None


def collision_warning(intersections):
    if intersections is not None:
        print("Collision Detected")
        

def reverse_transform_gps_data_array(coordinates_array):
    transformed_coordinates = []

    for lat, lon in coordinates_array:
        lat_new = round(((lat) / 1000000000000000) + 6.07, 15)
        lon_new = round((lon / 1000000000000000) + 80.19, 15)
        transformed_coordinates.append((lat_new, lon_new))
    
    return transformed_coordinates

def format_kalman_predictions_v1(transformed_coordinates):
    output_data = []

    for i, (lat, lon) in enumerate(transformed_coordinates):
        location_name = f"Location {2 * i + 1}"
        output_data.append({"name": location_name, "lat": lat, "lon": lon})
    
    return output_data


def format_kalman_predictions_v2(transformed_coordinates):
    output_data = []

    for i, (lat, lon) in enumerate(transformed_coordinates):
        location_name = f"Location {2 * i + 2}"
        output_data.append({"name": location_name, "lat": lat, "lon": lon})
    
    return output_data


async def send_data(uri,payload):
    
    payload_json = json.dumps(payload)
    
    async with websockets.connect(uri) as websocket:
        await websocket.send(payload_json)
        #print(f"Sent: {payload_json}")



###########################################################################################

def send_packets():
    seq_num=0
    try:
        now = int(time.time())
        data = (global_latitude, global_longitude,int(accel*100),int(global_speed_mps*100), seq_num, int.from_bytes(uuid.uuid4().bytes[:2], byteorder='big'), now)
        packet = packet_format.pack(*data)
        sock.sendto(packet, (broadcast_address, port))
        packet_size = len(packet)
        #print(f"[{time.strftime('%H:%M:%S')}] Sent packet with seq_num={global_latitude},{global_longitude} and uuid={data[4]}: size={packet_size} bytes")
        seq_num += 1
        time.sleep(0.1)  # Send a packet every 1 seconds
    except Exception as e:
        print("Exception:", e)

# Define a function to receive packets
#def receive_packets():

 #   while True:
  #      print("Waiting for reciving packets ....")
   #     message, address = sock.recvfrom(1024)
    #    if address[0] != '192.168.1.4':  # Check if the packet is from a different IP address
     #       print("Packet recived..")
      #      unpacked_data = packet_format.unpack(message)
       #     received_lat = unpacked_data[0]
       #     received_lon = unpacked_data[1]
       #     received_speed = unpacked_data[2] / 100
       #     received_seq_num = unpacked_data[3]
       #     received_uuid = unpacked_data[4]
       #     received_time = datetime.datetime.fromtimestamp(unpacked_data[5]).strftime("%Y-%m-%d %H:%M:%S")
       #     print(f"Received message from {address}: lat={received_lat:.9f}, lon={received_lon:.9f}, speed={received_speed:.5f}, seq_num={received_seq_num}, uuid={received_uuid}, time={received_time}")
       #     store_gps_data(gps_data2, received_lat, received_lon)
       # else:
           # print("Packet no recived")


def receive_packets():
    while True:
      try:
        #print("Waiting for receiving packets ...")
        message, address = sock.recvfrom(1024)
        if address[0] != '192.168.1.20' and address[0] != '192.168.1.21':
        #if address[0] != '192.168.1.20':  # Check if the packet is from a different IP address
            print("Packet received...")
            unpacked_data = packet_format.unpack(message)
            received_lat = unpacked_data[0]
            received_lon = unpacked_data[1]
            received_speed = unpacked_data[2] / 100
            received_seq_num = unpacked_data[3]
            received_uuid = unpacked_data[4]
            received_time = datetime.datetime.fromtimestamp(unpacked_data[5]).strftime("%Y-%m-%d %H:%M:%S")
            print(f"{time.strftime('%H:%M:%S')}] Received message from {address}: lat={received_lat:.9f}, lon={received_lon:.9f}, speed={received_speed:.5f}, seq_num={received_seq_num}, uuid={received_uuid}, time={received_time}")
            store_gps_data(gps_data2, received_lat, received_lon)
            print(f"[{time.strftime('%H:%M:%S')}] Vehicle 2 GPS data added to array")
            
            data2 = {
                "latitude": received_lat,
                "longitude": received_lon,
                
            }
            
            # Prepare data for HTTP POST request
            #data2 = {
               # "latitude": received_lat,
              #  "longitude": received_lon
             #   }

            # Send HTTP POST request
            response2 = requests.post("http://localhost:1880/gps2", json=data2)
            #print("HTTP POST Response 22:", response2.text)
            
      except Exception as e:
        print("Exception:", e)


# Create a serial connection to the GPS module
port2 = "/dev/ttyACM0" # port = "/dev/ttyAMA0"
ser = serial.Serial(port2, baudrate=9600, timeout=0.5)
print(f"[{time.strftime('%H:%M:%S')}] Connected to {port2}")


# Define a function to extract GPS data
def extract_gps_data():
    try:
        #port2 = "/dev/ttyACM0" 
        #ser = serial.Serial(port2, baudrate=9600, timeout=0.5)
        

        # print("Starting getting GPS data...")
        global global_latitude, global_longitude, global_speed_mps
        
        # print("Reading Serial..")
        newdata = ser.readline()

        try:
            newdata = newdata.decode('ISO-8859-1')
        except UnicodeDecodeError:
            newdata = newdata.decode('cp1252')

        if newdata[0:6] == "$GNRMC":
            msg = pynmea2.parse(newdata)

            # Extract information from the message
            global_latitude = msg.latitude
            global_longitude = msg.longitude
            speed_knots = msg.spd_over_grnd
            cog = msg.true_course

            try:
                global_speed_mps = round(float(speed_knots) * 0.514444, 4)
            except (ValueError, TypeError):
                global_speed_mps = None

            # Convert speed from knots to meters per second and limit to 4 decimals

            print("[{}]Latitude: {}".format(time.strftime('%H:%M:%S'),global_latitude))
            # print("Latitude: {}".format(global_latitude))
            # print("Longitude: {}".format(global_longitude))
            # print("Speed Over Ground: {} knots".format(global_speed_mps))
            # print("Course Over Ground: {} degrees".format(cog))

            #print("Latitude: {}\nLongitude: {}\nSpeed Over Ground: {} knots\nCourse Over Ground: {} degrees".format(global_latitude, global_longitude, global_speed_mps, cog))

            store_gps_data(gps_data1, global_latitude, global_longitude)
            send_packets()
            print(f"[{time.strftime('%H:%M:%S')}] Packet sent")
            
            

        elif newdata.startswith("$GNGGA"):
            msg = pynmea2.parse(newdata)

            satellites = msg.num_sats
            fix_status = msg.gps_qual

            # Prepare data for HTTP POST request
            data = {
                "latitude": global_latitude,
                "longitude": global_longitude,
                "speed": global_speed_mps if global_speed_mps is not None else 0.0,
                "status": fix_status,
                "sat": satellites
            }

            # Send HTTP POST request
            response = requests.post("http://localhost:1880/gps", json=data)
            #print("HTTP POST Response:", response.text)
            #uri = "ws://localhost:1880/ws/mygps"
            #asyncio.run(send_data(uri, data))
            


    except serial.SerialException as e:
        print(f"Error: {e}")


def data_trasfer():
    #print("Starting data transfer...")
    while True:
        try:
            extract_gps_data()
            #print(f"[{time.strftime('%H:%M:%S')}] Extracted GPS data")
            #send_packets()
            #print(f"[{time.strftime('%H:%M:%S')}] Packet sent")
            

        except Exception as e:
            print("Exception:", e)
   

def collision_detect_th():
      print(f"[{time.strftime('%H:%M:%S')}] Starting Collision Detection....")
      while True:
        print(f"[{time.strftime('%H:%M:%S')}] Chechking the speed and data...")
        if (len(gps_data1)>8):
          print(f"[{time.strftime('%H:%M:%S')}] Starting Detection")
          gps_data1_new = transform_gps_data(gps_data1)
          gps_data2_new = transform_gps_data(gps_data2)

          path1=kalman_filter(gps_data1_new)
          path2=kalman_filter(gps_data2_new)

          
          
          converted_kalman_predictions_v1 = reverse_transform_gps_data_array(path1)
          converted_kalman_predictions_v2 = reverse_transform_gps_data_array(path2)
          formated_kalman_data_v1 = format_kalman_predictions_v1(converted_kalman_predictions_v1)
          formated_kalman_data_v2 = format_kalman_predictions_v2(converted_kalman_predictions_v2)
          
          #print(f"[{time.strftime('%H:%M:%S')}] predicted one :",formated_kalman_data_v1[0])
          #print(f"[{time.strftime('%H:%M:%S')}] predicted 22 :",formated_kalman_data_v1[1])
          
          #uri11 = "ws://localhost:1880/ws/predictionV11"
          #asyncio.run(send_data(uri11, formated_kalman_data_v1[0]))
          
          #uri12 = "ws://localhost:1880/ws/predictionV12"
          #asyncio.run(send_data(uri12, formated_kalman_data_v1[1]))
          #asyncio.run(send_data(uri, formated_kalman_data_v1[2]))
          #asyncio.run(send_data(uri, formated_kalman_data_v1[3]))
          
          #uri = "ws://localhost:1880/ws/predictionV2"
          #asyncio.run(send_data(uri, formated_kalman_data_v2))
    
          
               
          # Get the last predicted points
          last_predicted = path1[-1]
          last_predicted2 = path2[-1]

          # Generate and plot the circle
          radius=50000000000 #Not in meters

          intersections = get_intersections(last_predicted[1], last_predicted[0], radius, last_predicted2[1], last_predicted2[0], radius)
          print_collision_warning=collision_warning(intersections)
            
          last_gps_data1 = gps_data1[-1]
          intersect_distance = get_distance_to_interset(intersections, last_gps_data1)

          # plt.ylabel('longitude')
          # plt.xlabel('latitude')
          # plt.title('GPS Positions with Kalman Filter Predictions')
          # plt.gca().set_aspect('equal', adjustable='box')

          # plt.show()
          #datapath1 = path1 

          #print("V2 Original GPS data:   ", gps_data2)
          #print("V2 Kalman predictions:   ", converted_kalman_predictions_v2)

          
          #response4 = requests.post("http://localhost:1880/test12", json=formated_kalman_data_v2)
          #response5 = requests.post("http://localhost:1880/test12", json=formated_kalman_data_v2[10])
          #response5 = requests.post("http://localhost:1880/test12", json=formated_kalman_data_v2[2])
          #response5 = requests.post("http://localhost:1880/test12", json=formated_kalman_data_v2[3])
          #response5 = requests.post("http://localhost:1880/test12", json=formated_kalman_data_v2[4])
          
          #response3 = requests.post("http://localhost:1880/test", json=formated_kalman_data_v1)
          
          

        #time.sleep()


try:
    # Start the threads
    data_trasfer_thread = threading.Thread(target=data_trasfer)
    receive_packets_thread = threading.Thread(target=receive_packets)
    collision_detect_th_thread = threading.Thread(target=collision_detect_th)

    data_trasfer_thread.start()
    receive_packets_thread.start()
    collision_detect_th_thread.start()

    # Wait for threads to finish
    data_trasfer_thread.join()
    receive_packets_thread.join()
    collision_detect_th_thread.join()
    
except KeyboardInterrupt:
    print(f"[{time.strftime('%H:%M:%S')}] Exiting due to keyboard interrupt...")
    
    
#finally:
    # Close the socket
    #sock.close()

    # Close the serial connection
    #ser.close()



