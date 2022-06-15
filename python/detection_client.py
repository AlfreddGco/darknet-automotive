#!/usr/bin/python3
import socket, time
import cv2, json
import numpy as np
from socketing import recv_variable_length, send_variable_length
from socketing import TCP_PORT
from threading import Thread

TCP_IP = 'localhost'

sock = socket.socket()
sock.connect((TCP_IP, TCP_PORT))

cap = cv2.VideoCapture(0)

c = 0
avg = 0

def resize_img(frame):
  h = frame.shape[0]
  frame = frame[h//3:]
  fr = (frame.shape[0] // 2)
  frame = np.concatenate((frame[:,:fr], frame[:,-fr:]), axis=1)
  frame = cv2.resize(frame, (416, 416))
  return frame


while cap.isOpened():
  start = time.time()
  ret, frame = cap.read()
  if(not ret or frame is None):
    break

  # frame = resize_img(frame)
  encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
  result, imgencode = cv2.imencode('.jpg', frame, encode_param)
  data = np.array(imgencode)
  data = data.tobytes()
  send_variable_length(sock, data)
  detection = recv_variable_length(sock).decode()
  detection = json.loads(detection)
  end = time.time()
  avg = (avg*c + (end - start))/(c + 1)
  c += 1
  if(len(detection) > 0):
    print(detection, avg)

sock.close()