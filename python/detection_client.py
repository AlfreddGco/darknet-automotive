#!/usr/bin/python3
import sys, time, socket
import cv2, json
import numpy as np
from socketing import recv_variable_length, send_variable_length
from socketing import TCP_PORT
from threading import Thread

TCP_IP = 'localhost'
if(len(sys.argv) > 1):
  TCP_IP = sys.argv[1]  

c = 0
avg = 0

class EncoderStream:
  def __init__(self):
    self.current_frame = None
    self.cap = None
    self.ENCODE_PARAM = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
  
  def start(self):
    self.cap = cv2.VideoCapture(0)
    t = Thread(target = self.update_frame, args=())
    t.daemon = True
    t.start()
    return self

  def update_frame(self):
    while self.running():
      ret, frame = self.cap.read()
      if(not ret or frame is None):
        self.stop()
        return
      
      #frame = self.crop_aoi_img(frame)
      frame = self.resize_img(frame)
      _, imgencode = cv2.imencode('.jpg', frame, self.ENCODE_PARAM)
      data = np.array(imgencode)
      self.current_frame = data.tobytes()

  def stop(self):
    self.cap.release()
  
  def running(self):
    return self.cap.isOpened()

  def crop_aoi_img(self, frame):
    h = frame.shape[0]
    frame = frame[h//3:]
    fr = (frame.shape[0] // 2)
    frame = np.concatenate((frame[:,:fr], frame[:,-fr:]), axis = 1)
    return frame

  def resize_img(self, frame):
    frame = cv2.resize(frame, (416, 416))
    return frame


if(__name__ == '__main__'):
  sock = socket.socket()
  sock.connect((TCP_IP, TCP_PORT))
  encoderStream = EncoderStream().start()
  while encoderStream.current_frame is None:
      pass

  while encoderStream.running():
    encoded_frame = encoderStream.current_frame
    start = time.time()
    send_variable_length(sock, encoded_frame)
    detection = recv_variable_length(sock).decode()
    detection = json.loads(detection)
    end = time.time()
    avg = (avg*c + (end - start))/(c + 1)
    c += 1
    if(len(detection) > 0):
      print(detection, avg)
  sock.close()
