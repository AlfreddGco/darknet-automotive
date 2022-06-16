#!/usr/bin/python3
import sys, time
import cv2
import numpy as np
from threading import Thread

class EncoderStream:
  def __init__(self):
    self.current_frame = None
    self.cap = None
    self.ENCODE_PARAM = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
  
  def start(self, src = 0):
    self.cap = cv2.VideoCapture(src)
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