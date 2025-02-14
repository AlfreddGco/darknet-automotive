#!/usr/bin/python3
import sys, time, socket
import cv2, json
import numpy as np
from socketing import recv_variable_length, send_variable_length
from socketing import TCP_PORT
from encoder_stream import EncoderStream

TCP_IP = 'localhost'
if(len(sys.argv) > 1):
  TCP_IP = sys.argv[1]  


if(len(sys.argv) > 2):
  TCP_PORT = int(sys.argv[2])

c, avg = 0, 0

if(__name__ == '__main__'):
  sock = socket.socket()
  sock.connect((TCP_IP, TCP_PORT))
  print('Connected to socket')
  encoderStream = EncoderStream().start()
  while encoderStream.current_frame is None:
      pass

  print('Started detecting...')
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
      print(detection)
      print('{:.2f}ms current, {:.2f}ms avg'.format((end-start)*1000, avg*1000))
  sock.close()
  encoderStream.stop()
