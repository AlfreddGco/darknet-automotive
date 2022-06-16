#!/usr/bin/python3
import sys, time, socket
import cv2, json
import numpy as np
from socketing import send_variable_length
from socketing import TCP_PORT
from encoder_stream import EncoderStream

TCP_IP = 'localhost'
if(len(sys.argv) > 1):
  TCP_IP = sys.argv[1]  


if(len(sys.argv) > 2):
  TCP_PORT = int(sys.argv[2])


if(__name__ == '__main__'):
  sock = socket.socket()
  sock.connect((TCP_IP, TCP_PORT))
  print('Connected to socket')
  encoderStream = EncoderStream().start()
  while encoderStream.current_frame is None:
    pass

  print('Started streaming...')
  while encoderStream.running():
    encoded_frame = encoderStream.current_frame
    send_variable_length(sock, encoded_frame)
  sock.close()
