#!/usr/bin/python3
import socket
import sys, cv2, json
import numpy as np
import darknet as dn
from socketing import recv_variable_length
from socketing import TCP_PORT

if(len(sys.argv) > 1):
  TCP_PORT = int(sys.argv[1])


s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('0.0.0.0', TCP_PORT))
s.listen(True)

net = dn.load_net(b"cfg/ois.cfg", b"weights/ois_final.weights", 0)
meta = dn.load_meta(b"cfg/ois.data")

def main():
  while True:
    conn, addr = s.accept()
    try:
      while True:
        bytedata = recv_variable_length(conn)
        frame = np.frombuffer(bytedata, dtype='uint8')
        frame = cv2.imdecode(frame, 1)
        cv2.imshow('Streaming video', frame)
        cv2.waitKey(1)
    except EOFError as err:
      print(err)
      conn.close()
      cv2.destroyAllWindows()


if __name__ == '__main__':
  try:
    main()
  except KeyboardInterrupt:
    s.shutdown()
    cv2.destroyAllWindows()