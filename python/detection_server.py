#!/usr/bin/python3
import socket
import sys, cv2, json
import numpy as np
import darknet as dn
from socketing import recv_variable_length, send_variable_length
from socketing import TCP_PORT

if(len(sys.argv) > 1):
  TCP_PORT = int(sys.argv[1])

net = dn.load_net(b"cfg/ois.cfg", b"weights/ois_final.weights", 0)
meta = dn.load_meta(b"cfg/ois.data")

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('0.0.0.0', TCP_PORT))
s.listen(True)

while True:
    conn, addr = s.accept()
    try:
        while True:
            bytedata = recv_variable_length(conn)
            frame = np.frombuffer(bytedata, dtype='uint8')
            frame = cv2.imdecode(frame, 1)
            img = dn.cv_img_to_darknet_img(frame)
            r = dn.detect(net, meta, img)
            send_variable_length(conn, json.dumps(r).encode('utf-8'))
    except:
        print('Client connection closed')
        conn.close()