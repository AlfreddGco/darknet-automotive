#!/usr/bin/python3
import socket
import cv2
import numpy
import darknet as dn

net = dn.load_net(b"cfg/ois.cfg", b"weights/ois_final.weights", 0)
meta = dn.load_meta(b"cfg/ois.data")

def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

TCP_IP = 'localhost'
TCP_PORT = 5001

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((TCP_IP, TCP_PORT))
s.listen(True)
conn, addr = s.accept()

while True:
    length = recvall(conn, 16)
    stringData = recvall(conn, int(length))
    frame = numpy.frombuffer(stringData, dtype='uint8')
    frame = cv2.imdecode(frame, 1)
    img = dn.cv_img_to_darknet_img(frame)
    r = dn.detect(net, meta, img)
    print(r)
s.close()

decimg = cv2.imdecode(frame, 1)
cv2.imshow('SERVER', decimg)
cv2.waitKey(0)
cv2.destroyAllWindows() 
