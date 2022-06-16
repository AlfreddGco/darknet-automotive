from ctypes import *
import sys, time
import cv2, pydash
import numpy as np
from imutils.video import FPS, FileVideoStream

import darknet as dn

def probability_filter(detections):
    detections = pydash.arrays.uniq_by(detections, lambda x: x['label'])
    filtered = []
    THRESHOLDS = {
        'Stop': 0.5,
        'No speed limit': 0.75,
        'Turn right': 0.55,
        'Ahead only': 0.8,
    }
    for detection in detections:
        if(detection['confidence'] >= THRESHOLDS[detection['label']]):
            filtered.append(detection)
    return filtered


def resize_img(frame):
    h = frame.shape[0]
    frame = frame[h//3:]
    fr = (frame.shape[0] // 2)
    frame = np.concatenate((frame[:,:fr], frame[:,-fr:]), axis=1)
    frame = cv2.resize(frame, (416, 416))
    return frame


if __name__ == "__main__":
    net = dn.load_net(b"cfg/ois.cfg", b"weights/ois_final.weights", 0)
    meta = dn.load_meta(b"cfg/ois.data")

    video_stream = None
    if(len(sys.argv) > 1):
        video_stream = FileVideoStream(sys.argv[1]).start()
    else:
        video_stream = FileVideoStream('data/stop_video.mp4').start()
    time.sleep(1)
    
    timer = FPS().start()
    while video_stream.more():
        frame = video_stream.read()
        if(frame is None):
            break
        # cv2.imshow('frame', frame)
        # cv2.waitKey(1)
        h, w, c = frame.shape
        frame = resize_img(frame)
        img = dn.cv_img_to_darknet_img(frame)
        r = dn.detect(net, meta, img)
        r = probability_filter(r)
        if(len(r) > 0):
            labels = pydash.collections.map_(r, 'label')
            confidences = pydash.collections.map_(r, 'confidence')
            pp = list(zip(labels, confidences))
            print(pp)
        timer.update()

    timer.stop()
    print("Detection time: {:.3f} seconds. {:.2f} FPS".format(timer.elapsed(), timer.fps()))
    # print("Conversion average: {:.4f}".format(sum(conversion_times)/len(conversion_times)))

