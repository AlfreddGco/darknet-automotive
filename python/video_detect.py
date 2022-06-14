from ctypes import *
import sys, time
import cv2, pydash
import numpy as np
from imutils.video import FPS, FileVideoStream


def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

    

#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("./libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

float_to_image = lib.float_to_image
float_to_image.argtypes = [c_int, c_int, c_int, POINTER(c_float)]
float_to_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res


def detect(net, meta, im, thresh=.5, hier_thresh=.5, nms=.45):
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    out = []
    for detection in res:
        t = {
            'label': detection[0],
            'confidence': detection[1],
            'position': detection[2],
        }
        out.append(t)
    free_image(im)
    free_detections(dets, num)
    return out


def cv_img_to_darknet_img(img):
    h, w, c = img.shape

    img = img.transpose(2, 0, 1)[::-1]
    img = img.astype(c_float)

    img = (img / 255).flatten()
    
    c_pointer = img.ctypes.data_as(POINTER(c_float))
    darknet_img = float_to_image(w, h, c, c_pointer)
    return darknet_img


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
        if(detection['confidence'] >= THRESHOLDS[detection['label'].decode()]):
            filtered.append(detection)
    return filtered


def resize_img(frame):
    h, w, c = frame.shape
    frame = frame[h//3:]
    fr = (frame.shape[0] // 2)
    frame = np.concatenate((frame[:,:fr], frame[:,-fr:]), axis=1)
    frame = cv2.resize(frame, (416, 416))
    return frame


if __name__ == "__main__":
    net = load_net(b"cfg/ois.cfg", b"weights/ois_final.weights", 0)
    meta = load_meta(b"cfg/ois.data")

    video_stream = None
    if(len(sys.argv) > 1):
        video_stream = FileVideoStream(sys.argv[1]).start()
    else:
        video_stream = FileVideoStream('data/stop_video.mp4').start()
    time.sleep(1)
    
    last_detected = None
    timer = FPS().start()
    while video_stream.more():
        frame = video_stream.read()
        if(frame is None):
            break
        # cv2.imshow('frame', frame)
        # cv2.waitKey(1)
        h, w, c = frame.shape
        frame = resize_img(frame)
        img = cv_img_to_darknet_img(frame)
        r = detect(net, meta, img)
        r = probability_filter(r)
        if(len(r) > 0 and r[0]['label'] != last_detected):
            print(pydash.collections.map_(r, 'label'), pydash.collections.map_(r, 'confidence'))
            last_detected = r[0]['label']
        timer.update()

    timer.stop()
    print("Detection time: {:.3f} seconds. {:.2f} FPS".format(timer.elapsed(), timer.fps()))

