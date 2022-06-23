![Darknet Logo](http://pjreddie.com/media/files/darknet-black-small.png)

# Darknet automotive #
Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

Darknet automotive is darknet specialized in autonomous driving machine learning algorithms. The idea is to provide machine learning models for autonomous driving, training environments, testing simulations, fast and lightweight code, and everything built on top of darknet.

# OIS dataset #
OIS (Objects in Street) is a dataset built for traffic signs detection. I know, the name is dope. 


## Measure detection speed performance ##
We made a python script to measure merely detection speed without the speed limitations of VideoCapture. This is because we couldn't find a correlation between image detection speed and video detection speed. We achieve this by using multithreading.
```
python3 python/video_detect.py <video_source>
```

## Detection server and client ##
Run detection server:
```
python3 python/detection_server.py <port>
```

Run detection client:
```
python3 python/detection_client.py <server-ip> <port>
```

## Streaming server and client ##
Run streaming server:
```
python3 python/streaming_server.py <port>
```

Run streaming client:
```
python3 python/streaming_client.py <server-ip> <port>
```