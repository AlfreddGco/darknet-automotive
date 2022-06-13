#!/usr/bin/python3
import os, random

images = os.listdir('OIS/images')
train_file = open('OIS/train.txt', 'w')
valid_file = open('OIS/valid.txt', 'w')

for image in images:
  if(random.random() <= 0.2):
    valid_file.write('OIS/images/%s\n' % image)
  else:
    train_file.write('OIS/images/%s\n' % image)

train_file.close()
valid_file.close()
