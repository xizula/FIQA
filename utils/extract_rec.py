import mxnet as mx
import matplotlib.pyplot as plt
import numpy as np
import os
import typer
import cv2


# Ścieżka do pliku .rec
# rec_file = '/mnt/c/Users/izam1/Documents/Studia/Mgr/Praca_magisterska/DANE/faces_webface_112x112/train.rec'


def extract_rec(rec_file, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    recordio = mx.recordio.MXRecordIO(rec_file, 'r')
    idx = 0

    while True:
        print("Started")
        item = recordio.read()
        if item is None:
            break
        
        header, img = mx.recordio.unpack(item)
        label = int(header.label)
        img = mx.image.imdecode(img).asnumpy()
        
        if img is None or img.size == 0:
            # print(f"Warning: image at index {idx} could not be decoded.")
            continue
        
        label_dir = os.path.join(output_dir, str(label))
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        
        img_filename = os.path.join(label_dir, f'{idx}.jpg')
        if not cv2.imwrite(img_filename, img):
            print(f"Error: failed to write image {img_filename}")

        
        idx += 1
# output_dir = '/mnt/c/Users/izam1/Documents/Studia/Mgr/Praca_magisterska/DANE/CasiaWebFace'


if __name__ == '__main__':
    # typer.run(extract_rec)
    extract_rec('mgr_data/faces_emore/train.rec', 'mgr_data/MS1M')

