import numpy as np
import os
import argparse
import torch
import re
from pathlib import Path
import cv2
import sys     
import time
from threading import Thread
from camera_discovery import CameraDiscovery
import multiprocessing
from Processer import Processor
"""name_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']"""
name_list = ['bicycle', 'motorcycle','childcar','person','doorgap']
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
person_num =[]
#from Visualizer import Visualizer
def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)
def cli():
    desc = 'Run TensorRT fall visualizer'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-model', help='trt engine file located in ./models', required=False)
    parser.add_argument('-image', help='image file path', required=False)
    parser.add_argument('--source', type=str, default=r'streams.txt', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--view-img', action='store_true', help='display results')
    args = parser.parse_args()
    model = args.model or 'best10s512.trt'
    img = args.image or 'no.jpg'
    return { 'model': model, 'image': img }
def clean_str(s):
    # Cleans a string by replacing special characters with underscore _
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)

def isdocker():
    # Is environment a Docker container
    return Path('/workspace').exists()  # or Path('/.dockerenv').exists()

def check_imshow():
    # Check if environment supports image displays
    try:
        assert not isdocker(), 'cv2.imshow() is disabled in Docker environments'
        cv2.imshow('test', np.zeros((1, 1, 3)))
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        return True
    except Exception as e:
        print(f'WARNING: Environment does not support cv2.imshow() or PIL Image.show() image displays\n{e}')
        return False

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    BLACK=[0,0,0]
    h, w = img.shape[0], img.shape[1]
    if h < w:
        img = cv2.copyMakeBorder(img, (w - h) // 2, (w - h) // 2, 0, 0, cv2.BORDER_CONSTANT, value=BLACK)
    if h > w:
        img = cv2.copyMakeBorder(img, 0, 0, (h - w) // 2, (h - w) // 2, cv2.BORDER_CONSTANT, value=BLACK)
    img = cv2.resize(img, (640, 640))
    return img, ratio, (dw, dh)
class LoadStreams:  # multiple IP or RTSP cameras
    def __init__(self, sources='streams.txt', img_size=640, stride=32):
        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride

        if os.path.isfile(sources):
            with open(sources, 'r') as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
        else:
            sources = [sources]

        n = len(sources)
        self.imgs = [None] * n
        self.sources = [clean_str(x) for x in sources]  # clean source names for later

        for i, s in enumerate(sources):
            # Start the thread to read frames from the video stream

            print(f'{i + 1}/{n}: {s}... ', end='')
            cap = cv2.VideoCapture(eval(s) if s.isnumeric() else s)
            assert cap.isOpened(), f'Failed to open {s}'
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) % 100
            _, self.imgs[i] = cap.read()  # guarantee first frame
            thread = Thread(target=self.update, args=([i, cap]), daemon=True)
            print(f' success ({w}x{h} at {fps:.2f} FPS).')
            thread.start()
        print('')  # newline

        # check for common shapes
        # s = np.hstack([letterbox(x, self.img_size, stride=self.stride)[0].shape for x in self.imgs], 0)  # shapes
        s = np.hstack([letterbox(x, self.img_size, stride=self.stride)[0].shape for x in self.imgs])  # shapes
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

    def update(self, index, cap):
        # Read next stream frame in a daemon thread
        n = 0
        while cap.isOpened():
            n += 1
            # _, self.imgs[index] = cap.read()
            cap.grab()
            if n == 4:  # read every 4th frame
                success, im = cap.retrieve()
                self.imgs[index] = im if success else self.imgs[index] * 0
                n = 0
            time.sleep(0.01)  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        img0 = self.imgs.copy()
        if cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        img = [letterbox(x, self.img_size, auto=self.rect, stride=self.stride)[0] for x in img0]

        # Stack
        # img = np.stack(img, 0)
        img = np.hstack(img)
        # Convert
        # img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416

        img = np.ascontiguousarray(img)

        return self.sources, img, img0, None

    def __len__(self):
        return 0  # 1E12 frames = 32 streams at 30 FPS for 30 years
def main(camera_ip):
    BLACK = [0, 0, 0]
    #camera_ip = CameraDiscovery.ws_discovery()
    user, pwd, ip, channel = "admin", "sy123456", camera_ip[0], 1
    cap_path = "rtsp://%s:%s@%s//Streaming/Channels/%d" % (user, pwd, ip, channel)  # HIKIVISION new version 2017
    # parse arguments
    args = cli()
    # setup processor and visualizer
    processor = Processor(model=args['model'])
    cap = cv2.VideoCapture(cap_path)
    #cap = cv2.VideoCapture("test1.mp4")
    count = 0
    
    while cap.isOpened():
        t_start = time.time()
        #print(t_start)
        ret, img = cap.read()
        A=time.time()-t_start
        #print('-----',A)
        #while not(ret):
          #cap = cv2.VideoCapture(cap_path)
          #ret, img = cap.read()
          
        #img = img[:, ::-1, :]
        # inference
        h,w = img.shape[0],img.shape[1]
        if h<w:
          img = cv2.copyMakeBorder(img, (w-h)//2, (w-h)//2, 0, 0, cv2.BORDER_CONSTANT, value=BLACK)
        if h>w:
          img = cv2.copyMakeBorder(img,0 , 0, (h-w)//2, (h-w)//2, cv2.BORDER_CONSTANT, value=BLACK)
        img = cv2.resize(img, (640, 640))
        B=time.time()-(A+t_start)
        #print('++++++++',B)
        if count % 3==0:
          output = processor.detect(img)
        boxes, confs, classes = processor.cls_process(output)
        fps = 1 / (time.time() - t_start)
        C = time.time()-(A+B+t_start)
        #print('=====',C)
        #print('fps:', fps)
        #print(boxes, confs, classes)
        if len(boxes) != 0:
          for (f ,conf, cls) in zip(boxes, confs, classes):
            x1 = int(f[0])
            y1 = int(f[1])
            x2 = int(f[2])
            y2 = int(f[3])
            #cls = int(f[-1])
            if int(cls)==1:
              filename = 'images/'+str(time.strftime('%Y_%m_%d %H_%M_%S', time.localtime(time.time()))) + '.jpg'
              cv2.imwrite(filename,img)
            if int(cls)==3:
              person_num.append(int(cls))      
            color_lb = compute_color_for_labels(int(cls))
            name = name_list[int(cls)]    
            cv2.rectangle(img, (x1, y1), (x2, y2), color_lb, thickness = 2)
            t_size = cv2.getTextSize(name, 0, fontScale=1, thickness=2)[0]
            cv2.rectangle(img, (x1-1, y1), (x1 + t_size[0] + 34, y1 - t_size[1]), color_lb, -1, cv2.LINE_AA)
            cv2.putText(img, name + str('%.2f'%conf), (x1, int(y1-2)), 0, 0.8, thickness=1,color = (255, 255, 255), lineType = cv2.LINE_AA)
            cv2.putText(img, 'FPS: ' + str('%.2f'%fps), (20, 30), 0, 1, thickness = 1, color = (255, 0, 0), lineType = cv2.LINE_AA)
        cv2.putText(img, 'person:'+str(len(person_num)), (20, 70), 0, 1, thickness = 1, color = (0, 0, 255), lineType = cv2.LINE_AA)
        count +=1
        person_num.clear()
        cv2.imshow("da", img)
        #print('============',time.time()-t_start)
        cv2.waitKey(1)
    
    
    '''img = cv2.imread("0.jpg")
    h,w = img.shape[0],img.shape[1]
    #img = img[:, ::-1, :]
    # inference
    if h<w:
      img = cv2.copyMakeBorder(img, (w-h)//2, (w-h)//2, 0, 0, cv2.BORDER_CONSTANT, value=BLACK)
    if h>w:
      img = cv2.copyMakeBorder(img,0 , 0, (w-h)//2, (w-h)//2, cv2.BORDER_CONSTANT, value=BLACK)
    img = cv2.resize(img,(640,640))
    t_start = time.time()
    output = processor.detect(img)
    boxes, confs, classes = processor.cls_process(output)
    fps = 1 / (time.time() - t_start)
    #print('fps:', fps)
    #print(boxes, confs, classes)
    if len(boxes) != 0:
      for (f ,conf, cls) in zip(boxes, confs, classes):
        x1 = int(f[0])
        y1 = int(f[1])
        x2 = int(f[2])
        y2 = int(f[3])
        #cls = int(f[-1])
        if int(cls)==1:
          filename = 'images/'+str(time.strftime('%Y_%m_%d %H_%M_%S', time.localtime(time.time()))) + '.jpg'
          cv2.imwrite(filename,img)
        if int(cls)==3:
          person_num.append(int(cls))
        color_lb = compute_color_for_labels(int(cls))
        name = name_list[int(cls)]    
        cv2.rectangle(img, (x1, y1), (x2, y2), color_lb, thickness = 2)
        t_size = cv2.getTextSize(name, 0, fontScale=1, thickness=2)[0]
        cv2.rectangle(img, (x1-1, y1), (x1 + t_size[0] + 34, y1 - t_size[1]), color_lb, -1, cv2.LINE_AA)
        cv2.putText(img, name + str('%.2f'%conf), (x1, int(y1-2)), 0, 0.8, thickness=1,color = (255, 255, 255), lineType = cv2.LINE_AA)
        cv2.putText(img, 'FPS: ' + str('%.2f'%fps), (20, 30), 0, 1, thickness = 1, color = (255, 0, 0), lineType = cv2.LINE_AA)
    cv2.putText(img, 'person:'+str(len(person_num)), (20, 70), 0, 1, thickness = 1, color = (255, 0, 0), lineType = cv2.LINE_AA)
    #cv2.imshow("da", img)
    cv2.imwrite("test.jpg",img)'''
def main1(save_img=False):
    BLACK = [0, 0, 0]
    
    model = args.model or 'best10s512.trt'
    img = args.image or 'no.jpg'
    source, view_img, imgsz,stride = args.source, args.view_img, args.img_size,32
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))
    processor = Processor(model=model)
    if webcam:
        #view_img = check_imshow()
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    count = 0
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img)#.to(device)
        img = img.half()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        # if img.ndimension() == 3:
        #     img = img.unsqueeze(0)
        img *=255.0
        #print(img.shape)
        img = (img.numpy()).astype(np.uint8) # hstack or vstack
        h,w = img.shape[0],img.shape[1]
        '''if h<w:
          img = cv2.copyMakeBorder(img, (w-h)//2, (w-h)//2, 0, 0, cv2.BORDER_CONSTANT, value=BLACK)
        if h>w:
          img = cv2.copyMakeBorder(img,0 , 0, (h-w)//2, (h-w)//2, cv2.BORDER_CONSTANT, value=BLACK)
        img = cv2.resize(img, (640, 640))'''
        output = processor.detect(img)
        boxes, confs, classes = processor.cls_process(output)
        #print('=====',C)
        #print('fps:', fps)
        #print(boxes, confs, classes)
        if len(boxes) != 0:
          for (f ,conf, cls) in zip(boxes, confs, classes):
            x1 = int(f[0])
            y1 = int(f[1])
            x2 = int(f[2])
            y2 = int(f[3])
            #cls = int(f[-1])
            if int(cls)==1:
              filename = 'images/'+str(time.strftime('%Y_%m_%d %H_%M_%S', time.localtime(time.time()))) + '.jpg'
              cv2.imwrite(filename,img)
            if int(cls)==3:
              person_num.append(int(cls))      
            color_lb = compute_color_for_labels(int(cls))
            name = name_list[int(cls)]    
            cv2.rectangle(img, (x1, y1), (x2, y2), color_lb, thickness = 2)
            t_size = cv2.getTextSize(name, 0, fontScale=1, thickness=2)[0]
            cv2.rectangle(img, (x1-1, y1), (x1 + t_size[0] + 34, y1 - t_size[1]), color_lb, -1, cv2.LINE_AA)
            cv2.putText(img, name + str('%.2f'%conf), (x1, int(y1-2)), 0, 0.8, thickness=1,color = (255, 255, 255), lineType = cv2.LINE_AA)
        cv2.putText(img, 'person:'+str(len(person_num)), (20, 70), 0, 1, thickness = 1, color = (0, 0, 255), lineType = cv2.LINE_AA)
        count +=1
        person_num.clear()
        cv2.imshow("da", img)
        #print('============',time.time()-t_start)
        cv2.waitKey(1)

class MyThread(Thread):
    def __init__(self, ip):
        super(MyThread, self).__init__()  # 重构run函数必须要写
        self.ip = ip

    def run(self):
        BLACK = [0, 0, 0]
        
        user, pwd, ip, channel = "admin", "sy123456", self.ip, 1
        cap_path = "rtsp://%s:%s@%s//Streaming/Channels/%d" % (user, pwd, ip, channel)  # HIKIVISION new version 2017
        # parse arguments
        args = cli()
        # setup processor and visualizer
        processor = Processor(model=args['model'])
        cap = cv2.VideoCapture(cap_path)
        #cap = cv2.VideoCapture("test1.mp4")
        count = 0
        windowname=self.ip
        while cap.isOpened():
            t_start = time.time()
            #print(t_start)
            ret, img = cap.read()
            A=time.time()-t_start
            #print('-----',A)
            #while not(ret):
              #cap = cv2.VideoCapture(cap_path)
              #ret, img = cap.read()
          
            #img = img[:, ::-1, :]
            # inference
            h,w = img.shape[0],img.shape[1]
            if h<w:
              img = cv2.copyMakeBorder(img, (w-h)//2, (w-h)//2, 0, 0, cv2.BORDER_CONSTANT, value=BLACK)
            if h>w:
              img = cv2.copyMakeBorder(img,0 , 0, (w-h)//2, (w-h)//2, cv2.BORDER_CONSTANT, value=BLACK)
            img = cv2.resize(img, (640, 640))
            B=time.time()-(A+t_start)
            #print('++++++++',B)
            if count % 3==0:
              output = processor.detect(img)
            boxes, confs, classes = processor.cls_process(output)
            fps = 1 / (time.time() - t_start)
            C = time.time()-(A+B+t_start)
            #print('=====',C)
            #print('fps:', fps)
            #print(boxes, confs, classes)
            if len(boxes) != 0:
              for (f ,conf, cls) in zip(boxes, confs, classes):
                x1 = int(f[0])
                y1 = int(f[1])
                x2 = int(f[2])
                y2 = int(f[3])
                #cls = int(f[-1])
                if int(cls)==1:
                   filename = 'images/'+str(time.strftime('%Y_%m_%d %H_%M_%S', time.localtime(time.time()))) + '.jpg'
                   cv2.imwrite(filename,img)
                if int(cls)==3:
                  person_num.append(int(cls))      
                color_lb = compute_color_for_labels(int(cls))
                name = name_list[int(cls)]    
                cv2.rectangle(img, (x1, y1), (x2, y2), color_lb, thickness = 2)
                t_size = cv2.getTextSize(name, 0, fontScale=1, thickness=2)[0]
                cv2.rectangle(img, (x1-1, y1), (x1 + t_size[0] + 34, y1 - t_size[1]), color_lb, -1, cv2.LINE_AA)
                cv2.putText(img, name + str('%.2f'%conf), (x1, int(y1-2)), 0, 0.8, thickness=1,color = (255, 255, 255), lineType = cv2.LINE_AA)
                cv2.putText(img, 'FPS: ' + str('%.2f'%fps), (20, 30), 0, 1, thickness = 1, color = (255, 0, 0), lineType = cv2.LINE_AA)
            cv2.putText(img, 'person:'+str(len(person_num)), (20, 70), 0, 1, thickness = 1, color = (0, 0, 255), lineType = cv2.LINE_AA)
            count +=1
            person_num.clear()
            cv2.imwrite(windowname+'{}.jpg'.format(count),img)
            #exit()
            cv2.imshow(windowname, img)
            #print('============',time.time()-t_start)
            cv2.waitKey(1)
          # visualizer.draw_results(img, boxes, confs, classes)

if __name__ == '__main__':
    desc = 'Run TensorRT fall visualizer'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--model', help='trt engine file located in ./models', required=False)
    parser.add_argument('--image', help='image file path', required=False)
    parser.add_argument('--source', type=str, default=r'streams.txt', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--view-img', action='store_true', help='display results')
    args = parser.parse_args()
    with torch.no_grad():
        main1()
       
    '''camera_ip = CameraDiscovery.ws_discovery()
    #muilte threading
    t1 = MyThread(camera_ip[0])
    t2 = MyThread(camera_ip[1])
    t1.start()
    t2.start()'''
    '''
    camera_ip = CameraDiscovery.ws_discovery()
    p1 = multiprocessing.Process(target = main, args =[camera_ip[0]] )
    p2 = multiprocessing.Process(target = main, args =[camera_ip[1]] )  
    p1.start()
    p2.start()'''
    
    
    
