import threading
import time
from onvif import ONVIFCamera
import zeep
import requests
from requests.auth import HTTPDigestAuth
import cv2
import numpy as np

a =True
def zeep_pythonvalue(self, xmlvalue):
    return xmlvalue
def snap(cam_ip,usr='sy',pwd='SY123456'):
    # Get target profile
    while a:
        zeep.xsd.simple.AnySimpleType.pythonvalue = zeep_pythonvalue
        mycam = ONVIFCamera(cam_ip, 80, usr, pwd)
        media = mycam.create_media_service()  # 创建媒体服务
        media_profile = media.GetProfiles()[0]  # 获取配置信息
        res = media.GetSnapshotUri({'ProfileToken': media_profile.token})
        response = requests.get(res.Uri, auth=HTTPDigestAuth("sy", "SY123456"))
        res = "{_time}.png".format(_time=time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time())))
        # cv2.imshow('==',response.content)
        nparr = np.frombuffer(response.content, dtype=np.uint8)
        segment_data = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        cv2.imshow('',segment_data)
        cv2.waitKey(1)
        print( time.localtime(time.time()))

class MyThread(threading.Thread):
    def __init__(self, n):
        super(MyThread, self).__init__()  # 重构run函数必须要写
        self.n = n

    def run(self):
        '''print("task", self.n)
        time.sleep(1)
        print('2s')
        time.sleep(1)
        print('1s')
        time.sleep(1)
        print('0s')
        time.sleep(1)
        print(time.time())'''
        
        '''print('start:',self.n)
        while a:
          zeep.xsd.simple.AnySimpleType.pythonvalue = zeep_pythonvalue
          mycam = ONVIFCamera(self.n, 80, 'sy', 'SY123456')
          media = mycam.create_media_service()  # 创建媒体服务
          media_profile = media.GetProfiles()[0]  # 获取配置信息
          res = media.GetSnapshotUri({'ProfileToken': media_profile.token})
          response = requests.get(res.Uri, auth=HTTPDigestAuth("sy", "SY123456"))
          res = "{_time}.png".format(_time=time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time())))
          # cv2.imshow('==',response.content)
          nparr = np.frombuffer(response.content, dtype=np.uint8)
          segment_data = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
          cv2.imshow(self.n,segment_data)
          cv2.waitKey(1)
          print( time.localtime(time.time()))'''
        ip = self.n
        print('start:', self.n)
        user ,pwd,channel= 'admin','sy123456',1
        cap_path = "rtsp://%s:%s@%s//Streaming/Channels/%d" % (user, pwd, ip, channel)  # HIKIVISION new version 2017
        cap = cv2.VideoCapture(cap_path)
        while True:
            ret,frame = cap.read()
            #cv2.imshow(self.n,frame)
            cv2.imwrite(self.n+'.jpg',frame)
            exit()
            cv2.waitKey(1) 
if __name__ == "__main__":
    # a = [1,2,3,4,5]
    # print(len(a))
    # for i in range(len(a)):
    #     if i ==0
    #         t1 = MyThread(a[i])
    #     if i ==2:
    #         t2 = MyThread(a[i])
    #     if i ==1:    
    #         t3 = MyThread(a[i])
    #     if i ==3:
    #         t4 = MyThread(a[i])
    #     if i ==4:   
    #         t5 = MyThread(a[i])
    from camera_discovery import CameraDiscovery
    camera_ip = CameraDiscovery.ws_discovery()
    print(camera_ip)
    t1 = MyThread(camera_ip[0])
    t2 = MyThread(camera_ip[1])
    t1.start()
    t2.start()
   
