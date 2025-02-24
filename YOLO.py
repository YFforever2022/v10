"""
This file is used to build WEB requests for image reasoning and return results
"""
import os
import sys
import threading
import base64
import time

from ultralytics import YOLOv10
import ultralytics.utils.torch_utils as torch_utils
import cv2
import numpy as np
from gevent.pywsgi import WSGIServer
from threading import Thread
from flask import Flask, request
import mmap

import ctypes
import ctypes.wintypes
import win32api
import win32con
import win32gui
import win32ui

import warnings
warnings.filterwarnings("ignore", message='.*__floordiv__ is deprecated.*')
import multiprocessing
multiprocessing.freeze_support()




lock = threading.Lock()

pid = os.getpid()

jcqbh = '1'
m_imgsz = 640
dk = 7700
m_device = ''
weights = './1/yolov10n.pt'


geshu = len(sys.argv)
if geshu < 6:
    print('参数不足，启动失败，2秒后自动退出')
    time.sleep(2)
    sys.exit()
else:
     jcqbh = sys.argv[1]
     m_imgsz = int(sys.argv[2])
     dk = int(sys.argv[3])
     temp_deivce = sys.argv[4]
     if temp_deivce == 'aidmlm':
         m_device = ''
     else:
         m_device = temp_deivce
     weights = sys.argv[5]


tt1 = time.time()
model = YOLOv10(weights)
torch_utils.select_device(m_device)
tt2 = time.time()
print('加载模型耗时' + str(int((tt2-tt1)*1000)) + 'ms')




def Window_Shot(hwnd):
    if win32gui.IsWindow(hwnd) == False:
        hwnd = win32gui.GetDesktopWindow()
        MoniterDev = win32api.EnumDisplayMonitors(None, None)
        w = MoniterDev[0][2][2]
        h = MoniterDev[0][2][3]
    else:
        ret = win32gui.GetClientRect(hwnd)
        w = ret[2]
        h = ret[3]

    hwndDC = win32gui.GetWindowDC(hwnd)
    mfcDC = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()
    saveBitMap = win32ui.CreateBitmap()
    saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)
    saveDC.SelectObject(saveBitMap)
    saveDC.BitBlt((0, 0), (w, h), mfcDC, (0, 0), win32con.SRCCOPY)

    signedIntsArray = saveBitMap.GetBitmapBits(True)
    im_opencv = np.frombuffer(signedIntsArray, dtype='uint8')
    im_opencv.shape = (h, w, 4)

    win32gui.DeleteObject(saveBitMap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwndDC)

    return im_opencv





def yuce(tp, client_ip=""):
    lock.acquire()

    if client_ip != "":
        m_text = "---------------------------------------------------\n" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + " 请求方IP:" + client_ip
    else:
        m_text = "---------------------------------------------------\n" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    results = model.predict(source=tp,
                            imgsz=m_imgsz,
                            conf=0.2,
                            save=False,
                            save_txt=False,
                            show_labels=False,
                            show_conf=False,
                            show_boxes=False,
                            line_width=None,
                            verbose=False
                            )

    ret = ""
    for r in results:
        int_cls = r.boxes.cls.int().tolist()
        float_conf = r.boxes.conf.float().tolist()
        int_xyxy = r.boxes.xyxy.int().tolist()
        for index, box in enumerate(int_xyxy):
            x = box[0]
            y = box[1]
            w = abs(box[2] - box[0])
            h = abs(box[3] - box[1])
            cls = int_cls[index]
            try:
                name = model.names[cls]
            except:
                try:
                    name = results[0].names[cls]
                except:
                    name = 'null'

            conf = "{:.2f}".format(float_conf[index])
            ret = ret + str(cls) + ',' + str(x) + ',' + str(y) + ',' + str(w) + ',' + str(h) + ',' + str(conf) + ',' + str(name) + "|"
    m_text = m_text + '\n' + ret
    print(m_text)
    xywh_list = ret.encode('utf-8')
    lock.release()
    return xywh_list


def bytesToMat(img):
    np_arr = np.frombuffer(img, dtype=np.uint8)
    img_mat = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
    if img_mat.shape[-1] == 4:
        img_mat = img_mat[:, :, :3]
    return img_mat



app = Flask(__name__)

@app.route('/pid', methods=['GET', 'POST'])
def YOLOv10_WEB_pid():
    return str(pid)

@app.route('/pic', methods=['GET', 'POST'])
def YOLOv10_WEB_pic():
    if request.data == b'':
        return ''
    try:
        tp = request.get_data()
        img = bytesToMat(tp)
        jieguo = yuce(img, client_ip=request.remote_addr)
        return jieguo
    except:
        return ''

@app.route('/hwnd',methods=['GET', 'POST'])
def YOLOv10_WEB_hwnd():
    try:
        tp = request.get_data()
        sj = tp.decode('utf-8', "ignore")
        jb = int(float(sj))
        img = Window_Shot(jb)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        jieguo = yuce(img, client_ip=request.remote_addr)
        return jieguo
    except:
        return ''

@app.route('/base64', methods=['GET', 'POST'])
def YOLOv10_WEB_base64():
    try:
        tp = request.get_data()
        tp_wb = tp.decode('utf-8', "ignore")
        b_tp = base64.b64decode(tp_wb)
        img = bytesToMat(b_tp)
        jieguo = yuce(img, client_ip=request.remote_addr)
        return jieguo
    except:
        return ''

@app.route('/file',methods=['GET', 'POST'])
def YOLOv10_WEB_file():
    try:
        tp = request.get_data()
        tp_file = tp.decode('utf-8', "ignore")

        np_arr = np.fromfile(file=tp_file, dtype=np.uint8)
        img_mat = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
        if img_mat.shape[-1] == 4:
            img_mat = img_mat[:, :, :3]

        jieguo = yuce(img_mat, client_ip=request.remote_addr)
        return jieguo
    except:
        return ''


def qidong():
    print('WEB服务器已开启，端口号：' + str(dk))
    print('')

    WSGIServer(('0.0.0.0', dk), app, log=None).serve_forever()


SendMessage = ctypes.windll.user32.SendMessageA

class COPYDATASTRUCT(ctypes.Structure):
    _fields_ = [
        ('dwData', ctypes.wintypes.LPARAM),
        ('cbData', ctypes.wintypes.DWORD),
        ('lpData', ctypes.c_void_p)
    ]

PCOPYDATASTRUCT = ctypes.POINTER(COPYDATASTRUCT)


class Listener:
    def __init__(self):
        WindowName = "aidmlm.com" + jcqbh
        message_map = {
            win32con.WM_COPYDATA: self.OnCopyData
        }
        wc = win32gui.WNDCLASS()
        wc.lpfnWndProc = message_map
        wc.lpszClassName = WindowName
        hinst = wc.hInstance = win32api.GetModuleHandle(None)
        classAtom = win32gui.RegisterClass(wc)
        self.hwnd = win32gui.CreateWindow(
            classAtom,
            "aidmlm.com" + jcqbh,
            0,
            0,
            0,
            win32con.CW_USEDEFAULT,
            win32con.CW_USEDEFAULT,
            0,
            0,
            hinst,
            None
        )
        print('pid', pid)
        print("hwnd", self.hwnd)
        Thread(target=qidong).start()


    def OnCopyData(self, hwnd, msg, wparam, lparam):
        pCDS = ctypes.cast(lparam, PCOPYDATASTRUCT)
        s = ctypes.string_at(pCDS.contents.lpData, pCDS.contents.cbData).decode()
        cd = int(float(s))

        if wparam == 1:

            file_name = 'aidmlm.com' + jcqbh
            shmem = mmap.mmap(0, cd, file_name, mmap.ACCESS_WRITE)
            tp = shmem.read(cd)

            img = bytesToMat(tp)
            jieguo = yuce(img)


            jgcd = len(jieguo)
            shmem.seek(0)
            shmem.write(jieguo)

            shmem.close()  # 关闭映射

            return jgcd
        if wparam == 0:
            return pid
        if wparam == 2:
            jb = cd
            if win32gui.IsWindow(jb) == False:
                return 11
            img = Window_Shot(jb)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            jieguo = yuce(img)

            file_name = 'aidmlm.com' + jcqbh
            shmem = mmap.mmap(0, 51200, file_name, mmap.ACCESS_WRITE)

            jgcd = len(jieguo)
            shmem.seek(0)
            shmem.write(jieguo)
            shmem.close()

            return jgcd
        return 10


l = Listener()
win32gui.PumpMessages()




