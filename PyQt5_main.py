import os.path
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow,QDoubleSpinBox
from PyQt5.QtCore import QTimer,Qt
from PyQt5.QtGui import QImage, QPixmap, QPalette, QBrush
from PyQt5.QtWidgets import QMessageBox
from PyQt_1 import Ui_MainWindow
import numpy as np
import cv2
import time
from random import uniform
from PyQt5.Qt import *
# from PyQt_predict import Predict
# from PyQt_yolo import PyQt_YOLO
from yolo import YOLO
from PIL import Image
from PIL.ImageQt import ImageQt
from PyQt5.QtWidgets import QMessageBox
from datetime import datetime
from skimage import io,data
from PyQt5.QtCore import QDate,   QDateTime , QTime
display_path = ["img_lable/screenshort.png","img_lable/screenshort.png","img_lable/screenshort.png",
               "img_lable/screenshort.png","img_lable/screenshort.png","img_lable/screenshort.png"]
refresh_num = 0



class Monitor(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(Monitor,self).__init__()
        self.setupUi(self)
        self.cap = cv2.VideoCapture('C:/Users/刘震/Desktop/1.mp4')  # 初始化摄像头
        self.photo_flag = 0
        self.label.setScaledContents(True)  # 图片自适应
        self.black_img = Image.open('img_lable/sourse.png')
        self.black_img = ImageQt(self.black_img)#摄像头区域未打开的图片显示

        self.screenshot = Image.open(display_path[0])#截图区域的图片
        self.screenshot = self.screenshot.resize((70,90))
        self.screenshot = ImageQt(self.screenshot)
        self.refresh_num = 0
        self.timer = QTimer()
        self.startTimer()
        # self.yolo = PyQt_YOLO()
        self.setWindowTitle("Water object detection platform")
        self.yolo = YOLO()
        self.cap = cv2.VideoCapture(0)

        self.label_6.setStyleSheet("color:rgb(10,10,10,255);font-size:18px;font-weight:bold;font-family:Roman times;")
        position_path = 'position.txt'
        with open(position_path, 'w') as f:
            f.close()
        self.num_smoke = 0
        self.init()#这个必须写在最后

    def init(self):
        self.display_image = dict()
        self.screenshotcount = -1
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.show_image)
        self.pushButton.clicked.connect(self.open_camera)
        self.pushButton_2.clicked.connect(self.close_camera)
        self.label.setPixmap(QPixmap.fromImage(self.black_img))  # 往显示视频的Label里显示QImage
        # self.screenshots_area_display()
        self.pushButton_4.clicked.connect(self.flag_screenshot)
        self.pushButton_3.clicked.connect(self.flag_clear_folder)
        self.timer.timeout.connect(self.showTime)

    def showTime(self):
        time = QDateTime.currentDateTime()
        # dddd是星期几
        timeDispaly = time.toString('yyyy-MM-dd hh:mm:ss dddd')
        # 将标签设置成当前时间
        # self.label_6.setStyleSheet("color:rgb(20,20,20,255);font-size:20px;font-weight:bold:text")
        self.label_6.setText(timeDispaly)

    def startTimer(self):
        # 参数是时间间隔，1000毫秒
        self.timer.start(1000)

    def flag_clear_folder(self):
        self.textBrowser.setText("The folder images are cleared！")
        self.textBrowser.repaint()
        screenphone = QImage("./img_lable/creenshort.png")
        self.labels_0.setPixmap(QPixmap.fromImage(screenphone))  # 往显示视频的Label里显示QImage
        self.labels_1.setPixmap(QPixmap.fromImage(screenphone))  # 往显示视频的Label里显示QImage
        self.labels_2.setPixmap(QPixmap.fromImage(screenphone))  # 往显示视频的Label里显示QImage
        self.labels_3.setPixmap(QPixmap.fromImage(screenphone))  # 往显示视频的Label里显示QImage
        self.labels_4.setPixmap(QPixmap.fromImage(screenphone))  # 往显示视频的Label里显示QImage
        self.labels_5.setPixmap(QPixmap.fromImage(screenphone))  # 往显示视频的Label里显示QImage
        self.screenshotcount = -1



    def get_time(self):
        now = datetime.now()
        suffix = f'{now.year:04d}{now.month:02d}{now.day:02d}{now.hour:02d}{now.minute:02d}{now.second:02d}'
        return suffix

    #显示截图区域的函数
    def screenshots_area_display(self):
        image = self.display_image[self.screenshotcount]
        # print(self.display_image)
        image_h,image_w,_ = image.data.shape
        if self.screenshotcount == 0:
            showImage = QtGui.QImage(image.data, image_w, image_h, QImage.Format_RGB888)
            height = self.labels_0.height()
            width = self.labels_0.width()
            showImage.scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.labels_0.setPixmap(QPixmap.fromImage(showImage))  # 往显示视频的Label里显示QImage
            self.labels_0.setScaledContents(True)  # 图片自适应
        if self.screenshotcount == 1:
            showImage = QtGui.QImage(image.data, image_w, image_h, QImage.Format_RGB888)
            height = self.labels_1.height()
            width = self.labels_1.width()
            showImage.scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.labels_1.setPixmap(QPixmap.fromImage(showImage))  # 往显示视频的Label里显示QImage
            self.labels_1.setScaledContents(True)  # 图片自适应
        if self.screenshotcount == 2:
            showImage = QtGui.QImage(image.data, image_w, image_h, QImage.Format_RGB888)
            height = self.labels_2.height()
            width = self.labels_2.width()
            showImage.scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.labels_2.setPixmap(QPixmap.fromImage(showImage))  # 往显示视频的Label里显示QImage
            self.labels_2.setScaledContents(True)  # 图片自适应
        if self.screenshotcount == 3:
            showImage = QtGui.QImage(image.data, image_w, image_h, QImage.Format_RGB888)
            height = self.labels_3.height()
            width = self.labels_3.width()
            showImage.scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.labels_3.setPixmap(QPixmap.fromImage(showImage))  # 往显示视频的Label里显示QImage
            # print(YOLO.class_number)
            self.labels_3.setScaledContents(True)  # 图片自适应
        if self.screenshotcount == 4:
            showImage = QtGui.QImage(image.data,image_w, image_h, QImage.Format_RGB888)
            height = self.labels_4.height()
            width = self.labels_4.width()
            showImage.scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.labels_4.setPixmap(QPixmap.fromImage(showImage))  # 往显示视频的Label里显示QImage
            # print(YOLO.class_number)
            self.labels_4.setScaledContents(True)  # 图片自适应
        if self.screenshotcount == 5:
            showImage = QtGui.QImage(image.data, image_w, image_h, QImage.Format_RGB888)
            height = self.labels_5.height()
            width = self.labels_5.width()
            showImage.scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.labels_5.setPixmap(QPixmap.fromImage(showImage))  # 往显示视频的Label里显示QImage
            # print(YOLO.class_number)
            self.labels_5.setScaledContents(True)  # 图片自适应


    #打开摄像头
    def open_camera(self):
        self.textBrowser.setText("Checking the number of cameras...")
        self.textBrowser.repaint()
        self.textBrowser.append("Ready to start testing...")
        self.textBrowser.repaint()
        # self.cap = cv2.VideoCapture(0)
        self.camera_timer.start(40)  # 每40毫秒读取一次，即刷新率为25帧

    #显示图片
    def show_image(self):
        count = 0
        # print("count",count)
       # flag, self.image = self.cap.read()  # 从视频流中读取图片

        self.image = cv2.imread(r"E:/yolo3-pytorch-master/yolo3-pytorch-master/VOCdevkit/VOC2007/JPEGImages/02430.jpg")
        image_show = cv2.resize(self.image, (1280, 720))  # 把读到的帧的大小重新设置
        image_show = self.image


        width, height = image_show.shape[:2]  # 行:宽，列:高
        # width, height = 1280, 720


        image_show = cv2.cvtColor(image_show, cv2.COLOR_BGR2RGB)  # opencv读的通道是BGR,要转成RGB

        image_show = cv2.flip(image_show, 1)  # 水平翻转，因为摄像头拍的是镜像的。

        image_show = Image.fromarray(np.uint8(image_show))# 进行检测
       # print("flag")
        image_show = self.yolo.detect_image(image_show)
        image_show = np.array(image_show)

        position_path = 'position.txt'
        with open(position_path,'r') as f:
            position = f.readline()
            f.close()
        if len(position):
            print('position',position)
            if position.split("_")[-1] != self.num_smoke:
                self.num_smoke = position.split("_")[-1]
                self.auto_screenshot(image_show)
                self.textBrowser.append("检测到吸烟人数{}".format(position.split("_")[-1]))
                self.textBrowser.repaint()
            with open(position_path, 'w') as f:
                f.close()
        else:
            self.num_smoke = 0



        if self.flag_screenshot == True:
            self.display_image[self.screenshotcount] = image_show
            self.screenshots_area_display()
            self.textBrowser.append("One image was captured！")
            self.textBrowser.repaint()
            self.flag_screenshot = False

        # 把读取到的视频数据变成QImage形式(图片数据、高、宽、RGB颜色空间，三个通道各有2**8=256种颜色)
        self.showImage = QtGui.QImage(image_show.data, height, width, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(self.showImage))  # 往显示视频的Label里显示QImage
        # print(YOLO.class_number)
        self.label.setScaledContents(True)  # 图片自适应

    def auto_screenshot(self,screenshot):
        self.screenshotcount += 1
        if self.screenshotcount == 6:
            self.screenshotcount = 0
        self.display_image[self.screenshotcount] = screenshot
        self.screenshots_area_display()
        self.textBrowser.append("One image was captured！")
        self.textBrowser.repaint()
        self.flag_screenshot = False

    def flag_screenshot(self):
        self.screenshotcount +=1
        if self.screenshotcount == 6:
            self.screenshotcount = 0
        self.flag_screenshot = True


    #关闭摄像头
    def close_camera(self):
        self.textBrowser.append("Stopping detection...")
        self.textBrowser.repaint()
        if self.camera_timer.isActive():
            self.camera_timer.stop()  # 停止读取
        self.label.clear()  # 清除label组件上的图片


        self.label.setPixmap(QPixmap.fromImage(self.black_img))

if __name__ == '__main__':
    from PyQt5 import QtCore

    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)  # 自适应分辨率

    app = QtWidgets.QApplication(sys.argv)
    ui = Monitor()

    ui.show()
    sys.exit(app.exec_())

