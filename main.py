from PyQt5 import QtWidgets,QtGui
from ui.PyQT_Form import Ui_MainWindow
import cv2
import sys
import chinese_character_recognition
import time
import os
import tensorflow as tf
import shutil
import numpy as np
import product_character
from ui import character_dialog
from PyQt5.QtWidgets import *


class MyPyQT_Form(QtWidgets.QMainWindow,Ui_MainWindow):
    def __init__(self):
        super(MyPyQT_Form,self).__init__()
        self.setupUi(self)
        self.filename = ''
        self.weizhi =[]
        self.index = -1
        #打开文件
        self.pushButton.clicked.connect(lambda:self.openfile())
        #切分字符
        self.pushButton_2.clicked.connect(lambda:self.cut_image())
        #汉字预测
        self.pushButton_3.clicked.connect(lambda:self.predict())

        #
        # self.dialog
        self.result_rec = []
        self.label.clicked.connect(lambda:self.showDialog())

    def openfile(self):
        # imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", 'Image files(*.jpg *.gif *.png)')
        self.filename = imgName

        # print("mingzhi")
        # print(self.filename)
        # 载入图片
        img = cv2.imread(self.filename)
        img = cv2.resize(img, (self.label.height(), self.label.width()))
        cv2.imwrite(self.filename,img)

        image = QtGui.QImage(img, img.shape[1], img.shape[0], \
                             img.shape[1] * 3, QtGui.QImage.Format_RGB888)
        jpg = QtGui.QPixmap(image).scaled(self.label.width(), self.label.height())
        self.label.setPixmap(jpg)


        # print(self.filename)


    def cut_image(self):
        self.weizhi = []
        self.result = []
        minWidth = 5  # 最小宽度
        minHeight = 20  # 最小高度
        nrootdir = ("./tmp/")
        if not os.path.isdir(nrootdir):
            os.makedirs(nrootdir)
        else:
            shutil.rmtree(nrootdir)
        os.makedirs(nrootdir)
        # 载入图片
        img = cv2.imread(self.filename)

        # img = cv2.resize(img, (500, 500))
        img = cv2.resize(img,(self.label.height(),self.label.width()))
        image_temp = np.copy(img)
        height, width, bytesPerComponent = img.shape
        bytesPerLine = 3 * width
        # 高斯去噪
        blured = cv2.GaussianBlur(img, (5, 5), 0)
        # cv2.imshow("blured",blured)
        # 灰度化
        # 把输入图像灰度化
        gray = cv2.cvtColor(blured, cv2.COLOR_RGB2GRAY)
        # cv2.imshow("gray", gray)

        binary = cv2.adaptiveThreshold(gray, 255, \
                                       cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 5)
        # cv2.imshow("binary", binary)

        # 开操作
        kernel_1 = cv2.getStructuringElement(cv2.MORPH_RECT, (150, 10))
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 10))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_1)
        # cv2.imshow("opened", opened)
        # # 关操作
        # # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # # binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        # # cv2.imshow("closed", binary)
        not_img = cv2.bitwise_not(opened)
        # cv2.imshow("not", not_img)

        contours, hierarchy = cv2.findContours(not_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bianjie = []
        for i in range(0, len(contours)):
            if cv2.contourArea(contours[i]) < 100:
                continue
            x, y, w, h = cv2.boundingRect(contours[i])
            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
            bianjie.append([x, y, w, h])
        # print(bianjie)
        bianjie = sorted(bianjie, key=lambda y: y[1])
        # print("bianjie")
        # print(bianjie)
        # cv2.imshow("11", img)

        # 形态学操作， 圆形核腐蚀
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        back_mask = cv2.erode(binary, kernel, iterations=6)
        # 反色 变为数字的掩模
        num_mask = cv2.bitwise_not(back_mask)
        # 中值滤波
        num_mask = cv2.medianBlur(num_mask, 3)
        # cv2.imshow("num_mask", num_mask)

        # 寻找轮廓
        contours, hierarchy = cv2.findContours(num_mask, \
                                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # cv2.drawContours(img,contours,-1,(0,255,0),3)
        # cv2.imshow("cou",img)
        result_1 = []
        for i in range(0, len(contours)):
            # if cv2.contourArea(contours[i]) < 100:
            #     continue
            # print(contours)
            x, y, w, h = cv2.boundingRect(contours[i])
            if w < minWidth or h < minHeight:
                # 如果不满足条件就过滤掉
                continue
            # print(x, y, w, h)
            result_1.append([x, y, w, h])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)

        self.result = []
        for y in range(len(bianjie)):
            temp = []
            for i in range(len(result_1)):
                if (result_1[i][1] + result_1[i][3] / 2 < bianjie[y][1] + bianjie[y][3]) and \
                        (result_1[i][1] + result_1[i][3] / 2 > bianjie[y][1]):
                    temp.append(result_1[i])
            temp = sorted(temp, key=lambda x: x[0])
            self.result.append(temp)
        # print("result")
        # print(self.result)

        for i in range(len(self.result)):
            # print(result[i])
            self.weizhi.append(len(self.result[i]))
            for j in range(len(self.result[i])):
                x = self.result[i][j][0]
                y = self.result[i][j][1]
                w = self.result[i][j][2]
                h = self.result[i][j][3]
                # newimage = image_temp[y:y + h , x:x + w]
                newimage = image_temp[y+2:y + h-2, x+2:x + w-2]
                cv2.imwrite(nrootdir + str(i) + "_" + str(j) + ".png", newimage)
        image = QtGui.QImage(img, img.shape[1], img.shape[0], img.shape[1] * 3, QtGui.QImage.Format_RGB888)
        png1 = QtGui.QPixmap(image).scaled(self.label.width(), self.label.height())
        self.label.setPixmap(png1)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


    def predict(self):
        tf.reset_default_graph()
        self.textBrowser.setPlainText("")
        label_dict = chinese_character_recognition.get_label_dict()
        name_list = chinese_character_recognition.get_file_list('./tmp')
        final_predict_val, final_predict_index = chinese_character_recognition.inference(name_list)
        final_reco_text = []  # 存储最后识别出来的文字串
        # 给出top 3预测，candidate1是概率最高的预测
        for i in range(len(final_predict_val)):
            candidate1 = final_predict_index[i][0][0]
            final_reco_text.append(label_dict[int(candidate1)])
        # print('=====================OCR RESULT=======================\n')
        # 打印出所有识别出来的结果（取top 1）
        self.textBrowser.setPlainText("")
        # for i in range(len(final_reco_text)):
        num = 0
        for i in range(len(self.weizhi)):
            # print(final_reco_text[i])
            for j in range(self.weizhi[i]):
                print(final_reco_text[num])
                self.textBrowser.insertPlainText(final_reco_text[num])
                num += 1
            self.textBrowser.append("")  # 文本框逐条添加数据
            self.textBrowser.moveCursor(self.textBrowser.textCursor().End)  # 文本框显示到底部
            time.sleep(0.5)

        # 给出top 3预测，candidate1是概率最高的预测
        self.result_rec.clear()
        for i in range(len(final_predict_val)):
            temp = []
            candidate1 = final_predict_index[i][0][0]
            candidate2 = final_predict_index[i][0][1]
            candidate3 = final_predict_index[i][0][2]
            temp.append(label_dict[int(candidate1)])
            temp.append(label_dict[int(candidate2)])
            temp.append(label_dict[int(candidate3)])
            temp.append(final_predict_val[i])
            self.result_rec.append(temp)
        #     logger.info('[the result info] image: {0} predict: {1} {2} {3}; predict index {4} predict_val {5}'.format(
        #         name_list[i],
        #         label_dict[int(candidate1)], label_dict[int(candidate2)], label_dict[int(candidate3)],
        #         final_predict_index[i], final_predict_val[i]))
        # print('=====================OCR RESULT=======================\n')
        # # 打印出所有识别出来的结果（取top 1）
        for i in range(len(self.result_rec)):
            print(self.result_rec[i], )


    def showDialog(self):
        chinese_dir = 'chinese_word'
        if os.path.exists(chinese_dir):
            shutil.rmtree(chinese_dir)
        os.mkdir(chinese_dir)
        self.index = -1
        flag = False
        # print(self.result)
        # print("zuobiao")
        print(self.label.pos_x,self.label.pos_y)
        if self.label.pos_x != 0 and self.label.pos_y != 0:
            for i in range(len(self.result)):
                # print(self.result[i])
                for j in range(len(self.result[i])):
                    self.index += 1
                    if self.label.pos_x > self.result[i][j][0] -10 and \
                            self.label.pos_x < (self.result[i][j][0]+self.result[i][j][2]+10) and\
                            self.label.pos_y > self.result[i][j][1]-30 and \
                            self.label.pos_y < (self.result[i][j][1]+self.result[i][j][3]+10):
                        flag = True

                    if flag:
                        break
                if flag:
                    break
        print(self.result_rec[self.index])
        self.dialog = character_dialog.dialog_Form()
        filenames=[]
        for i in range(3):
            filenames.append(product_character.product(self.result_rec[self.index][i],500))

        for j in range(3):
            # print(type(self.result_rec[self.index][3]))
            # print(type(self.result_rec[self.index][3][0][j]))
            bb = "%.2f%%" % (self.result_rec[self.index][3][0][j] * 100)
            filenames.append(product_character.product(str(bb),100))
        # print(filenames)
        self.dialog.setfilename(filenames)
        self.dialog.openfile_1()
        self.dialog.openfile_2()
        self.dialog.openfile_3()
        self.dialog.openfile_4()
        self.dialog.openfile_5()
        self.dialog.openfile_6()
        self.dialog.show()
        # # self.dialog.exec()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    my_pyqt_form = MyPyQT_Form()
    my_pyqt_form.show()
    sys.exit(app.exec_())