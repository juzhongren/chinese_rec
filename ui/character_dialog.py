from PyQt5 import QtWidgets,QtGui
from ui.dialog import Ui_Form
import cv2
import sys



class dialog_Form(QtWidgets.QMainWindow,Ui_Form):
    def __init__(self):
        super(dialog_Form, self).__init__()
        self.setupUi(self)
        self.filename_1 = ""
        self.filename_2 = ""
        self.filename_3 = ""
        self.filename_4 = ""
        self.filename_5 = ""
        self.filename_6 = ""


    def setfilename(self,filename):
        self.filename_1 = filename[0]
        self.filename_2 = filename[1]
        self.filename_3 = filename[2]
        self.filename_4 = filename[3]
        self.filename_5 = filename[4]
        self.filename_6 = filename[5]


    def openfile_1(self):
        # 载入图片
        img = cv2.imread(self.filename_1)
        img = cv2.resize(img, (self.label_1.height(), self.label_1.width()))
        image = QtGui.QImage(img, img.shape[1], img.shape[0], \
                             img.shape[1] * 3, QtGui.QImage.Format_RGB888)
        png = QtGui.QPixmap(image).scaled(self.label_1.width(), self.label_1.height())
        self.label_1.setPixmap(png)
        self.label_1.setScaledContents(True)

    def openfile_2(self):
        # 载入图片
        img = cv2.imread(self.filename_2)
        img = cv2.resize(img, (self.label_2.height(), self.label_2.width()))
        image = QtGui.QImage(img, img.shape[1], img.shape[0], \
                             img.shape[1] * 3, QtGui.QImage.Format_RGB888)
        png = QtGui.QPixmap(image).scaled(self.label_2.width(), self.label_2.height())
        self.label_2.setPixmap(png)
        self.label_2.setScaledContents(True)

    def openfile_3(self):
        # 载入图片
        img = cv2.imread(self.filename_3)
        img = cv2.resize(img, (self.label_3.height(), self.label_3.width()))
        image = QtGui.QImage(img, img.shape[1], img.shape[0], \
                             img.shape[1] * 3, QtGui.QImage.Format_RGB888)
        png = QtGui.QPixmap(image).scaled(self.label_3.width(), self.label_3.height())
        self.label_3.setPixmap(png)
        self.label_3.setScaledContents(True)

    def openfile_4(self):
        # 载入图片
        img = cv2.imread(self.filename_4)
        img = cv2.resize(img, (self.label_4.height(), self.label_4.width()))
        image = QtGui.QImage(img, img.shape[1], img.shape[0], \
                             img.shape[1] * 3, QtGui.QImage.Format_RGB888)
        png = QtGui.QPixmap(image).scaled(self.label_4.width(), self.label_4.height())
        self.label_4.setPixmap(png)
        self.label_4.setScaledContents(True)

    def openfile_5(self):
        # 载入图片
        img = cv2.imread(self.filename_5)
        img = cv2.resize(img, (self.label_5.height(), self.label_5.width()))
        image = QtGui.QImage(img, img.shape[1], img.shape[0], \
                             img.shape[1] * 3, QtGui.QImage.Format_RGB888)
        png = QtGui.QPixmap(image).scaled(self.label_5.width(), self.label_5.height())
        self.label_5.setPixmap(png)
        self.label_5.setScaledContents(True)

    def openfile_6(self):
        # 载入图片
        img = cv2.imread(self.filename_6)
        img = cv2.resize(img, (self.label_6.height(), self.label_6.width()))
        image = QtGui.QImage(img, img.shape[1], img.shape[0], \
                             img.shape[1] * 3, QtGui.QImage.Format_RGB888)
        png = QtGui.QPixmap(image).scaled(self.label_6.width(), self.label_6.height())
        self.label_6.setPixmap(png)
        self.label_6.setScaledContents(True)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    my_pyqt_form = dialog_Form()
    my_pyqt_form.show()
    sys.exit(app.exec_())