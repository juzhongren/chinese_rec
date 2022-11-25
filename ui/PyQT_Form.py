# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'PyQT_Form.ui'
#
# Created by: PyQt5 UI code generator 5.12
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import os
import shutil

class QLabel(QtWidgets.QLabel):
    clicked = pyqtSignal()

    def __init__(self, parent=None):
        super(QLabel, self).__init__(parent)
        self.if_mouse_press = False
        self.pos_x = 0
        self.pos_y = 0

    # def mouseMoveEvent(self, e):
    #     print('mouse move:(%d,%d)\n' % (e.pos().x(), e.pos().y()))
    #     # if self.if_mouse_press:
    #         # dialog.move_point(e.pos().x(), e.pos().y())

    # def mousePressEvent(self, e):
    #     print('mousePressEvent(%d,%d)\n' % (e.pos().x(), e.pos().y()))
    #     self.if_mouse_press = True
    #     self.pos_x = e.pos().x()
    #     self.pos_y = e.pos().y()
    #     # dialog.move_point(e.pos().x(), e.pos().y())
    #
    # def mouseReleaseEvent(self, e):
    #     print('mouseReleaseEvent(%d,%d)\n' % (e.pos().x(), e.pos().y()))
    #     self.if_mouse_press = False

    def mouseDoubleClickEvent(self,e):
        print('mousePressEvent(%d,%d)\n' % (e.pos().x(), e.pos().y()))


        self.pos_x = e.pos().x()
        self.pos_y = e.pos().y()
        self.clicked.emit()
        chinese_dir = 'chinese_word'
        if os.path.exists(chinese_dir):
            shutil.rmtree(chinese_dir)
        print('mouse double clicked')


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1006, 603)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(670, 30, 89, 25))
        self.pushButton.setObjectName("pushButton")
        self.label = QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 30, 551, 521))
        self.label.setAutoFillBackground(False)
        self.label.setStyleSheet("background:white;\n"
"color:rgb(255, 255, 255);\n"
"font-size:100px;\n"
"ont-weight:bold;\n"
"font-family:宋体;\n"
"")
        self.label.setObjectName("label")

        #鼠标事件
        # self.label = MyLabel(self)  # 重定义的label


        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(780, 30, 89, 25))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(880, 30, 89, 25))
        self.pushButton_3.setObjectName("pushButton_3")
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(670, 60, 301, 491))
        self.textBrowser.setObjectName("textBrowser")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1006, 28))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "打开文件"))
        self.label.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:28pt; font-weight:600; color:#ef2929;\">显示图片</span></p></body></html>"))
        self.pushButton_2.setText(_translate("MainWindow", "切分字符"))
        self.pushButton_3.setText(_translate("MainWindow", "汉字预测"))
        self.textBrowser.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">显示预测的汉字...</p></body></html>"))


