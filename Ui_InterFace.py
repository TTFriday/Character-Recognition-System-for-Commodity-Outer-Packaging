# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'd:\外包项目\2.包装字符识别\OCR_Detect0510\InterFace.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_BottleDetect(object):
    def setupUi(self, BottleDetect):
        BottleDetect.setObjectName("BottleDetect")
        BottleDetect.resize(840, 591)
        BottleDetect.setStyleSheet("Qmebar{\n"
"    \n"
"    background-color: rgb(170, 170, 170);\n"
"}")
        self.centralwidget = QtWidgets.QWidget(BottleDetect)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gps_img = QtWidgets.QGraphicsView(self.centralwidget)
        self.gps_img.setStyleSheet("background-color: rgb(197, 197, 197);")
        self.gps_img.setObjectName("gps_img")
        self.gridLayout_2.addWidget(self.gps_img, 0, 0, 1, 1)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setObjectName("label_3")
        self.verticalLayout_2.addWidget(self.label_3)
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setObjectName("label_5")
        self.verticalLayout_2.addWidget(self.label_5)
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setObjectName("label_4")
        self.verticalLayout_2.addWidget(self.label_4)
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setObjectName("label_6")
        self.verticalLayout_2.addWidget(self.label_6)
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setObjectName("label_7")
        self.verticalLayout_2.addWidget(self.label_7)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.led_num = QtWidgets.QLineEdit(self.centralwidget)
        self.led_num.setObjectName("led_num")
        self.verticalLayout.addWidget(self.led_num)
        self.led_ProDay = QtWidgets.QLineEdit(self.centralwidget)
        self.led_ProDay.setObjectName("led_ProDay")
        self.verticalLayout.addWidget(self.led_ProDay)
        self.led_ValDay = QtWidgets.QLineEdit(self.centralwidget)
        self.led_ValDay.setObjectName("led_ValDay")
        self.verticalLayout.addWidget(self.led_ValDay)
        self.led_code = QtWidgets.QLineEdit(self.centralwidget)
        self.led_code.setObjectName("led_code")
        self.verticalLayout.addWidget(self.led_code)
        self.led_Qcode = QtWidgets.QLineEdit(self.centralwidget)
        self.led_Qcode.setObjectName("led_Qcode")
        self.verticalLayout.addWidget(self.led_Qcode)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.gridLayout.addLayout(self.horizontalLayout, 1, 0, 1, 1)
        self.verticalLayout_3.addLayout(self.gridLayout)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_3.addWidget(self.label_2)
        self.teb_recode = QtWidgets.QTextBrowser(self.centralwidget)
        self.teb_recode.setObjectName("teb_recode")
        self.verticalLayout_3.addWidget(self.teb_recode)
        self.gridLayout_2.addLayout(self.verticalLayout_3, 0, 1, 1, 1)
        self.gridLayout_2.setColumnStretch(0, 3)
        self.gridLayout_2.setColumnStretch(1, 1)
        BottleDetect.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(BottleDetect)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 840, 25))
        self.menubar.setMinimumSize(QtCore.QSize(0, 25))
        self.menubar.setMaximumSize(QtCore.QSize(16777215, 30))
        self.menubar.setStyleSheet("background-color:rgb(232, 232, 232)")
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setAutoFillBackground(False)
        self.menu.setObjectName("menu")
        self.meun_2 = QtWidgets.QMenu(self.menubar)
        self.meun_2.setObjectName("meun_2")
        BottleDetect.setMenuBar(self.menubar)
        self.action_openImg = QtWidgets.QAction(BottleDetect)
        self.action_openImg.setObjectName("action_openImg")
        self.action_executive = QtWidgets.QAction(BottleDetect)
        self.action_executive.setObjectName("action_executive")
        self.action_openTem = QtWidgets.QAction(BottleDetect)
        self.action_openTem.setObjectName("action_openTem")
        self.menu.addAction(self.action_openImg)
        self.menu.addSeparator()
        self.menu.addAction(self.action_openTem)
        self.meun_2.addAction(self.action_executive)
        self.menubar.addAction(self.menu.menuAction())
        self.menubar.addAction(self.meun_2.menuAction())

        self.retranslateUi(BottleDetect)
        QtCore.QMetaObject.connectSlotsByName(BottleDetect)

    def retranslateUi(self, BottleDetect):
        _translate = QtCore.QCoreApplication.translate
        BottleDetect.setWindowTitle(_translate("BottleDetect", "MainWindow"))
        self.label.setText(_translate("BottleDetect", "The test results are displayed："))
        self.label_3.setText(_translate("BottleDetect", "Product lot number："))
        self.label_5.setText(_translate("BottleDetect", "Date of manufacture："))
        self.label_4.setText(_translate("BottleDetect", "Valid until："))
        self.label_6.setText(_translate("BottleDetect", "Barcode information："))
        self.label_7.setText(_translate("BottleDetect", "QR code information："))
        self.label_2.setText(_translate("BottleDetect", "Detection records："))
        self.menu.setTitle(_translate("BottleDetect", "open"))
        self.meun_2.setTitle(_translate("BottleDetect", "run"))
        self.action_openImg.setText(_translate("BottleDetect", "open image"))
        self.action_executive.setText(_translate("BottleDetect", "Perform detections"))
        self.action_openTem.setText(_translate("BottleDetect", "Open the template"))