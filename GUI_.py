# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'GUI_.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from tkinter.filedialog import askopenfilename, asksaveasfilename
import cv2


class Ui_MainWindow(object):
    filename = ""
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("Dog Detector")
        MainWindow.resize(1114, 883)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.input_image_field = QtWidgets.QLabel(self.centralwidget)
        self.input_image_field.setGeometry(QtCore.QRect(10, 120, 391, 281))
        self.input_image_field.setText("")
        self.input_image_field.setPixmap(QtGui.QPixmap("C:/Users/Somto\'s PC/OneDrive - MNSCU/Pictures/Capture.PNG"))
        self.input_image_field.setScaledContents(True)
        self.input_image_field.setObjectName("input_image_field")
        self.choose_image_btn = QtWidgets.QPushButton(self.centralwidget)
        self.choose_image_btn.setGeometry(QtCore.QRect(80, 570, 151, 81))
        self.choose_image_btn.setObjectName("choose_image_btn")
        self.detect_btn = QtWidgets.QPushButton(self.centralwidget)
        self.detect_btn.setGeometry(QtCore.QRect(440, 570, 151, 81))
        self.detect_btn.setObjectName("detect_btn")
        self.output_image_field = QtWidgets.QLabel(self.centralwidget)
        self.output_image_field.setGeometry(QtCore.QRect(660, 120, 361, 291))
        self.output_image_field.setText("")
        self.output_image_field.setPixmap(QtGui.QPixmap("C:/Users/Somto\'s PC/OneDrive - MNSCU/Pictures/Capture.PNG"))
        self.output_image_field.setScaledContents(True)
        self.output_image_field.setObjectName("output_image_field")
        self.save_image_btn = QtWidgets.QPushButton(self.centralwidget)
        self.save_image_btn.setGeometry(QtCore.QRect(800, 560, 151, 81))
        self.save_image_btn.setObjectName("save_image_btn")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(360, 30, 371, 51))
        font = QtGui.QFont()
        font.setPointSize(26)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1114, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.choose_image_btn.clicked.connect(self.getImage)
        self.detect_btn.clicked.connect(self.detectImage)
        #self.save_image_btn.clicked.connect(self.saveImage)

    def getImage(self):
        self.filename = askopenfilename()
        print("Type: ", type(self.filename))
        if(self.filename):
            self.input_image_field.setPixmap(QtGui.QPixmap(self.filename))

    def detectImage(self):
        filepath = self.filename
        img = cv2.imread(filepath)
        img = cv2.rectangle(img, (100,100), (500,500), (0, 0, 255), 2)
        #cv2.imshow("window", img)
        savedImage = "savedImage.jpg"
        cv2.imwrite(savedImage,img)
        self.output_image_field.setPixmap(QtGui.QPixmap(savedImage))

    def saveImage(self):
        asksaveasfilename()


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Dog Detector"))
        self.choose_image_btn.setText(_translate("MainWindow", "Choose Image"))
        self.detect_btn.setText(_translate("MainWindow", "Detect"))
        self.save_image_btn.setText(_translate("MainWindow", "Save Image"))
        self.label.setText(_translate("MainWindow", "Dog Detector"))





if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
