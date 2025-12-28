# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'yolo11OUGLoz.ui'
##
## Created by: Qt User Interface Compiler version 6.10.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QComboBox, QFrame, QHBoxLayout,
    QLabel, QLineEdit, QMainWindow, QPushButton,
    QSizePolicy, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(865, 518)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.horizontalLayout_2 = QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.frame = QFrame(self.centralwidget)
        self.frame.setObjectName(u"frame")
        self.frame.setEnabled(True)
        self.frame.setMinimumSize(QSize(181, 500))
        self.frame.setMaximumSize(QSize(181, 500))
        self.frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame.setFrameShadow(QFrame.Shadow.Raised)
        self.comboBox = QComboBox(self.frame)
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.setObjectName(u"comboBox")
        self.comboBox.setGeometry(QRect(80, 10, 91, 26))
        self.label = QLabel(self.frame)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(10, 14, 61, 18))
        self.label_2 = QLabel(self.frame)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(10, 44, 61, 18))
        self.label_3 = QLabel(self.frame)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(10, 104, 63, 18))
        self.comboBox_3 = QComboBox(self.frame)
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.setObjectName(u"comboBox_3")
        self.comboBox_3.setGeometry(QRect(80, 100, 91, 26))
        self.label_5 = QLabel(self.frame)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setGeometry(QRect(10, 330, 66, 18))
        self.pushButton = QPushButton(self.frame)
        self.pushButton.setObjectName(u"pushButton")
        self.pushButton.setGeometry(QRect(40, 460, 101, 26))
        self.pushButton_2 = QPushButton(self.frame)
        self.pushButton_2.setObjectName(u"pushButton_2")
        self.pushButton_2.setGeometry(QRect(10, 66, 161, 26))
        self.pushButton_3 = QPushButton(self.frame)
        self.pushButton_3.setObjectName(u"pushButton_3")
        self.pushButton_3.setGeometry(QRect(10, 350, 161, 26))
        self.label_6 = QLabel(self.frame)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setGeometry(QRect(10, 140, 81, 18))
        self.lineEdit = QLineEdit(self.frame)
        self.lineEdit.setObjectName(u"lineEdit")
        self.lineEdit.setGeometry(QRect(100, 136, 71, 26))
        self.DS_NMS = QLabel(self.frame)
        self.DS_NMS.setObjectName(u"DS_NMS")
        self.DS_NMS.setGeometry(QRect(10, 174, 121, 18))
        self.DS_NMS_VAL = QLineEdit(self.frame)
        self.DS_NMS_VAL.setObjectName(u"DS_NMS_VAL")
        self.DS_NMS_VAL.setGeometry(QRect(132, 170, 40, 26))
        self.DS_OBJ = QLabel(self.frame)
        self.DS_OBJ.setObjectName(u"DS_OBJ")
        self.DS_OBJ.setGeometry(QRect(10, 204, 121, 18))
        self.DS_OBJ_VAL = QLineEdit(self.frame)
        self.DS_OBJ_VAL.setObjectName(u"DS_OBJ_VAL")
        self.DS_OBJ_VAL.setGeometry(QRect(132, 200, 40, 26))
        self.PO_NMS = QLabel(self.frame)
        self.PO_NMS.setObjectName(u"PO_NMS")
        self.PO_NMS.setGeometry(QRect(10, 234, 121, 18))
        self.PO_NMS_VAL = QLineEdit(self.frame)
        self.PO_NMS_VAL.setObjectName(u"PO_NMS_VAL")
        self.PO_NMS_VAL.setGeometry(QRect(132, 230, 41, 26))
        self.PO_OBJ = QLabel(self.frame)
        self.PO_OBJ.setObjectName(u"PO_OBJ")
        self.PO_OBJ.setGeometry(QRect(10, 264, 121, 18))
        self.PO_OBJ_VAL = QLineEdit(self.frame)
        self.PO_OBJ_VAL.setObjectName(u"PO_OBJ_VAL")
        self.PO_OBJ_VAL.setGeometry(QRect(132, 260, 41, 26))
        self.MAX_DETECT_VAL = QLineEdit(self.frame)
        self.MAX_DETECT_VAL.setObjectName(u"MAX_DETECT_VAL")
        self.MAX_DETECT_VAL.setGeometry(QRect(132, 290, 41, 26))
        self.MAX_DETECT = QLabel(self.frame)
        self.MAX_DETECT.setObjectName(u"MAX_DETECT")
        self.MAX_DETECT.setGeometry(QRect(8, 294, 121, 18))
        self.label_12 = QLabel(self.frame)
        self.label_12.setObjectName(u"label_12")
        self.label_12.setGeometry(QRect(10, 380, 121, 18))
        self.lineEdit_7 = QLineEdit(self.frame)
        self.lineEdit_7.setObjectName(u"lineEdit_7")
        self.lineEdit_7.setGeometry(QRect(10, 400, 161, 26))

        self.horizontalLayout_2.addWidget(self.frame)

        self.frame_2 = QFrame(self.centralwidget)
        self.frame_2.setObjectName(u"frame_2")
        self.frame_2.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_2.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout = QHBoxLayout(self.frame_2)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.label_4 = QLabel(self.frame_2)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.horizontalLayout.addWidget(self.label_4)


        self.horizontalLayout_2.addWidget(self.frame_2)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.pushButton_2.clicked.connect(MainWindow.model_select)
        self.pushButton_3.clicked.connect(MainWindow.image_select)
        self.comboBox.activated.connect(MainWindow.task_select)
        self.comboBox_3.activated.connect(MainWindow.dataset_select)
        self.pushButton.clicked.connect(MainWindow.inference)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.comboBox.setItemText(0, QCoreApplication.translate("MainWindow", u"Detect", None))
        self.comboBox.setItemText(1, QCoreApplication.translate("MainWindow", u"Pose", None))
        self.comboBox.setItemText(2, QCoreApplication.translate("MainWindow", u"Seg", None))
        self.comboBox.setItemText(3, QCoreApplication.translate("MainWindow", u"OBB", None))

        self.label.setText(QCoreApplication.translate("MainWindow", u"TASK", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"MODEL", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"DATASET", None))
        self.comboBox_3.setItemText(0, QCoreApplication.translate("MainWindow", u"COCO", None))
        self.comboBox_3.setItemText(1, QCoreApplication.translate("MainWindow", u"FIRE", None))
        self.comboBox_3.setItemText(2, QCoreApplication.translate("MainWindow", u"CRACK", None))
        self.comboBox_3.setItemText(3, QCoreApplication.translate("MainWindow", u"LICENSE", None))
        self.comboBox_3.setItemText(4, QCoreApplication.translate("MainWindow", u"GARBAGE", None))

        self.label_5.setText(QCoreApplication.translate("MainWindow", u"IMAGE", None))
        self.pushButton.setText(QCoreApplication.translate("MainWindow", u"INFERENCE", None))
        self.pushButton_2.setText(QCoreApplication.translate("MainWindow", u"Select", None))
        self.pushButton_3.setText(QCoreApplication.translate("MainWindow", u"Select", None))
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"INPUT_SIZE", None))
        self.lineEdit.setText(QCoreApplication.translate("MainWindow", u"640", None))
        self.DS_NMS.setText(QCoreApplication.translate("MainWindow", u"DS_NMS_THRESH", None))
        self.DS_NMS_VAL.setText(QCoreApplication.translate("MainWindow", u"0.45", None))
        self.DS_OBJ.setText(QCoreApplication.translate("MainWindow", u"DS_OBJ_THRESH", None))
        self.DS_OBJ_VAL.setText(QCoreApplication.translate("MainWindow", u"0.25", None))
        self.PO_NMS.setText(QCoreApplication.translate("MainWindow", u"PO_NMS_THRESH", None))
        self.PO_NMS_VAL.setText(QCoreApplication.translate("MainWindow", u"0.4", None))
        self.PO_OBJ.setText(QCoreApplication.translate("MainWindow", u"PO_OBJ_THRESH", None))
        self.PO_OBJ_VAL.setText(QCoreApplication.translate("MainWindow", u"0.5", None))
        self.MAX_DETECT_VAL.setText(QCoreApplication.translate("MainWindow", u"300", None))
        self.MAX_DETECT.setText(QCoreApplication.translate("MainWindow", u"MAX_DETECT", None))
        self.label_12.setText(QCoreApplication.translate("MainWindow", u"INFERENCE TIME", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"Image", None))
    # retranslateUi

