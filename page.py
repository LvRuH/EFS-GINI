# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'page.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_EFS(object):
    def setupUi(self, EFS):
        EFS.setObjectName("EFS")
        EFS.resize(1415, 974)
        self.label = QtWidgets.QLabel(EFS)
        self.label.setGeometry(QtCore.QRect(250, 30, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.comboBox = QtWidgets.QComboBox(EFS)
        self.comboBox.setGeometry(QtCore.QRect(370, 30, 131, 31))
        self.comboBox.setObjectName("comboBox")
        self.label_2 = QtWidgets.QLabel(EFS)
        self.label_2.setGeometry(QtCore.QRect(730, 30, 131, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.lineEdit = QtWidgets.QLineEdit(EFS)
        self.lineEdit.setGeometry(QtCore.QRect(860, 30, 131, 31))
        self.lineEdit.setObjectName("lineEdit")
        self.label_3 = QtWidgets.QLabel(EFS)
        self.label_3.setGeometry(QtCore.QRect(1000, 40, 72, 15))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.pushButton = QtWidgets.QPushButton(EFS)
        self.pushButton.setGeometry(QtCore.QRect(320, 80, 93, 28))
        self.pushButton.setObjectName("pushButton")
        self.graphicsView = QtWidgets.QGraphicsView(EFS)
        self.graphicsView.setGeometry(QtCore.QRect(0, 490, 1421, 491))
        self.graphicsView.setObjectName("graphicsView")
        self.tableView = QtWidgets.QTableView(EFS)
        self.tableView.setGeometry(QtCore.QRect(0, 110, 771, 381))
        self.tableView.setObjectName("tableView")
        self.pushButton_2 = QtWidgets.QPushButton(EFS)
        self.pushButton_2.setGeometry(QtCore.QRect(830, 80, 93, 28))
        self.pushButton_2.setObjectName("pushButton_2")
        self.graphicsView_2 = QtWidgets.QGraphicsView(EFS)
        self.graphicsView_2.setGeometry(QtCore.QRect(770, 110, 651, 381))
        self.graphicsView_2.setObjectName("graphicsView_2")
        self.comboBox_2 = QtWidgets.QComboBox(EFS)
        self.comboBox_2.setGeometry(QtCore.QRect(820, 130, 101, 21))
        self.comboBox_2.setObjectName("comboBox_2")

        self.retranslateUi(EFS)
        QtCore.QMetaObject.connectSlotsByName(EFS)

    def retranslateUi(self, EFS):
        _translate = QtCore.QCoreApplication.translate
        EFS.setWindowTitle(_translate("EFS", "EFS-GINI"))
        self.label.setText(_translate("EFS", "请选择数据集"))
        self.label_2.setText(_translate("EFS", "请输入特征占比"))
        self.label_3.setText(_translate("EFS", "%"))
        self.pushButton.setText(_translate("EFS", "运行"))
        self.pushButton_2.setText(_translate("EFS", "清屏"))
