import sys
import time

from PyQt5.QtWidgets import *
from PyQt5 import uic
from matplotlib.figure import Figure

import EFS_GINI
import matplotlib.pyplot as plt
import seaborn as sn
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtGui import QPixmap, QStandardItemModel, QStandardItem


class MyWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.ui = uic.loadUi("./page.ui")
        # print(self.ui.__dict__)  # 查看ui文件中有哪些控件
        dataset_name = ['jaffe', 'COIL20', 'Isolet', 'lung', 'ORL', 'TOX_171', 'warpPIE10P', 'Prostate_GE']
        rate=['accuracy', 'precision','recall','f1_score','runtime']
        self.ui.comboBox.addItems(dataset_name)
        self.ui.comboBox.setCurrentIndex(-1)
        self.ui.comboBox_2.addItems(rate)
        self.ui.comboBox_2.setCurrentIndex(-1)
        self.datadset_name = self.ui.comboBox # 数据集名选择框
        self.percentage= self.ui.lineEdit  # 数据输入框
        self.run_btn = self.ui.pushButton  # 运行按钮
        self.clear_btn=self.ui.pushButton_2

        # 绑定信号与槽函数
        self.run_btn.clicked.connect(self.main)
        self.clear_btn.clicked.connect(self.clear_all)
        self.ui.comboBox_2.activated.connect(self.paint)


    def paint(self):
        selected=self.ui.comboBox_2.currentText()
        rate = ['accuracy', 'precision', 'recall', 'f1_score','runtime']
        select_index=rate.index(selected)+1
        x_data = ['MI','f_classif','ReliefF','SURF','SURF*','EFS-GINI']
        y_data = [float(self.model.index(row, select_index).data()) for row in range(self.model.rowCount())]
        width = self.ui.graphicsView_2.width()
        height = self.ui.graphicsView_2.height()
        fig = Figure()
        ax = fig.add_subplot(111)
        fig.set_size_inches(width / fig.dpi, height / fig.dpi)
        ax.plot(x_data, y_data, marker='s')
        ax.set_xlabel('method')
        ax.set_ylabel('rate')
        for i, j in zip(x_data, y_data):
            ax.text(i, j, str(j), ha='center', va='bottom')
        canvas = FigureCanvas(fig)
        self.ui.graphicsView_2.setScene(QGraphicsScene(self))
        self.ui.graphicsView_2.scene().addWidget(canvas)


    def show_result(self,text,runtime,method_name):
        row = [QStandardItem(method_name), QStandardItem(str(text[0])), QStandardItem(str(text[1])),
               QStandardItem(str(text[2])),QStandardItem(str(text[3])),QStandardItem(str(round(runtime,4)))]
        self.model.appendRow(row)

    def draw_conf(self,fig, ax, conf, row, col, name):
        sn.heatmap(conf, annot=True, cmap="YlGnBu", ax=ax[row, col])
        ax[row, col].set_title('{0} filter confusion matrix'.format(name))
        ax[row, col].set_xlabel('predict')  # x轴
        ax[row, col].set_ylabel('true')
        ax[row, col].set_aspect('equal')

    def clear_all(self):
        self.model.removeRows(0, self.model.rowCount())
        self.ui.graphicsView.scene().clear()
        self.ui.graphicsView_2.scene().clear()


    def main(self):
        name = self.datadset_name.currentText()
        percent_num = eval(self.percentage.text())/100
        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))
        X_train_scaled, X_test_scaled, y_train, y_test, k = EFS_GINI.load_dataset(name,percent_num)

        self.model = QStandardItemModel()
        self.model.setHorizontalHeaderLabels(['method', 'accuracy', 'precision','recall','f1_score','runtime'])
        self.ui.tableView.setModel(self.model)

        starttime=time.time()
        subset =EFS_GINI. mi_filter(X_train_scaled, y_train, k)
        runtime=time.time()-starttime
        result, conf = EFS_GINI.ESF(X_train_scaled, X_test_scaled, y_train, y_test, k, subset)
        self.show_result(result,runtime, "MI")
        self.draw_conf(fig, ax, conf, 0, 0, 'MI')

        starttime = time.time()
        subset = EFS_GINI.f_filter(X_train_scaled, y_train, k)
        runtime = time.time() - starttime
        result, conf = EFS_GINI.ESF(X_train_scaled, X_test_scaled, y_train, y_test, k, subset)
        self.show_result(result, runtime,"f_classif")
        self.draw_conf(fig, ax, conf, 0, 1, 'f_classif')

        starttime = time.time()
        subset = EFS_GINI.reliefF(X_train_scaled, y_train, k)
        runtime = time.time() - starttime
        result, conf = EFS_GINI.ESF(X_train_scaled, X_test_scaled, y_train, y_test, k, subset)
        self.show_result(result,runtime, "ReliefF")
        self.draw_conf(fig, ax, conf, 0, 2, 'ReliefF')

        starttime = time.time()
        subset = EFS_GINI.Surf(X_train_scaled, y_train, k)
        runtime = time.time() - starttime
        result, conf = EFS_GINI.ESF(X_train_scaled, X_test_scaled, y_train, y_test, k, subset)
        self.show_result(result,runtime, "SURF")
        self.draw_conf(fig, ax, conf, 1, 0, 'SURF')

        starttime = time.time()
        subset = EFS_GINI.Surfstar(X_train_scaled, y_train, k)
        runtime = time.time() - starttime
        result, conf = EFS_GINI.ESF(X_train_scaled, X_test_scaled, y_train, y_test, k, subset)
        self.show_result(result, runtime,"SURF*")
        self.draw_conf(fig, ax, conf, 1, 1, 'SURF*')

        starttime = time.time()
        subset = EFS_GINI.combiner(X_train_scaled, y_train, k)
        runtime = time.time() - starttime
        result, conf = EFS_GINI.ESF(X_train_scaled, X_test_scaled, y_train, y_test, k, subset)
        self.show_result(result,runtime, "EFS_GINI")
        self.draw_conf(fig, ax, conf, 1, 2, 'EFS-GINI')

        width = self.ui.graphicsView.width()
        height = self.ui.graphicsView.height()
        fig.subplots_adjust(hspace=0.5, wspace=0.2)
        fig.set_size_inches(width / fig.dpi, height / fig.dpi)
        canvas = FigureCanvas(fig)
        self.ui.graphicsView.setScene(QGraphicsScene(self))
        self.ui.graphicsView.scene().addWidget(canvas)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    w = MyWindow()
    # 展示窗口
    w.ui.show()

    app.exec()
