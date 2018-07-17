# -*- coding: utf-8 -*-
import sys
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import *
import numpy as np
import pylab as pl
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor
from sklearn import model_selection
from sklearn import metrics
from sklearn.metrics import make_scorer

pl.rcParams['font.sans-serif'] = ['FangSong']
pl.rcParams['axes.unicode_minus'] = False

class Boston(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.startUI()      # UI界面开发设计
        self.show()         # 显示界面

    def startUI(self):
        # UI界面布局设计
        self.setFixedSize(900,600)
        self.setWindowTitle('波士顿房价预测（决策树方法）')

        titleLabel = QLabel("波士顿房价预测",self)
        titleLabel.setFont(QFont("Roman times",15 ,QFont.Bold))
        titleLabel.move(380, 40)
        authorLabel = QLabel("作者：笔尖 / bj",self)
        authorLabel.move(760, 70)

        depLabel = QLabel(" 查看不同深度的学习曲线：",self)
        depLabel.move(50,510)
        self.depCombo = QComboBox(self)
        self.depCombo.setFixedSize(100,27)
        self.depCombo.move(270,510)
        for i in range(1, 11):
            self.depCombo.addItem(str(i))
        self.depCombo.addItem('ALL')
        depBtn = QPushButton("确定",self)
        depBtn.setFixedHeight(30)
        depBtn.move(390,510)
        depBtn.clicked.connect(self.show_curve)

        self.CRIM_Edit = QLineEdit("11.95")
        self.ZN_Edit = QLineEdit("0.00")
        self.INDUS_Edit = QLineEdit("18.100")
        self.CHAS_Edit = QLineEdit("0")
        self.NOX_Edit = QLineEdit("0.6590")
        self.RM_Edit = QLineEdit("5.6090")
        self.AGE_Edit = QLineEdit("90.00")
        self.DIS_Edit = QLineEdit("1.385")
        self.RAD_Edit = QLineEdit("24")
        self.TAX_Edit = QLineEdit("680.0")
        self.PTRATIO_Edit = QLineEdit("20.20")
        self.B_Edit = QLineEdit("332.09")
        self.LSTAT_Edit = QLineEdit("12.13")

        chooseForm = QFormLayout()
        chooseForm.addRow(QLabel("城镇人均犯罪率/CRIM"), self.CRIM_Edit)
        chooseForm.addRow(QLabel("住宅用地25000英尺的比例/ZN"), self.ZN_Edit)
        chooseForm.addRow(QLabel("城镇中非商业用地的比例/INDUS"), self.INDUS_Edit)
        chooseForm.addRow(QLabel("边界是否查理斯河/CHAS"), self.CHAS_Edit)
        chooseForm.addRow(QLabel("一氧化氮浓度/NOX"), self.NOX_Edit)
        chooseForm.addRow(QLabel("住宅平均房间数/RM"), self.RM_Edit)
        chooseForm.addRow(QLabel("1940年前建成的自用房屋比例/AGE  "), self.AGE_Edit)
        chooseForm.addRow(QLabel("到波士顿五个中心区域的距离/DIS  "), self.DIS_Edit)
        chooseForm.addRow(QLabel("距离高速公路的便利指数/RAD"), self.RAD_Edit)
        chooseForm.addRow(QLabel("每一万美元的全值财产税率/TAX"), self.TAX_Edit)
        chooseForm.addRow(QLabel("城镇中师生比例/PTRATIO"), self.PTRATIO_Edit)
        chooseForm.addRow(QLabel("城镇中黑人比例/B"), self.B_Edit)
        chooseForm.addRow(QLabel("房东是低收入阶层的的比例/LSTAT"), self.LSTAT_Edit)
        chooseWidget = QWidget(self)
        chooseWidget.setLayout(chooseForm)
        chooseWidget.move(50, 100)

        chooseBtn = QPushButton("预测房价",self)
        chooseBtn.setFixedSize(200,50)
        chooseBtn.move(600,500)
        chooseBtn.clicked.connect(self.show_result)

        self.showEdit = QTextEdit(self)
        self.showEdit.setFixedSize(350,360)
        self.showEdit.move(520,110)

        self.city_data = self.load_data()    # 载入波士顿房价数据
        self.split_data(self.city_data)

    def load_data(self):
        boston = datasets.load_boston()
        return boston

    def explore_city_data(self,city_data):
        housing_prices = city_data.target
        housing_features = city_data.data
        num_houses = np.shape(city_data.data)
        num_features = np.shape(city_data.data)
        min_price = np.min(city_data.target)
        max_price = np.max(city_data.target)
        mean_price = np.mean(city_data.target)
        median_price = np.median(city_data.target)
        stand_dev = np.std(city_data.target)

        baseInfo = str("====================================="+
                    "\n\nBoston房价数据集中共有"+str(num_houses)+"组数据。\n\n"+
                    "\n房价最高为"+str(max_price)+"\n房价最低为"+str(min_price)+
                    "\n房价中值为"+str(median_price)+"\n房价均值为"+str(mean_price)+
                    "\n房价标准差为"+str(stand_dev))
        self.showEdit.append(baseInfo)

    def split_data(self,city_data):
       X, y = city_data.data, city_data.target
       self.X_train, self.X_test, self.y_train, self.y_test = model_selection.train_test_split(X, y, test_size=0.30, random_state=42)

    def performance_metric(self,label, prediction):
        mse = metrics.mean_squared_error(label, prediction)
        return mse

    def learning_curve(self, depth, X_train, y_train, X_test, y_test):
        sizes = np.round(np.linspace(1, len(X_train), 50))
        train_err = np.zeros(len(sizes))
        test_err = np.zeros(len(sizes))

        for i, s in enumerate(sizes):
            s = int(s)
            regressor = DecisionTreeRegressor(max_depth=depth)
            regressor.fit(X_train[:s], y_train[:s])
            train_err[i] = self.performance_metric(y_train[:s], regressor.predict(X_train[:s]))
            test_err[i] = self.performance_metric(y_test, regressor.predict(X_test))

        self.learning_curve_graph(depth, sizes, train_err, test_err)

    def learning_curve_graph(self, depth, sizes, train_err, test_err):
        pl.figure(depth)
        pl.title(str('决策树(深度为'+str(depth)+')'))
        pl.plot(sizes, test_err, lw=2, label = '测试误差')
        pl.plot(sizes, train_err, lw=2, label = '训练误差')
        pl.legend()
        pl.xlabel('Training Size')
        pl.ylabel('Error')
        pl.show()

    def model_complexity(self, X_train, y_train, X_test, y_test):
        max_depth = np.arange(1, 25)
        train_err = np.zeros(len(max_depth))
        test_err = np.zeros(len(max_depth))

        for i, d in enumerate(max_depth):
            regressor = DecisionTreeRegressor(max_depth=d)
            regressor.fit(X_train, y_train)
            train_err[i] = self.performance_metric(y_train, regressor.predict(X_train))
            test_err[i] = self.performance_metric(y_test, regressor.predict(X_test))

        self.model_complexity_graph(max_depth, train_err, test_err)

    def model_complexity_graph(self, max_depth, train_err, test_err):
        pl.figure(0)
        pl.title('波士顿房价预测模型——决策树')
        pl.plot(max_depth, test_err, lw=2, label = '测试误差')
        pl.plot(max_depth, train_err, lw=2, label = '训练误差')
        pl.legend()
        pl.xlabel('Max Depth')
        pl.ylabel('Error')
        pl.show()

    def fit_predict_model(self,city_data):
        X, y = city_data.data, city_data.target
        regressor = DecisionTreeRegressor()
        parameters = {'max_depth':(1,2,3,4,5,6,7,8,9,10)}
        scores = make_scorer(metrics.mean_squared_error, greater_is_better=False)
        clf = model_selection.GridSearchCV(regressor, parameters, scoring=scores)
        clf.fit(X,y)

        best_clf = clf.best_estimator_
        print(best_clf)
        # x = [[11.95, 0.00, 18.100, 0, 0.6590, 5.6090, 90.00, 1.385, 24, 680.0, 20.20, 332.09, 12.13]]
        x = [[float(self.CRIM_Edit.text()),float(self.ZN_Edit.text()),float(self.INDUS_Edit.text()),float(self.CHAS_Edit.text())
                 , float(self.NOX_Edit.text()),float(self.RM_Edit.text()),float(self.AGE_Edit.text()),float(self.DIS_Edit.text())
                 , float(self.RAD_Edit.text()),float(self.TAX_Edit.text()),float(self.PTRATIO_Edit.text()),float(self.B_Edit.text())
                 , float(self.LSTAT_Edit.text())]]
        y = best_clf.predict(x)

        paramInfo = str('\n\n最佳模型：'+ str(clf.best_params_))
        houseInfo = str('\n\n房屋参数为：\nCRIM   :' + self.CRIM_Edit.text() + "\tZN  :" + self.ZN_Edit.text()+ "\n" +
                    "INDUS  :" + self.INDUS_Edit.text() + "\tCHAS:" + self.CHAS_Edit.text() + "\n" +
                    "NOX    :" + self.NOX_Edit.text() +  "\tRM  :" + self.RM_Edit.text() + "\n" +
                    "AGE    :" + self.AGE_Edit.text() + "\tDIS :" + self.DIS_Edit.text() + "\n" +
                    "RAD    :" + self.RAD_Edit.text() + "\tTAX :" + self.TAX_Edit.text() + "\n" +
                    "PTRATIO:" + self.PTRATIO_Edit.text() + "\tB   :" + self.B_Edit.text() + "\n" +
                    "LSTAT  :" + self.LSTAT_Edit.text())
        priceInfo = str('\n\n房子的预测价格为：' + str(y[0]) + '\n\n')
        self.showEdit.append(paramInfo)
        self.showEdit.append(houseInfo)
        self.showEdit.append(priceInfo)

    def show_curve(self):
        depth = self.depCombo.currentIndex() + 1
        if depth == 11:
            max_depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            for max_depth in max_depths:
                self.learning_curve(max_depth, self.X_train, self.y_train, self.X_test, self.y_test)
        else:
            self.learning_curve(depth, self.X_train, self.y_train, self.X_test, self.y_test)

    def show_result(self):
        self.explore_city_data(self.city_data)
        self.model_complexity(self.X_train, self.y_train, self.X_test, self.y_test)
        self.fit_predict_model(self.city_data)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Boston()
    sys.exit(app.exec_()) 