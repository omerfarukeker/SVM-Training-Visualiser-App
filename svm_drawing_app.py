# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 23:53:07 2019
SVM Desktop Application
@author: omerzulal
"""

from PyQt5.QtWidgets import*
from PyQt5.uic import loadUi
from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn import preprocessing
from sklearn.svm import SVC

#load iris data from sklearn datasets
iris = load_iris()
     
class MatplotlibWidget(QMainWindow):
    
    def __init__(self):      
        QMainWindow.__init__(self)
        
        #the .ui file we created in Qt Designer
        loadUi("svm_drawing_app.ui",self)
        
        self.setWindowTitle("PyQt5 & Matplotlib Example GUI")
        
        self.radioButton_linear.clicked.connect(self.plot_svm)
        self.radioButton_rbf.clicked.connect(self.plot_svm)
        
        self.doubleSpinBox_c.valueChanged.connect(self.plot_svm)
        self.doubleSpinBox_gamma.valueChanged.connect(self.plot_svm)
        
        self.dial_c.valueChanged.connect(self.doubleSpinBox_c.setValue)
        self.dial_gamma.valueChanged.connect(self.doubleSpinBox_gamma.setValue)
        
        self.doubleSpinBox_c.valueChanged.connect(self.dial_c.setValue)
        self.doubleSpinBox_gamma.valueChanged.connect(self.dial_gamma.setValue)
        
        self.comboBox_f1.currentTextChanged.connect(self.plot_svm)
        self.comboBox_f2.currentTextChanged.connect(self.plot_svm)

        self.addToolBar(NavigationToolbar(self.MplWidget.canvas, self))
        
    def get_C(self):
        return float(self.doubleSpinBox_c.value())
    
    def get_gamma(self):
        return float(self.doubleSpinBox_gamma.value())
    
    def get_kernel(self):
        if self.radioButton_rbf.isChecked():
            return "rbf"
        elif self.radioButton_linear.isChecked():
            return "linear"
    
    def get_fs(self):
        return [str(self.comboBox_f1.currentText()),str(self.comboBox_f2.currentText())]

    def plot_svm(self):
        #take SVM choices from user
        C = self.get_C()
        kernel = self.get_kernel()
        gamma = self.get_gamma()
        feature_1 = self.get_fs()[0].lower()
        feature_2 = self.get_fs()[1].lower()

        features = [feature_1,feature_2]
        
        #only versicolor and virginica are the targets
        flowers = [1,2]
        target_cond = (iris.target == flowers[0]) | (iris.target == flowers[1])
        
        #construct a dataframe with the conditions
        df_features = pd.DataFrame(preprocessing.scale(iris.data[target_cond,:]),
                                   columns = iris.feature_names)
        df_features = df_features[features]
        df_targets = pd.DataFrame(iris.target[target_cond],columns=["Targets"])
        df = pd.concat([df_features,df_targets],axis=1)
        
        #shuffle the dataset
        df = df.reindex(np.random.RandomState(seed=2).permutation
                        (df.index))
        df = df.reset_index(drop=True)
        
        #generate an SVM model
        if kernel == "linear":
            SVM_model = SVC(kernel=kernel,C=C)
        elif kernel == "rbf":
            SVM_model = SVC(kernel=kernel,C=C,gamma=gamma)
        
        #training data
        X = df.iloc[:,:2]
        y = df["Targets"]
        SVM_model.fit(X,y)
        
        classes = df.Targets.value_counts().index
        scores = SVM_model.score(X,y)
        self.lcdNumber_accuracy.display(scores)
        
        f1 = df[features[0]]
        f2 = df[features[1]]
    
        self.MplWidget.canvas.axes.clear()
        f11 = self.MplWidget.canvas.axes.scatter(f1[y==classes[0]],f2[y==classes[0]],c="cornflowerblue",marker="o")
        f12 = self.MplWidget.canvas.axes.scatter(f1[y==classes[1]],f2[y==classes[1]],c="sandybrown",marker="^")
        
        #draw the SVM boundary line
        #prepare data for decision boundary plotting
        x_min = X.iloc[:,0].min()
        x_max = X.iloc[:,0].max()
        y_min = X.iloc[:,1].min()
        y_max = X.iloc[:,1].max()
        XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
        Z1 = SVM_model.decision_function(np.c_[XX.ravel(), YY.ravel()])
        Z1 = Z1.reshape(XX.shape)

#        set xticks off
        self.MplWidget.canvas.axes.set_xticklabels([])
        self.MplWidget.canvas.axes.set_yticklabels([])
        
        #plot the decision boundary
        self.MplWidget.canvas.axes.contour(XX, YY, Z1, colors=['darkgrey','dimgrey','darkgrey'],
                    linestyles=[':', '--', ':'], levels=[-.5, 0, .5])
        
        #title wont work with celluloid package, text is an alternative to workaround
        self.MplWidget.canvas.axes.text(0.15, 1.03, f"SVM Training Visualisation for the Iris Dataset",
                                        fontweight="bold", transform=self.MplWidget.canvas.axes.transAxes)

        self.MplWidget.canvas.axes.set_xlim([-3,3])
        self.MplWidget.canvas.axes.set_ylim([-3,3])
        
        #x-y labels
        self.MplWidget.canvas.axes.set_xlabel(f"F1: {feature_1}")
        self.MplWidget.canvas.axes.set_ylabel(f"F2: {feature_2}")
        
        #legend
        self.MplWidget.canvas.axes.legend([f11,f12],iris.target_names[flowers],fontsize=9)
        
        #draw the plot
        self.MplWidget.canvas.draw()

#run the application
app = QApplication([])
window = MatplotlibWidget()
window.show()
app.exec_()