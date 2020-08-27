from tkinter import *
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt


def naive_command():
    list1.delete(0, END)
    data = pd.read_excel('Book2.xlsx')
    df = pd.DataFrame(data)
    Y_target = df['number of instances per class']
    X_data = df.drop(['number of instances per class'], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(X_data, Y_target)
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    y_predict = gnb.predict(x_test)
    plt.show()
    list1.insert(END, accuracy_score(y_test, y_predict))



def svc_command():
    list1.delete(0, END)
    data = pd.read_excel('Book2.xlsx')
    df = pd.DataFrame(data)
    Y_target = df['number of instances per class']
    X_data = df.drop(['number of instances per class'], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(X_data, Y_target)
    clf2=SVC(kernel='poly', degree=3, gamma='auto')
    clf2.fit(x_train, y_train)
    y_predict = clf2.predict(x_test)
    plt.show()
    list1.insert(END, accuracy_score(y_test, y_predict))


def decision_command():
    list1.delete(0, END)
    data = pd.read_excel('Book2.xlsx')
    df = pd.DataFrame(data)
    Y_target = df['number of instances per class']
    X_data = df.drop(['number of instances per class'], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(X_data, Y_target)
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)
    list1.insert(END, precision_score(y_test, y_predict,average='micro'))
    #plot_tree(clf.fit(x_train, y_train))


def NN_command():
    list1.delete(0, END)
    data = pd.read_excel('Book2.xlsx')
    df = pd.DataFrame(data)
    Y_target = df['number of instances per class']
    X_data = df.drop(['number of instances per class'], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(X_data, Y_target)
    clf = MLPClassifier(hidden_layer_sizes=(5, 3))
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)
    list1.insert(END, accuracy_score(y_test, y_predict))



wind = Tk()

list1 = Listbox(wind, height=3, width=20)
list1.grid(row=6, column=1, rowspan=6, columnspan=3)

b1 = Button(wind, text="Naive_bayes", width=12, command=naive_command)
b1.grid(row=2, column=1)

b2 = Button(wind, text="Supportvector", width=12, command=svc_command)
b2.grid(row=2, column=2)

b3 = Button(wind, text="DecisionTree", width=12, command=decision_command)
b3.grid(row=4, column=1)

b4 = Button(wind, text="NeraulNetwork", width=12, command=NN_command)
b4.grid(row=4, column=2)

wind.mainloop()