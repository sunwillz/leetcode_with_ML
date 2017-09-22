# -*- coding: utf-8 -*-

from sklearn import linear_model
from sklearn.model_selection import train_test_split

from metrics import *
from trapwater.data_process import *


def main():
    dataSet, labels = generate_data()

    # 将二维数组展开成一维数组
    X = np.reshape(dataSet, newshape=(data_size, -1))
    y = labels
    X_trian, X_test, y_trian, y_test = train_test_split(X, y, test_size=testing_percentage)

    lr = linear_model.LinearRegression()
    lr.fit(X_trian, y_trian)

    pred = np.array(lr.predict(X_test))
    for i in range(100):
        print y_test[i], '-------', pred[i]
    print 'accuarcy: {:.2f}'.format(accuracy(y_test, pred))
    print 'rmse:{}'.format(rmse(y_test, pred))

if __name__ == '__main__':
    main()