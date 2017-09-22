# -*- coding: utf-8 -*-

import numpy as np

# 数据总量,2**data_size
num_size = 12
data_size = 2**num_size


def fizzBuzz(n):
    """
    :type n: int
    :rtype: List[str]
    """
    res = []
    for i in range(1, n + 1):
        if i % 3 == 0 and i % 5 == 0:
            res.append('FizzBuzz')
        elif i % 3 == 0:
            res.append('Fizz')
        elif i % 5 == 0:
            res.append('Buzz')
        else:
            res.append(str(i))
    return res

def generate_data():
    dataSet = np.arange(1, data_size+1)
    labels = fizzBuzz(data_size)

    return np.array(dataSet), np.array(labels)


def label_to_categorical(data):
    labels = []
    for i in range(len(data)):
        if data[i] == 'Fizz':
            labels.append([1, 0, 0, 0])
        elif data[i] == 'Buzz':
            labels.append([0, 1, 0, 0])
        elif data[i] == 'FizzBuzz':
            labels.append([0, 0, 1, 0])
        else:
            labels.append([0, 0, 0, 1])

    return np.array(labels)


def binary_encode(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)])


def data_formation(labels):
    labels = label_to_categorical(labels)
    data_df = np.array([binary_encode(i, num_size) for i in range(2 ** num_size)])
    return data_df, labels


def main():
    dataSet, labels = generate_data()
    dataSet, labels = data_formation(dataSet, labels)

    print dataSet.shape
    print labels

if __name__ == '__main__':
    main()
