# -*- coding: utf-8 -*-

import numpy as np
import heapq


data_size = 10000  # 数据总量
rows = 10  # 输入矩阵行数
cols = 10  # 数据矩阵列数
max_ele = 8  # 元素最大高度
testing_percentage = 0.3  # 测试集


# 传统解法
def trapRainWater(heightMap):

    if not heightMap or not heightMap[0]:
        return 0

    m, n = len(heightMap), len(heightMap[0])
    heap = []
    visited = [[0] * n for _ in xrange(m)]

    for i in xrange(m):
        for j in xrange(n):
            if i == 0 or j == 0 or i == m - 1 or j == n - 1:
                heapq.heappush(heap, (heightMap[i][j], i, j))
                visited[i][j] = 1

    result = 0
    while heap:
        height, i, j = heapq.heappop(heap)
        for x, y in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
            if 0 <= x < m and 0 <= y < n and not visited[x][y]:
                result += max(0, height - heightMap[x][y])
                heapq.heappush(heap, (max(heightMap[x][y], height), x, y))
                visited[x][y] = 1
    return result


# 生成数据
def generate_data():
    dataSet = []
    labels = []

    for i in range(data_size):
        data = np.random.randint(0, max_ele + 1, size=(rows, cols))
        dataSet.append(data)
        labels.append(trapRainWater(data.tolist()))

    return np.array(dataSet), np.array(labels)


def main():
    dataSet, labels = generate_data()
    print dataSet.shape
    print labels.shape

if __name__ == '__main__':
    main()
