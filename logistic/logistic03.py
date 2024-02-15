import matplotlib.pyplot as plt
import numpy as np
import random

#load the dataset

def loadDataSet():
    dataMat = []
    labelMat =[]
    fr=open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    fr.close()
    return dataMat,labelMat

#sigmoid function
def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))

#modifies grad ascent algorithm

def stocGradAscent1(dataMatrix,classLabels,numIter=150):
    m,n = np.shapes(dataMatrix)
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01
            randIndex = int(random.uniform(0,len(dataIndex)))
            h=sigmoid(sum(dataMatrix[dataIndex[randIndex]*weights]))
            error = classLabels[dataIndex[randIndex]]-h
            weights = weights+alpha*error*dataMatrix[dataIndex[randIndex]]
            del(dataIndex[randIndex])
    return weights

