import matplotlib.pyplot as plt
import numpy as np 
"""
function description: load the dataset 
pafram: none
returns : dataMat-data list
labelMat - label's list
"""

def loadDataSet():
    dataMat = []                                                       
    labelMat = []                                                        
    fr = open('testSet.txt')                                            
    for line in fr.readlines():                                            
        lineArr = line.strip().split()                                    
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])       
        labelMat.append(int(lineArr[2]))                                
    fr.close()                                                            
    return dataMat, labelMat                                            
 
"""
function description: visualize the dataset
param: None
returns:None
"""
#def plotDataSet():
#   dataMat,labelMat=loadDataSet()
 #   dataArr=np.array(dataMat)
 #  n= np.shape(dataMat)[0] # how many samples
   # xcord1 =[];ycord1=[] #positive sample
  #  xcord2=[];ycord2=[] #negative sample
   # for i in range(n):
   #     if int(labelMat[i])==1:
   #         xcord1.append(dataArr[i,1]);ycord1.append(dataArr[i,2])
   #     else:
   #         xcord2.append(dataArr[i,1]);ycord2.append(dataArr[i,2])

    #fig=plt.figure()
   # ax= fig.add_subplot(111)
    #ax.scatter(xcord1,ycord1,s=20,c='red',marker='s',alpha=.5)
    #ax.scatter(xcord2,ycord2,s=20,c='green',alpha=.5)
    #plt.title('Dataset')
    #plt.xlabel('x')
    #plt.ylabel('y')
    #plt.show()
"""
function description : sigmoid function 
param: inX - data
returns: sigmoid


"""
def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))

"""
function description: gradient ascent 
param:
dataMatIn
classLabels
return weights.getA() 

"""
def gradAscent(dataMatIn,classLabels):
    dataMatrix = np.mat(dataMatIn) # convet to numpy's matrix
    labelMat = np.mat(classLabels).transpose()
    m,n= np.shape(dataMatrix)
    alpha= 0.001
    maxCycles = 500 # maximum iteration
    weights= np.ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)
        error = labelMat-h
        weights = weights + alpha * dataMatrix.transpose()*error
    return weights.getA()

def plotBestFit(weights):
    dataMat, labelMat = loadDataSet()                                   
    dataArr = np.array(dataMat)                                            
    n = np.shape(dataMat)[0]                                            
    xcord1 = []; ycord1 = []                                            
    xcord2 = []; ycord2 = []                                            
    for i in range(n):                                                    
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])    
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])    
    fig = plt.figure()
    ax = fig.add_subplot(111)                                            
    ax.scatter(xcord1, ycord1, s = 20, c = 'red', marker = 's',alpha=.5)
    ax.scatter(xcord2, ycord2, s = 20, c = 'green',alpha=.5)            
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.title('BestFit')                                               
    plt.xlabel('X1'); plt.ylabel('X2')                                   
    plt.show()       
 
     

if __name__ =="__main__":
        dataMat,labelMat= loadDataSet()
        weights = gradAscent(dataMat,labelMat)
        plotBestFit(weights)