import numpy as np
import operator

def createDataSet():
   
    group = np.array([[1,101],[5,89],[108,5],[115,8]])
    
    labels = ['romantic','romantic','action','action']
    return group, labels


def classify0(inX, dataSet, labels, k):
    
    dataSetSize = dataSet.shape[0]
    #0 is the size of row, 1 is the features in each so it is 2 
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    #coping inX to a matrix for each calculation 
    #inx is our x1 y1
    sqDiffMat = diffMat**2
    #sum()
    sqDistances = sqDiffMat.sum(axis=1)
    #axis 1 means one row
    distances = sqDistances**0.5
 
    sortedDistIndices = distances.argsort()
    #from smallest to biggest
    classCount = {}
    for i in range(k):
        
        voteIlabel = labels[sortedDistIndices[i]]
        
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
        #get the count, if not appears efire give 0

   
   
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
     #key=operator.itemgetter(1) order by the counting value
    
    return sortedClassCount[0][0]

if __name__ == '__main__':
    
    group, labels = createDataSet()
    
    test = [101,20]
    
    test_class = classify0(test, group, labels, 3)
    
    print(test_class)