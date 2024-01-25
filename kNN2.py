# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.font_manager import FontProperties
import operator

def classify0(inX, dataSet, labels, k):
 
    dataSetSize = dataSet.shape[0]
  
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
 
    sqDiffMat = diffMat**2

    sqDistances = sqDiffMat.sum(axis=1)

    distances = sqDistances**0.5

    sortedDistIndices = distances.argsort()

    classCount = {}
    for i in range(k):

        voteIlabel = labels[sortedDistIndices[i]]

        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1

    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
   
    return sortedClassCount[0][0]


def file2matrix(filename):
    # ... (your file2matrix function remains the same)
    fr = open(filename)
    arraylines = fr.readlines()
    numberlines = len(arraylines)
    #3 column , numberlines rows
    returnMat = np.zeros((numberlines,3))
    classLabelVector =[]
    #this is the label
    index=0
    for line in arraylines:
        line = line.strip()
        #delete \n,\t,\r
        listfromline = line.split('\t')
        returnMat[index,:]=listfromline[0:3]
        # these 3 are the characteristic
        if listfromline[-1] == 'didntLike':
            classLabelVector.append(1)
        elif listfromline[-1] == 'smallDoses':
            classLabelVector.append(2)
        elif listfromline[-1] == 'largeDoses':
            classLabelVector.append(3)
        index +=1
    return returnMat,classLabelVector


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals-minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m= dataSet.shape[0] #rows num
    normDataSet=dataSet-np.tile(minVals,(m,1))
    #original - min
    normDataSet= normDataSet/np.tile(ranges,(m,1))
    return normDataSet,ranges,minVals

def datingclasstest():
    filename="datingtestset.txt"
    datingDataMat,datingLabels=file2matrix(filename)
    horatio = 0.1 # take 10% to test
    normMat, ranges, minVals= autoNorm(datingDataMat)
    #normmat rows
    m = normMat.shape[0]

    numTestVecs = int (m*horatio)
    errorCount =0
    for i in range (numTestVecs):
        classfierRes = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],4)
        print("result: %d\t real : %d" %(classfierRes,datingLabels[i]))
        if classfierRes != datingLabels[i]:
            errorCount+=1.0
        print("error rate%f%%"%(errorCount/float(numTestVecs)*100))


def classifyperson():
    resultlist = ['nah','maybe','like']
    precentTats= float(input("game time:"))
    ffmiles = float(input ("miles:"))
    icecream = float(input("ice cream per week"))
    filename= "datingtestset.txt"
    datingDataMat,datingLabels= file2matrix(filename)
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr= np.array([ffmiles,precentTats,icecream])
    normArr = (inArr-minVals)/ranges
    classfierRes = classify0(normArr,normMat,datingLabels,3)
    print("you may %s the person "%(resultlist[classfierRes-1]))

def showdatas(datingDataMat, datingLabels):
   
    
    
    # Create subplots
    figs, axs = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(13, 8))
    
    # Get the number of labels
    numberOfLabels = len(datingLabels)
    
    LabelsColors = []
    for i in datingLabels:
        if i == 1:
            LabelsColors.append('black')
        if i == 2:
            LabelsColors.append('orange')
        if i == 3:
            LabelsColors.append('red')
    
    # Scatter plot for the first subplot
    axs[0][0].scatter(x=datingDataMat[:, 0], y=datingDataMat[:, 1], c=LabelsColors, s=15, alpha=0.5)
    
    axs0_title = axs[0][0].set_title(u'time of airline miles vs  spent on game evey year')
    axs0_xlabel = axs[0][0].set_xlabel(u'airline miles every year')
    axs0_ylabel = axs[0][0].set_ylabel(u'time spend on game')
    
    plt.setp(axs0_title, size=9, weight='bold', color='red')
    plt.setp(axs0_xlabel, size=7, weight='bold', color='black')
    plt.setp(axs0_ylabel, size=7, weight='bold', color='black')
    
    # Scatter plot for the second subplot
    axs[0][1].scatter(x=datingDataMat[:, 0], y=datingDataMat[:, 2], c=LabelsColors, s=15, alpha=0.5)
    
    axs1_title = axs[0][1].set_title(u'time of airline miles vs volume of ice cream every week')
    axs1_xlabel = axs[0][1].set_xlabel(u'airline miles every year')
    axs1_ylabel = axs[0][1].set_ylabel(u'volume of ice cream')
    
    plt.setp(axs1_title, size=9, weight='bold', color='red')
    plt.setp(axs1_xlabel, size=7, weight='bold', color='black')
    plt.setp(axs1_ylabel, size=7, weight='bold', color='black')
    
    # Scatter plot for the third subplot
    axs[1][0].scatter(x=datingDataMat[:, 1], y=datingDataMat[:, 2], c=LabelsColors, s=15, alpha=0.5)
    
    axs2_title = axs[1][0].set_title(u'game vs ice cream')
    axs2_xlabel = axs[1][0].set_xlabel(u'game time')
    axs2_ylabel=axs[1][0].set_ylabel(u'ice')
    
    plt.setp(axs2_title, size=9, weight='bold', color='red')
    plt.setp(axs2_xlabel, size=7, weight='bold', color='black')
    plt.setp(axs2_ylabel, size=7, weight='bold', color='black')
    
    # Legend
    didntLike = mlines.Line2D([], [], c="black", marker='.', markersize=6, label='didntLike')
    smallDoses = mlines.Line2D([], [], c='orange', marker=".", markersize=6, label="smallDoses")
    largeDoses = mlines.Line2D([], [], c='red', marker='.', markersize=6, label="largeDoses")
    
    axs[0][0].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[0][1].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[1][0].legend(handles=[didntLike, smallDoses, largeDoses])

if __name__ == '__main__':
  classifyperson()