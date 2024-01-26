from math import log 
import operator
import matplotlib.pyplot as plt
#copied from last decision tree file
def calcShannonEnt(dataSet):
    numEntires = len(dataSet)                        
    labelCounts = {}                                
    for featVec in dataSet:                            
        currentLabel = featVec[-1]                    
        if currentLabel not in labelCounts.keys():    
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1                
    shannonEnt = 0.0                                
    for key in labelCounts:                            
        prob = float(labelCounts[key]) / numEntires    
        shannonEnt -= prob * log(prob, 2)            
    return shannonEnt                                
 
def createDataSet():
    dataSet = [[0, 0, 0, 0, 'no'],                        
            [0, 0, 0, 1, 'no'],
            [0, 1, 0, 1, 'yes'],
            [0, 1, 1, 0, 'yes'],
            [0, 0, 0, 0, 'no'],
            [1, 0, 0, 0, 'no'],
            [1, 0, 0, 1, 'no'],
            [1, 1, 1, 1, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [2, 0, 1, 2, 'yes'],
            [2, 0, 1, 1, 'yes'],
            [2, 1, 0, 1, 'yes'],
            [2, 1, 0, 2, 'yes'],
            [2, 0, 0, 0, 'no']]
    labels = ['age', 'job', 'house', 'credit score']     
    return dataSet, labels 

def splitDataSet(dataSet, axis, value):       
    retDataSet = []                                        
    for featVec in dataSet:                            
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]                
            reducedFeatVec.extend(featVec[axis+1:])     
            retDataSet.append(reducedFeatVec)
    return retDataSet 

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1                    
    baseEntropy = calcShannonEnt(dataSet)                 
    bestInfoGain = 0.0                                  
    bestFeature = -1                                    
    for i in range(numFeatures):                         
        
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)                         
        newEntropy = 0.0                                  
        for value in uniqueVals:                        
            subDataSet = splitDataSet(dataSet, i, value)         
            prob = len(subDataSet) / float(len(dataSet))           
            newEntropy += prob * calcShannonEnt(subDataSet)     
        infoGain = baseEntropy - newEntropy                     
        # print("第%d个特征的增益为%.3f" % (i, infoGain))            
        if (infoGain > bestInfoGain):                           
            bestInfoGain = infoGain                            
            bestFeature = i                                    
    return bestFeature 

"""
function description:
    get the most frequent elemnt in the classList
param:
    classList
returns:
    sortedClassCount[0][0]

"""

def majorityCnt (classList):
    classCnt={}
    for vote in classList:
        if vote not in classCnt.keys():classCnt[vote]=0
        classCnt[vote]+=1
    sortedClassCount = sorted(classCnt.item(),key = operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

"""
function description :
    create the decision tree
param:
    dataset 
    label
    featlabel

returns;
    mytree
"""
def createtree (dataset,label,featlabel):
    classList = [example[-1]for example in dataset]# the last element yes or no
    if classList.count(classList[0])==len(classList):
        #if all examples in the dataset have the same class label it returns 
        #that class label as a leaf node
        return classList[0]
    if len(dataset[0])== 1 or len(label)==0:
        #if there are no more features to split on (len(dataset[0]==1)) or no more
        #features left in the label list returns the class label with highest freq 
        return majorityCnt(classList) 
    
    bestfeat = chooseBestFeatureToSplit(dataset)
    bestfeatlabel = label[bestfeat]
    featlabel.append(bestfeatlabel)
    mytree= {bestfeatlabel:{}} 
    del(label[bestfeat]) #delete the best feature we alredy used 
    featvalue=[example[bestfeat]for example in dataset]
    uniquevals = set(featvalue)
    for value in uniquevals:
        sublabel=label[:]
        mytree[bestfeatlabel][value]=createtree(splitDataSet(dataset,bestfeat,value),sublabel,featlabel)

    return mytree

def getNumLeafs(mytree):
    numleafs=0
    firstStr = next(iter(mytree))
    secondDict = mytree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            numleafs += getNumLeafs(secondDict[key])
        else : 
            numleafs+=1
    return numleafs

def getTreeDepth(mytree):
    maxDepth = 0
    firstStr = next(iter(mytree))
    secondDict= mytree[firstStr]
    for key in secondDict.keys():

        if type(secondDict[key]).__name__=='dict':
            thisDepth = 1+getTreeDepth(secondDict[key])
        else:
            thisDepth=1
        if thisDepth>maxDepth:
            maxDepth=thisDepth
    return maxDepth

def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    arrow_args= dict(arrowstyle='<-')
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,xycoords="axes fraction",xytext=centerPt,
                          textcoords="axes fraction",va="center",ha='center',bbox=nodeType,arrowprops=arrow_args )


def plotMidText (cntrPt,parentPt,txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]                                                        
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

def plotTree(myTree, parentPt, nodeTxt):
    decisionNode = dict(boxstyle="sawtooth", fc="0.8")                                        
    leafNode = dict(boxstyle="round4", fc="0.8")                                            #设置叶结点格式
    numLeafs = getNumLeafs(myTree)                                                          #获取决策树叶结点数目，决定了树的宽度
    depth = getTreeDepth(myTree)                                                            #获取决策树层数
    firstStr = next(iter(myTree))                                                            #下个字典                                                 
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)    #中心位置
    plotMidText(cntrPt, parentPt, nodeTxt)                                                    #标注有向边属性值
    plotNode(firstStr, cntrPt, parentPt, decisionNode)                                        #绘制结点
    secondDict = myTree[firstStr]                                                            #下一个字典，也就是继续绘制子结点
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD                                        #y偏移
    for key in secondDict.keys():                               
        if type(secondDict[key]).__name__=='dict':                                            #测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            plotTree(secondDict[key],cntrPt,str(key))                                        #不是叶结点，递归调用继续绘制
        else:                                                                                #如果是叶结点，绘制叶结点，并标注有向边属性值                                             
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD

def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')                                                    #创建fig
    fig.clf()                                                                                #清空fig
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)                                #去掉x、y轴
    plotTree.totalW = float(getNumLeafs(inTree))                                            #获取决策树叶结点数目
    plotTree.totalD = float(getTreeDepth(inTree))                                            #获取决策树层数
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;                                #x偏移
    plotTree(inTree, (0.5,1.0), '')                                                            #绘制决策树
    plt.show()  
if __name__=='__main__':

    dataset,label=createDataSet()
    featlabel=[]
    mytree= createtree(dataset,label,featlabel)
    print(mytree)
    createPlot(mytree)