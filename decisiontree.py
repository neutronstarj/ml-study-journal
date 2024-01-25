from math import log

#age : 0 is youth 1 is middle age 2 is old
#job 0 is no, 1 is yes
#house : 0 is no, 1 is yes
#credit score: 0 is satisfactory 1 is good 2 is great
#category(decision) yes approve, no =not approve

"""
function description
    creatintg the training set
param:
    NA
returns:
    dataset
    labels

"""

def readDataset():
    dataset=[
        [0,0,0,0,'no'],
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
    
    labels = ['not approved','approved']
    return dataset,labels

"""
function description:
    calculating the emprirical entropy 
param:
    dataset
returns
    empE
"""

def calcempE(dataset):
    numentries = len(dataset)
    labelcount= {}
    for i in dataset:
        currentlabel = i[-1] #yes or no
        if currentlabel not in labelcount.keys():
            labelcount[currentlabel]=0 #add the label into the dict 
        labelcount[currentlabel]+=1
    empE = 0.0
    for key in labelcount:
        prob = float(labelcount[key])/numentries
        empE -= prob* log(prob,2)
    return empE

"""
function description 
    split the traning set by features
param
    dataset
    axis : the charcatersitic to split the set
    value 
returns 
    None
"""
def splitDataset(dataset,axis,value):
    retdataset=[]
    for feature in dataset:
        if feature[axis]==value:
            reducedFeat= feature[:axis]
            reducedFeat.extend(feature[axis+1:]) #remove axis feature 
            retdataset.append(reducedFeat)
    return retdataset

"""
description:
    choose the best feature, most information gain

param :
    dataset
returns :
    beatFeatures
"""
def chooseBestFeat2split (dataset):
    numFeatures=len(dataset[0])-1
    baseEntropy = calcempE(dataset)
    bestInfoGain=0.0
    bestFeature = -1
    for i in range (numFeatures):
        featList = [example[i] for example in dataset]
        uniqueVals = set(featList)
        newEntro = 0.0
        for value in uniqueVals:
            subdataset = splitDataset(dataset,i,value)
            prob = len(subdataset)/float(len(dataset))
            newEntro+=prob*calcempE(subdataset)
        infoGain=baseEntropy-newEntro
        print("the %d th features gain is %.3f" %(i,infoGain))
        if(infoGain>bestInfoGain):
            bestInfoGain=infoGain
            bestFeature=i
    return bestFeature

if __name__=='__main__':
    dataset,feature=readDataset()
    print("best feature index :"+ str(chooseBestFeat2split(dataset)))

        
