#filter of bad words, comments
import numpy as np
from functools import reduce
def loadDataSet():
    postingList = [['my','dog','has','flea','problems','help','please'],
                   ['maybe','not','take','him','to','the','park','stupid'],
                   ['my','dalmation','is','so','cute','I','love','him'],
                   ['stop','posting','stupid','worthless','garbage'],
                   ['mr','licks','ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    
    classVec = [0,1,0,1,0,1]#1 means bad word, ordered by the postinglist 
    return postingList,classVec


#vectorization

def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)                                    
    for word in inputSet:                                                
        if word in vocabList:                                            
            returnVec[vocabList.index(word)] = 1
        else: print("the word: %s is not in my Vocabulary!" % word)
    return returnVec                                                    
 

def createVocabList(dataset):
    vocabSet = set([])
    for document in dataset:
        vocabSet = vocabSet|set(document) # find the union
    return list(vocabSet)

#postingList is the original sample
#myVocabList (createVocabList)is the list to find all unique word
#vectorization if the word already appeared then become 1, otherwirse it is 0


#trainMatrix is what the setofwords2vec returns
#trainCatgory is loadDataset returns classvec
#p0Vect is non insultingg conditional probability 
#p1Vect is abuse conditional probability
#pAbusive is the probability the document is abusive
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix) #how many document
    numWords = len(trainMatrix[0]) #how may word in each document 
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num= np.zeros(numWords);p1Num=np.zeros(numWords)
    p0Denom = 0.0;p1Denom =0.0
    for i in range(numTrainDocs):
        if trainCategory[i]==1:   #stat of abusive words, P(w0|1),P(w1|1)
            p1Num += trainMatrix[i]
            p1Denom+=sum(trainMatrix[i])
        else:
            p0Num+=trainMatrix[i] #P(w0|0)
            p0Denom += sum(trainMatrix[i])
    
    p1Vect = p1Num/p1Denom
    p0Vect = p0Num/p0Denom
    return p0Vect,p1Vect,pAbusive

#pab is prior probability
#p0v= P(A|B)= p(is|non-abusive)
#p1v is p(is|abusive) P(b|A)


def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1= reduce(lambda x,y : x*y, vec2Classify,p1Vec)*pClass1
    p0 = reduce(lambda x,y:x*y,vec2Classify*p0Vec)*(1.0-pClass1)
    print('p0 ',p0)
    print('p1 ',p1)
    if p1>p0:
        return True
    else:
        return False

def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList=createVocabList(listOPosts)
    trainMat = []
    for post in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList,post))

    p0V,p1V,pAb = trainNB0(np.array(trainMat),np.array(listClasses))
    testEntry = ['love','my','dalmation']
    thisDoc=np.array(setOfWords2Vec(myVocabList,testEntry))
    if classifyNB(thisDoc,p0V,p1V,pAb):
        print(testEntry,"abusive")
    else:
        print(testEntry,"non abusive")

    testEntry = ['garbage','stupid']
    thisDoc=np.array(setOfWords2Vec(myVocabList,testEntry))
    if classifyNB(thisDoc,p0V,p1V,pAb):
        print(testEntry,"abusive")
    else:
        print(testEntry,"non abusive")




if __name__ =='__main__':
    testingNB()

