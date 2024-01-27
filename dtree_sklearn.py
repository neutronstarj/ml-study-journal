from sklearn import tree
import pandas as pd 
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import pydotplus
import numpy as np
from six import StringIO

if __name__=='__main__':
    with open('lenses.txt','r')as fr:
        lenses=[inst.strip().split('\t')for inst in fr.readlines()]
    lenses_target = []
    for each in lenses:
        lenses_target.append(each[-1])

    lenses_label = ['age','prescript','astigmatic','tearrate']
    lenses_list=[]
    lenses_dict ={}
    for each_label in lenses_label:
        for each in lenses:
            lenses_list.append(each[lenses_label.index(each_label)])
        lenses_dict[each_label]=lenses_list
        lenses_list=[]
    print(lenses_dict)
    lenses_pd = pd.DataFrame(lenses_dict)
#label encoder: transform the string type to asceding numbers
    print(lenses_pd)
    le= LabelEncoder()
    for col in lenses_pd.columns:
        lenses_pd[col]=le.fit_transform(lenses_pd[col])
    clf = tree.DecisionTreeClassifier(max_depth=4)
    clf= clf.fit(lenses_pd.values.tolist(),lenses_target)
    dot_data = StringIO()
    tree.export_graphviz(clf,out_file=dot_data,feature_names=lenses_pd.keys(),class_names=clf.classes_,
                         filled=True,rounded=True,
                         special_characters=True)
    
    graph= pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("tree.pdf")