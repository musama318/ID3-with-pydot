import numpy as np 
import pandas as pd
import pydot 
 
def entropy(target_col): 
    val,counts = np.unique(target_col,return_counts = True) 
    ent = sum( (-counts[i]/np.sum(counts)) * np.log2( counts[i]/np.sum(counts) ) for i in range(len(val))) 
    return ent 
 
def infoGain(data,features,target): 
    te = entropy(data[target]) 
    val,counts = np.unique(data[features],return_counts = True) 
    eg = sum((counts[i]/sum(counts)) * entropy(data[data[features] == val[i]][target] ) for i in range(len(val))) 
    InfoGain = te-eg 
    return InfoGain 
def ID3(data,features,target,pnode): 
    if len(np.unique(data[target])) == 1: 
        return np.unique(data[target])[0] 
    elif len(features) == 0: 
        return pnode 
    else: 
        pnode = np.unique(data[target])[np.argmax(np.unique(data[target])[1])] 
        IG = [infoGain(data,f,target) for f in features] 
        index = np.argmax(IG) 
        col = features[index] 
        tree = {col:{}} 
        features = [f for f in features if f!=col] 
        for val in np.unique(data[col]): 
            sub_data = data[data[col]==val].dropna() 
            subtree = ID3(sub_data,features,target,pnode) 
            tree[col][val] = subtree 
        return tree 
 
data = pd.read_csv('tennis.csv') 
testData = data.sample(frac = 0.1) 
data.drop(testData.index,inplace = True) 
print(data) 
target = 'play' 
features = data.columns[data.columns!=target] 
tree = ID3(data,features,target,None) 
print (tree) 
test = testData.to_dict('records')[0] 
print(test,'=>', test['play'])
def draw(parent_name, child_name):
    edge = pydot.Edge(parent_name, child_name)
    graph.add_edge(edge)

def visit(node, parent=None):
    for k,v in node.items():
        if isinstance(v, dict):
            if parent:
                draw(parent, k)
            visit(v, k)
        else:
            draw(parent, k)
            draw(k, k+'_'+v)

graph = pydot.Dot(graph_type='graph')
visit(tree)
graph.write_png('example3_graph.png') 