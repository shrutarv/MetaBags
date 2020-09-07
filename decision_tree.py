# Split a dataset based on an attribute and an attribute value
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.svm import SVR
from pymfe.mfe import MFE
from pyearth import Earth
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from random import randrange
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from skpp import ProjectionPursuitRegressor
from sklearn import datasets, ensemble
from sklearn.metrics import mean_squared_error

def bootstrap(data, number):
    boot = []
    for i in range(number):
        b = resample(data, replace=True, n_samples=int(0.1*len(data)))
        boot.append(b)
    return boot
    
def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

def index(groups, classes):
    # count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))
    ind = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        prob = []
        score = 0.0
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score = (score + p*p)
        prob.append(score*(size / n_instances))
      
# base model can be SVR, PPR or GB
def Impurity(base_model,node, groups):
    # Calculate loss function
    
    return 1
    


"""
# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
    # count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))
    # sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)
    return gini
"""
# Select the best split point for a dataset
def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(dataset[0])-1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
           # gini = gini_index(groups, class_values)
            #if gini < b_score:
               # b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index':b_index, 'value':b_value, 'groups':b_groups}

# Create a terminal node value
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del(node['groups'])
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth+1)
    # process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth+1)

# Build a decision tree
def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root

# Print a decision tree
def print_tree(node, depth=0):
    if isinstance(node, dict):
        print('%s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
        print_tree(node['left'], depth+1)
        print_tree(node['right'], depth+1)
    else:
        print('%s[%s]' % ((depth*' ', node)))

def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

def removearray(L,arr):
    ind = 0
    size = len(L)
    while ind != size and not np.array_equal(L[ind],arr):
        ind += 1
    if ind != size:
        L.pop(ind)
    else:
        raise ValueError('array not found in list.')
    return L

def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds,dep, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set = removearray(train_set, fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted, depth = algorithm(train_set, test_set, *args, dep)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores, depth

def predict(node, row, dep):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            dep += 1
            return predict(node['left'], row, dep)
        else:
            dep += 1
            return node['left'], dep
    else:
        if isinstance(node['right'], dict):
            dep += 1
            return predict(node['right'], row, dep)
        else:
            dep += 1
            return node['right'], dep

# Classification and Regression Tree Algorithm
def decision_tree(train, test, max_depth, min_size, dep):
    tree = build_tree(train, max_depth, min_size)
    predictions = list()
    depths = []
    for row in test:
        prediction, dep = predict(tree, row, dep)
        predictions.append(prediction)
        depths.append(dep)
        dep = 0
    return(predictions, depths)

def base_features(train_features,train_labels):
    # pymfe
    #mfe = MFE()
    mfe = MFE(features=["tree_shape", "tree_depth"])
    mfe.fit(train_features, train_labels)
    ft = mfe.extract()
    #print(ft)
    meta_feature_names = ft[0]
    meta_feature_values = ft[1]
    zip_iterator = zip(meta_feature_names, meta_feature_values)
    dictionary = dict(zip_iterator)
    return dictionary

# landmarker
def NN_1(train_features,train_labels,test_features):
        # 1 NN
    knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    knn.fit(train_features,train_labels)
    dist, ind = knn.kneighbors(test_features)
    #zip_iterator = zip(ind.tolist(),dist.tolist())
    #dictionary = dict(zip_iterator)
    return dist
    
def get_features(train_features, train_labels, test_features, test_labels):
    var = []
    dist_edge = []
    meta_features = np.empty(shape=(len(test_features),5))
    bf = base_features(train_features,train_labels)
    dist = NN_1(train_features,train_labels,test_features)
    depth, num_example, sample = CART(train_features,train_labels, test_features)
    for i in range(len(sample)):
        var.append(np.var(sample[i]))
    for j in range(len(test_features)):
        dist_edge.append(MARS(train_features,train_labels, test_features[i]))
    depth = np.asarray(depth, dtype='float64')
    depth = depth.reshape(len(depth),1)
    num_example = np.asarray(num_example, dtype='float64')
    num_example = num_example.reshape(len(depth),1)
    dist = np.asarray(dist, dtype='float64')
    dist = dist.reshape(len(depth),1)
    var = np.asarray(var, dtype='float64')
    var = var.reshape(len(depth),1)
    dist_edge = np.asarray(dist_edge, dtype='float64')
    dist_edge = dist_edge.reshape(len(depth),1)
    meta_features = np.concatenate((depth, num_example, dist, var, dist_edge), axis=1)
    dist_edge = np.asarray(dist_edge)
    #dic = {"Depth":depth, "number of samples":num_example, "variance":var, "dist to nearest edge":dist_edge, "distance to nearest neighbour":dist}
    #dic.update(bf)
    return meta_features
    
def MARS(train_features,train_labels, test_features):
     # MARS Learning algorthm as landmarker
    model = Earth()
    model.fit(train_features,train_labels)
    #print(model.trace())
    #print(model.summary())
    prediction_mars = model.predict(test_features.reshape(1,-1))
    distance_edge = np.min(np.abs(prediction_mars - train_labels))
    model.score(train_features, train_labels)
    return distance_edge
    
def CART(X_train, y_train, X_test):
    estimator = DecisionTreeClassifier(max_leaf_nodes=20, random_state=0)
    estimator.fit(X_train, y_train)
    
    # The decision estimator has an attribute called tree_  which stores the entire
    # tree structure and allows access to low level attributes. The binary tree
    # tree_ is represented as a number of parallel arrays. The i-th element of each
    # array holds information about the node `i`. Node 0 is the tree's root. NOTE:
    # Some of the arrays only apply to either leaves or split nodes, resp. In this
    # case the values of nodes of the other type are arbitrary!
    #
    # Among those arrays, we have:
    #   - left_child, id of the left child of the node
    #   - right_child, id of the right child of the node
    #   - feature, feature used for splitting the node
    #   - threshold, threshold value at the node
    #
    
    # Using those arrays, we can parse the tree structure:
    
    n_nodes = estimator.tree_.node_count
    children_left = estimator.tree_.children_left
    children_right = estimator.tree_.children_right
    feature = estimator.tree_.feature
    threshold = estimator.tree_.threshold
    
    
    # The tree structure can be traversed to compute various properties such
    # as the depth of each node and whether or not it is a leaf.
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1
    
        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True
    
    
    # First let's retrieve the decision path of each sample. The decision_path
    # method allows to retrieve the node indicator functions. A non zero element of
    # indicator matrix at the position (i, j) indicates that the sample i goes
    # through the node j.
    
    node_indicator = estimator.decision_path(X_test)
    
    # Similarly, we can also have the leaves ids reached by each sample.
    
    leave_id = estimator.apply(X_test)
    
    # Now, it's possible to get the tests that were used to predict a sample or
    # a group of samples. First, let's make it for the sample.
    depth = []
    example_leaf = []
    samples = []
    for sample_id in range(len(leave_id)):
        #sample_id = 2
        node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                            node_indicator.indptr[sample_id + 1]]
        t = X_train
       
        d = 0
        #print('Rules used to predict sample %s: ' % sample_id)
        for node_id in node_index:
            if leave_id[sample_id] == node_id:
                continue
        
            if (X_test[sample_id, feature[node_id]] <= threshold[node_id]):
                threshold_sign = "<="
            else:
                threshold_sign = ">"
        
            d += 1
            if (threshold_sign == "<="):
                index = t[:,feature[node_id]]<=threshold[node_id]
            else:
                index = t[:,feature[node_id]]>=threshold[node_id]
            t = t[index[:],:]
        samples.append(t)
        depth.append(d)   
        example_leaf.append(len(t))
    return depth, example_leaf, samples

def get_SVR(train_features, train_labels,test_features):
    # SVR
    svr_rbf = SVR(kernel='rbf', C=100, gamma='auto', degree=3, epsilon=.1, coef0=1)
    svr_rbf.fit(train_features,train_labels)
    svr_pred = svr_rbf.predict(test_features)
    
    return svr_pred

def PPR(train_features,train_labels,test_features):
    #PPR
    estimator = ProjectionPursuitRegressor()
    estimator.fit(train_features,train_labels)
    ppr_pred = estimator.predict(test_features)
    return ppr_pred

def RandomForest(train_features,train_labels,test_features):
    # Random Forest
    rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
    rf.fit(train_features, train_labels)
    rf_predict = rf.predict(test_features)
    return rf_predict

def GB(train_features, train_labels,test_features):
    #GB
    reg = ensemble.GradientBoostingRegressor()
    reg.fit(train_features,train_labels)
    gb_predict = reg.predict(test_features)
    return gb_predict

def Lasso():
     # Lasso
    parameters = {'alpha': np.concatenate((np.arange(0.1,2,0.1), np.arange(2, 5, 0.5), np.arange(5, 25, 1)))}
    lasso = Lasso()
    gridlasso = GridSearchCV(lasso, parameters, scoring ='r2')
    gridlasso.fit(train_features,train_labels)
    gridlasso.predict(test_features)
    scaler = StandardScaler()
    scaler.fit(train_features)
    sel_ = SelectFromModel(LogisticRegression(C=1, penalty='l2'))
    sel_.fit(scaler.transform(train_features), train_labels)
    sel_.get_support()
    train_score=gridlasso.score(train_features,train_labels)
    
    test_score=gridlasso.score(test_features,test_labels)
    print ("training score:", train_score) 
    print ("test score: ", test_score)
    print("lasso best parameters:", gridlasso.best_params_)

'''
dataset = [[2.771244718,1.784783929,0],
    [1.728571309,1.169761413,0],
    [3.678319846,2.81281357,0],
    [3.961043357,2.61995032,0],
    [2.999208922,2.209014212,0],
    [7.497545867,3.162953546,1],
    [9.00220326,3.339047188,1],
    [7.444542326,0.476683375,1],
    [10.12493903,3.234550982,1],
    [6.642287351,3.319983761,1]]
tree = build_tree(dataset, 1, 1)
print_tree(tree)
'''
# Random Forest
data = pd.read_csv("S:/Job/Time Series analysis/Task3/Concrete_Data_new.csv")
# Implement boot strapping
data = data.values
data = data[:int(0.9*len(data)),:]
bootstrapped_samples = bootstrap(data, number=100)
print("number of bootstrapped samples",len(bootstrapped_samples))

dt = []
for index in range(len(bootstrapped_samples)):
    data = bootstrapped_samples[index]
    feature = data[:,0:7]
    labels = data[:,8]
    train_features, test_features, train_labels, test_labels = train_test_split(feature, labels, test_size = 0.25, random_state = 42)
    n_folds = 5
    max_depth = 15
    min_size = 10
    dep = 0
    #scores, dep = evaluate_algorithm(data, decision_tree, n_folds, dep, max_depth, min_size)
    label = []
    # Random Forest
    rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
    rf.fit(train_features, train_labels)
    # SVR
    svr_rbf = SVR(kernel='rbf', C=100, gamma='auto', degree=3, epsilon=.1)
    svr_rbf.fit(train_features,train_labels)
    #PPR
    estimator = ProjectionPursuitRegressor()
    estimator.fit(train_features,train_labels)
    #GB
    reg = ensemble.GradientBoostingRegressor()
    reg.fit(train_features,train_labels)
    for i in range(len(test_features)):
                
        rf_predict = rf.predict(test_features[i,:].reshape(1,-1))
        mse_rf = mean_squared_error(test_labels[i].reshape(1,-1), rf_predict)
            
        svr_pred = svr_rbf.predict(test_features[i,:].reshape(1,-1))
        mse_svr = mean_squared_error(test_labels[i].reshape(1,-1), svr_pred)
            
        ppr_pred = estimator.predict(test_features[i,:].reshape(1,-1))
        mse_ppr = mean_squared_error(test_labels[i].reshape(1,-1), ppr_pred)
          
        gb_pred = estimator.predict(test_features[i,:].reshape(1,-1))
        mse_gb = mean_squared_error(test_labels[i].reshape(1,-1), gb_pred)
        
        dist_ppr = np.linalg.norm(ppr_pred-test_labels[i].reshape(1,-1))
        dist_svr = np.linalg.norm(svr_pred-test_labels[i].reshape(1,-1))
        dist_rf = np.linalg.norm(rf_predict-test_labels[i].reshape(1,-1))
        dist_gb = np.linalg.norm(gb_pred-test_labels[i].reshape(1,-1))
        label.append(np.argmin([dist_ppr,dist_svr,dist_rf,dist_gb]))
     
    label = np.asarray(label, dtype='float64')
    label = label.reshape(len(label),1)
    meta_features = get_features(train_features, train_labels, test_features, test_labels)  
    clf = DecisionTreeClassifier()
    dt.append(clf.fit(meta_features,label))
    
    #Predict the response for test dataset
    y_pred = clf.predict(meta_features[0].reshape(1,-1))
    
    '''
    # Random Noise
    noise = np.random.normal(0,1,(1030,9))
    data = data + noise
     
    # CART
    clf = tree.DecisionTreeRegressor()
    clf = clf.fit(train_features,train_labels)
    pred = clf.predict(test_features)
    pred,depth = decision_tree(train_features[1:100,:],test_features[1:10,:],15,10,0)
    #depth = clf.get_depth()
    clf.get_n_leaves()
    dec_paths = clf.decision_path(test_features)
    # Main
    meta_features = get_features(train_features,train_labels, test_features, test_labels)
    '''
# Testing
prediction=[]
pred_per_dt = []
error = []
test_set = data[int(0.9*len(data)):,:]
l = test_set[:,8]
train_features, test_features, train_labels, test_labels = train_test_split(test_set, l, test_size = 0.25, random_state = 42)
features = get_features(train_features, train_labels, test_features, test_labels) 
print("number of decision trees", len(dt)) 
for i in range(len(dt)):
    pred = dt[i].predict(features)
    pred = np.asarray(pred,dtype='int64')
    count = np.bincount(pred)
    predictor_label = np.argmax(count)
    if (predictor_label==0):
        prediction = PPR(train_features, train_labels,test_features)  
    elif(predictor_label==1):
        prediction = get_SVR(train_features, train_labels,test_features)
    elif(predictor_label==2):
         prediction = RandomForest(train_features, train_labels,test_features)
    elif(predictor_label==3):
         prediction = GB(train_features, train_labels,test_features)
    error.append(mean_squared_error(test_labels, prediction))      
            
    pred_per_dt.append(sum(prediction)/len(prediction))

mean_error = sum(error)/len(error)
output = sum(pred_per_dt)/len(pred_per_dt)    
print("Predicted value", output, "mean square error",mean_error)   
        
        