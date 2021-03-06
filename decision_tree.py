"""
@author: Shrutarv Awasthi
shrutarv.awasthi@gmail.com
"""

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
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
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

def base_features(test_features):
    # pymfe
    '''
    mfe = MFE(features=["leaves","nodes","tree_shape", "tree_depth"])
    mfe.fit(train_features, train_labels)
    ft = mfe.extract()
    base = np.transpose(np.repeat(np.reshape(ft[1],(len(ft[1]),1)),len(test_features), axis=1))
    '''
    base = test_features
    return base
    
def get_features(train_features, train_labels, test_features):
    var = []
    dist_edge = []
    meta_features = np.empty(shape=(len(test_features),5))
    bf = test_features
    # Calling function NN_1 gives the absolute distance to nearest neighbour. 
    dist = NN_1(train_features,train_labels,test_features)
    # Calling CART functions gives the depth of the leaf which makes the
    # prediction, number of examples in that leaf and the value of these examples.
    depth, num_example, sample = CART(train_features,train_labels, test_features)
    for i in range(len(sample)):
        var.append(np.var(sample[i]))
    # Calling MARS gives the distance to nearest edge
    for j in range(len(test_features)):
        dist_edge.append(MARS(train_features,train_labels, test_features[i]))
    # All the features are reformatted and returned as a numpy array 
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
    meta_features = np.concatenate((depth, num_example, dist, var, dist_edge,bf), axis=1)
    #dic = {"Depth":depth, "number of samples":num_example, "variance":var, "dist to nearest edge":dist_edge, "distance to nearest neighbour":dist}
    #dic.update(bf)
    return meta_features

'''
landmarker
    1 nearest neighbour:
    Inputs: train and test data with labels
    Output : absolute distance to nearest neighbour
'''
def NN_1(train_features,train_labels,test_features):
    knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    knn.fit(train_features,train_labels)
    dist, ind = knn.kneighbors(test_features)
    #zip_iterator = zip(ind.tolist(),dist.tolist())
    #dictionary = dict(zip_iterator)
    return dist

'''
landmarker
    Multivariate adaptive regression splines (MARS):
    Inputs: train and test data with labels
    Output : distance to nearest edge
'''
    
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
    
'''
landmarker
    Classification And Regression Trees:
    Inputs: X_train : train data
            y_train : train labels
            X_test : test data
    Output : depth of the leaf which makes the prediction,
            number of examples in that leaf, the value of these examples.
'''
def CART(X_train, y_train, X_test):
    #estimator = DecisionTreeClassifier(max_leaf_nodes=20, random_state=0)
    estimator = DecisionTreeRegressor(max_leaf_nodes=20, random_state=0)
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
    # Support vector regression
    svr_rbf = SVR(kernel='rbf', C=100, gamma='auto', degree=3, epsilon=.1, coef0=1)
    svr_rbf.fit(train_features,train_labels)
    svr_pred = svr_rbf.predict(test_features)
    return svr_pred

def PPR(train_features,train_labels,test_features):
    # Projection Pursuit regression
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
    # Gradient Boosting
    reg = ensemble.GradientBoostingRegressor()
    reg.fit(train_features,train_labels)
    gb_predict = reg.predict(test_features)
    return gb_predict

'''
Landmarker
    This can also be used to extract meta features
'''
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

#################################################################################
######################################### MAIN ##################################
#################################################################################
if __name__ == '__main__':
        
    # Load the data. The below algorithm assumes that labels column should be the last one
    data_orig = pd.read_csv("S:/Job/Time Series analysis/Task3/Concrete_Data.csv")
    # Implement boot strapping. The argument while calling bootstrap method decides the number
    # of random bootstrap samples. It takes into account the location
    data_orig = data_orig.values
    data = data_orig[:int(0.9*len(data_orig)),:]
    bootstrapped_samples = bootstrap(data, number=100)
    '''
        # Random Noise
        noise = np.random.normal(0,1,(1030,9))
        data = data + noise
    '''
        
    print("number of bootstrapped samples",len(bootstrapped_samples))
    # Training
    dt = []         #stores the decision trees for each bootstrap
    # We iterate over each bootstrap samples. For each sample we extract meta
    # features, assign labels based on the best base learner. Then create a 
    # decision tree for predicting the best base learner.
    for index in range(len(bootstrapped_samples)):
        print("bootstrap training sample"+ str(index))
        data = bootstrapped_samples[index]
        feature = data[:,0:(data.shape[1]-1)]
        labels = data[:,data.shape[1]-1]
        train_features, test_features, train_labels, test_labels = train_test_split(feature, labels, test_size = 0.25, random_state = 42)
        n_folds = 5
        max_depth = 15
        min_size = 10
        dep = 0
        #scores, dep = evaluate_algorithm(data, decision_tree, n_folds, dep, max_depth, min_size)
        '''
        labels assigned as per
        predictor                               index
        ppr (projection pursuit regression):      0
        svr (support vector regression):          1
        rf (random forest):                       2
        gb (gradient boosting):                   3
        '''
        # Below code till end of the inner for loop is used to get the 
        # labels for each bootstrapped sample
        label = []
        # Random Forest
        rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
        rf.fit(train_features, train_labels)
        # SVR
        svr_rbf = SVR(kernel='rbf', C=100, gamma='auto', degree=3, epsilon=.1)
        svr_rbf.fit(train_features,train_labels)
        # PPR
        estimator = ProjectionPursuitRegressor()
        estimator.fit(train_features,train_labels)
        # GB
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
            # We use the predictions from rf, gb, ppr and svr to calculate the 
            # distance between the true values and prediction for each test sample.
            # The label coorresponding to the predictor whose distance is least is 
            # chosen as the label for that test sample.
            dist_ppr = np.linalg.norm(ppr_pred-test_labels[i].reshape(1,-1))
            dist_svr = np.linalg.norm(svr_pred-test_labels[i].reshape(1,-1))
            dist_rf = np.linalg.norm(rf_predict-test_labels[i].reshape(1,-1))
            dist_gb = np.linalg.norm(gb_pred-test_labels[i].reshape(1,-1))
            label.append(np.argmin([dist_ppr,dist_svr,dist_rf,dist_gb]))
         
        label = np.asarray(label, dtype='float64')
        label = label.reshape(len(label),1)
        # Extract the meta features 
        meta_features = get_features(train_features, train_labels, test_features)  
        # create a decision tree to select the best algorithm out of ppr, svr, gb and rf
        # use meta features and label to create decision tree
        clf = DecisionTreeClassifier()
        dt.append(clf.fit(meta_features,label))
     
       
    # Testing
    prediction = 0.0
    predicted_values = []
    pred_per_dt = []
    error = []
    predictions = []
    pred_pr_tree = []
    test_set = data_orig[int(0.9*len(data_orig)):,:]
    test_features = test_set[:,0:(test_set.shape[1]-1)]
    l = test_set[:,test_set.shape[1]-1]
    train_features, test_features, train_labels, test_labels = train_test_split(test_features, l, test_size = 0.25, random_state = 42)
    features = get_features(train_features, train_labels, test_features)
    print("number of decision trees", len(dt)) 
    # Outer loop loops over all the test set features
    for j in range(len(test_features)):
        print(j)
        pred_pr_tree = []
        prediction = 0.0
        # Inner loop selects each bootstrapped decision tree and gives the selected 
        # test sample from outer loop as the input. The predictions tell us which base learner is 
        # the best predictor. 
        # and then averages all the outputs
        for i in range(len(dt)):
            pred = dt[i].predict(features[j].reshape(1,-1))
            pred = np.asarray(pred,dtype='int64')
            count = np.bincount(pred)
            # The best predictor is choosen and used to make predictions for the 
            # selected test sample from outer loop. The predictions are summed, averaged
            # and appended in 'prediction' variable. The prediction labels for every
            # tree are stored in 'predictions' variable.
            predictor_label = np.argmax(count)
            if (predictor_label==0):
                prediction += PPR(train_features, train_labels,test_features[j].reshape(1,-1))  
            elif(predictor_label==1):
                prediction += get_SVR(train_features, train_labels,test_features[j].reshape(1,-1))
            elif(predictor_label==2):
                 prediction += RandomForest(train_features, train_labels,test_features[j].reshape(1,-1))
            elif(predictor_label==3):
                 prediction += GB(train_features, train_labels,test_features[j].reshape(1,-1))
            pred_pr_tree.append(predictor_label)
            #error.append(mean_squared_error(test_labels[0], prediction[0]))   
        predictions.append(pred_pr_tree)
        predicted_values.append(prediction/len(dt))           
       #pred_per_dt.append(sum(prediction)/len(prediction))
    mse = mean_squared_error(test_labels[:20], predicted_values[:20])
    std = np.asarray(predicted_values).std()
    # Base Learners
    # We compare the predictions of metabags we created, with the below base learners.
    pred_ppr = PPR(train_features, train_labels,test_features[:20,:])
    mse_ppr = mean_squared_error(test_labels[:20], pred_ppr[:20])
    pred_svr = get_SVR(train_features, train_labels,test_features)
    mse_svr = mean_squared_error(test_labels[:20], pred_svr[:20])
    pred_rf = RandomForest(train_features, train_labels,test_features)
    mse_rf = mean_squared_error(test_labels[:20], pred_rf[:20])
    pred_gb = GB(train_features, train_labels,test_features)
    mse_gb = mean_squared_error(test_labels[:20], pred_gb[:20])
    print("average error",mse) 
    # predictions variable contains the predicted labels for all the 100 bootstrapped
    # decision trees  for each test sample. mse_ variables are mean square errors.
     
            