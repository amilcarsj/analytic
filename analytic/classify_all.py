import math
import numpy as np
from collections import defaultdict
import json
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import scipy.sparse as ss

def run_classification(dataset, classifier_name, labeled_data):
    t = 1  # seed value
    all_data_dict = {}
    labeled_data_dict = {}

    for label in labeled_data:
        labeled_data_dict[label["tid"]] = label["label_value"]

    file_name = None

    if (dataset == 'fishing'):
        file_name = './data/fishing/fishing_vessels.geojson'

    elif (dataset == 'geolife'):
        file_name = './data/geolife/geolife.geojson'

    elif (dataset == 'hurricanes'):
        file_name = './data/hurricanes/hurricanes.geojson'

    with open(file_name) as data_file:
        for line in data_file:
            line_object = json.loads(str(line))
            all_data_dict[line_object["tid"]] = line_object["properties"]

    model = None
    if classifier_name=='Logistic Regression':
        model = LogisticRegression()
        classifier_name='LogisticRegression'
    elif classifier_name=='Random Forest':
        classifier_name='RandomForestClassifier'
        model = RandomForestClassifier()
    elif classifier_name=='KNN':
        classifier_name='KNeighborsClassifier'
        model = KNeighborsClassifier()
    elif classifier_name=='Decision Tree':
        classifier_name='DecisionTreeClassifier'
        model = DecisionTreeClassifier()
    elif classifier_name == 'Ada Boost':
        classifier_name = 'AdaBoostClassifier'
        model = AdaBoostClassifier()
    elif classifier_name == 'Gaussian Naive Bayes':
        classifier_name = 'GaussianNB'
        model = GaussianNB()

    data_to_train_array = []
    data_already_labeled_array = []
    provided_labels = []
    tids_labeled = []
    tids_not_labeled = []
    map_tid_to_index = {}

    index = 0
    for tid_key in all_data_dict:
        data_line = []
        map_tid_to_index[index] = tid_key
        if tid_key in labeled_data_dict:
            tids_labeled.append(index)
            # for feature in all_data_dict[tid_key]:
            #    data_line.append(all_data_dict[tid_key][feature])
            label = str(labeled_data_dict[tid_key])
            provided_labels.append(label)
            # data_line.append(label)
            # data_already_labeled_array.append(data_line)

        else:
            tids_not_labeled.append(index)

        for feature in all_data_dict[tid_key]:
            data_line.append(all_data_dict[tid_key][feature])

        data_to_train_array.append(data_line)
        index += 1

    possible_labels = np.unique(provided_labels)
    map_label_bin = {}
    numeric_label = 0
    numeric_labels = []

    for l in possible_labels:
        map_label_bin[l] = numeric_label
        numeric_label += 1

    for l in provided_labels:
        numeric_labels.append(map_label_bin[l])

    # print "numeric labels", numeric_labels
    print "tids not labeled", [map_tid_to_index[i] for i in tids_not_labeled]
    print "tids labeled", [map_tid_to_index[i] for i in tids_labeled]
    # print "data to train", data_to_train_array

    data_to_train_array = np.array(data_to_train_array)

    #print 'tids labeled index', tids_labeled
    train_data = data_to_train_array[tids_labeled]
    #print "train data", train_data
    #print "out train data", provided_labels
    model.fit(train_data, provided_labels)
    #print 'tids not labeled index', tids_not_labeled
    test_data = data_to_train_array[tids_not_labeled]
    #print 'test data', test_data
    predicted = model.predict(test_data)
    #print predicted
    final_indices = [map_tid_to_index[i] for i in tids_not_labeled]
    final_dict = {}
    #print final_indices
    index = 0
    for i in final_indices:
        final_dict[i] = predicted[index]
        index += 1

    out_json = {}
    out_json.update(predicted_values=final_dict)
    return out_json
