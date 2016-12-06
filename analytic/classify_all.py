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
from analytic import trajectory_manager
import http_get_solr_data
import scipy.sparse as ss


def run_classification(dataset, classifier_name, labeled_data):
    t = 1  # seed value
    labeled_data_dict = {}
    classifier_args = ''

    for label in labeled_data:
        labeled_data_dict[label["tid"]] = label["label_value"]

    file_name, point_file_name = trajectory_manager.get_file_name(dataset)

    # all_data_dict = trajectory_manager.load_line_data(file_name)
    all_data_tids = http_get_solr_data.get_tids_in_database(dataset)
    all_data_traj_feats = http_get_solr_data.get_all_trajectory_features(dataset)

    model = trajectory_manager.get_classifier_model(classifier_name)

    data_to_train_array = []
    data_already_labeled_array = []
    provided_labels = []
    tids_labeled = []
    tids_not_labeled = []
    map_tid_to_index = {}

    index = 0
    for tid_key in all_data_tids:
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

        # for feature in all_data_dict[tid_key]['properties']:
        #     data_line.append(all_data_dict[tid_key]['properties'][feature])
        for feature in all_data_traj_feats[tid_key]:
            data_line.append(all_data_traj_feats[tid_key][feature])

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
    #print "tids not labeled", [map_tid_to_index[i] for i in tids_not_labeled]
    #print "tids labeled", [map_tid_to_index[i] for i in tids_labeled]
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
    final_labeled_indices = [map_tid_to_index[i] for i in tids_labeled]
    final_dict = {}
    #print final_indices
    index = 0
    for i in final_indices:
        final_dict[i] = predicted[index]
        index += 1
    index = 0
    for i in final_labeled_indices:
        final_dict[i] = provided_labels[index]
        index += 1

    # out_json = {}
    # out_json.update(predicted_values=final_dict)
    # return out_json
    return final_dict