import json
import numpy as np
import json
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import al_strategies

def load_data(filename):
    all_data_dict = {}
    with open(filename) as data_file:
        for line in data_file:
            line_object = json.loads(str(line))
            all_data_dict[line_object["tid"]] = line_object["properties"]
    data_file.close()
    return all_data_dict


def get_trajectories_geojson(filename, tids):
    out = ''
    with open(filename) as data_file:
        for line in data_file:
            line_object = json.loads(str(line))
            if line_object["tid"] in tids:
                out += line
    data_file.close()
    return out


def get_random_trajectories(filename):
    bag_size = 20
    all_data_dict = load_data(filename)
    all_tids = all_data_dict.keys()
    randgen = np.random.RandomState()
    all_tids = randgen.permutation(all_tids)
    tids = all_tids[:bag_size]
    out = get_trajectories_geojson(filename, tids)
    return out


def get_file_name(dataset_name):
    if (dataset_name=='fishing'):
        file_name = './data/fishing/fishing_vessels.geojson'

    elif (dataset_name=='geolife'):
        file_name = './data/geolife/geolife.geojson'

    elif (dataset_name=='hurricanes'):
        file_name = './data/hurricanes/hurricanes.geojson'

    return file_name


def get_classifier_model(classifier_name):
    model = None
    if classifier_name=='Logistic Regression':
        model = LogisticRegression()
        #classifier_name='LogisticRegression'
    elif classifier_name=='Random Forest':
        #classifier_name='RandomForestClassifier'
        model = RandomForestClassifier()
    elif classifier_name=='KNN':
        #classifier_name='KNeighborsClassifier'
        model = KNeighborsClassifier()
    elif classifier_name=='Decision Tree':
        #classifier_name='DecisionTreeClassifier'
        model = DecisionTreeClassifier()
    elif classifier_name == 'Ada Boost':
        #classifier_name = 'AdaBoostClassifier'
        model = AdaBoostClassifier()
    elif classifier_name == 'Gaussian Naive Bayes':
        #classifier_name = 'GaussianNB'
        model = GaussianNB()

    return model


def get_al_strategy(strategy_name, model):
    active_s = None
    t = 1
    if strategy_name == 'Random Sampling':
        active_s = al_strategies.RandomStrategy(seed=t)
    elif strategy_name == 'Query-by-committee':
        active_s = al_strategies.QBCStrategy(classifier=model, classifier_args=model.get_params())
    elif strategy_name == 'Uncertain Sampling':
        active_s = al_strategies.UncStrategy(seed=t)

    return active_s


def add_label_to_geojson(json_out, dict_tid_label):
    new_json_out = ''
    lines = json_out.split('\n')
    # print json_out
    total = 0
    for line in lines:
        # print 'line'
        # print line
        if len(line) > 1:
            data_to_parse = json.loads(line)
            tid = data_to_parse['tid']
            data_to_parse['label'] = dict_tid_label[tid]
            #print data_to_parse
            final_line = json.dumps(data_to_parse)+'\n'
            new_json_out += final_line
            total += 1
            #print 'final line'
            #print final_line

    #print new_json_out
    #print total
    return new_json_out