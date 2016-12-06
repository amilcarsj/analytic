import json
import numpy as np
import json
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import al_strategies, http_get_solr_data


# def load_line_data(filename):
#     all_data_dict = {}
#     with open(filename) as data_file:
#         for line in data_file:
#             line_object = json.loads(str(line))
#             # all_data_dict[line_object["tid"]] = line_object["properties"]
#             all_data_dict[line_object["tid"]] = line_object
#     data_file.close()
#     return all_data_dict
#
#
# def load_point_data(filename):
#     all_data_dict = {}
#     with open(filename) as data_file:
#         for point in data_file:
#             point_object = json.loads(str(point))
#             # all_data_dict[line_object["tid"]] = line_object["properties"]
#             all_data_dict[point_object["pid"]] = point_object
#     data_file.close()
#     return all_data_dict
#
#
# def get_trajectories_geojson(lines, points, tids):
#     out = ''
#     # with open(trajectory_filename) as trajectory_data_file:
#     #     for traj_line in trajectory_data_file:
#     print tids
#     # for traj_line in lines:
#     for tid in tids:
#         # print traj_line
#             # trajectory_line_object = json.loads(str(traj_line))
#         trajectory_line_object = {}
#         points_json = []
#             # tid = trajectory_line_object["tid"]
#         # tid = traj_line
#             # print 'tid', tid
#             # trajectory_line_object['lines'] = lines[traj_line]
#
#         # if tid in tids:
#             # print tid
#         trajectory_line_object = lines[tid]
#                 # out += line old ate aqui
#                 # print 'old ', trajectory_line_object
#                 # get points new
#                 # with open(point_file_name) as point_data_file:
#                 #     for point_line in point_data_file:
#
#         for point_line in points:
#                         # point_line_object = json.loads(str(point_line))
#                         # point_tid = point_line_object['tid']
#                         # print point_line
#             point_object = points[point_line]
#                 # print 'pid', pid, 'tid', point_object['tid']
#                         # print point_object
#             if point_object['tid'] == tid:
#                 # pid = point_line
#                             # points_json.append(point_line_object)
#                     # print point_object['tid'], pid
#                 points_json.append(points[point_line])
#
#         trajectory_line_object['points'] = points_json
#                 # new_traj_line = json.dumps(trajectory_line_object)
#         new_traj_line = json.dumps(trajectory_line_object)
#                 # print 'new ',new_traj_line
#         out += new_traj_line+'\n'
#
#     # trajectory_data_file.close()
#     # point_data_file.close()
#     print out
#     return out


# def get_random_trajectories(trajectory_filename, point_file_name):
# def get_random_trajectories(lines, points):
def get_random_trajectories(dataset):
    bag_size = 20
    # all_data_dict = load_data(trajectory_filename)
    # all_tids = lines.keys()
    all_tids = http_get_solr_data.get_tids_in_database(dataset)
    # print all_tids
    randgen = np.random.RandomState()
    all_tids = randgen.permutation(all_tids)
    tids = all_tids[:bag_size]
    # out = get_trajectories_geojson(trajectory_filename, point_file_name, tids)
    out = http_get_solr_data.get_trajectories_geojson(dataset, tids)
    return out
    # return tids


def get_file_name(dataset_name):

    if (dataset_name=='fishingvessels'):
        file_name = '../data/fishing_vessels.geojson'
        point_file_name = '../data/fishing_vessels_points.geojson'

    elif (dataset_name=='geolife'):
        file_name = '../data/geolife.geojson'
        point_file_name = '../data/geolife_points.geojson'

    elif (dataset_name=='hurricanes'):
        file_name = '../data/hurricanes.geojson'
        point_file_name = '../data/hurricanes_points.geojson'

    elif (dataset_name=='animals'):
        file_name = '../data/hurricanes.geojson'
        point_file_name = '../data/hurricanes_points.geojson'

    return file_name, point_file_name


def get_classifier_model(classifier_name):
    model = None
    if classifier_name=='Logistic Regression':
        classifier_name='LogisticRegression'
        model = LogisticRegression()
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

    # model = eval((classifier_name))
    return model


def get_al_strategy(strategy_name, model, classifier_name):
    active_s = None
    t = 1
    if strategy_name == 'Random Sampling':
        active_s = al_strategies.RandomStrategy(seed=t)
    elif strategy_name == 'Query-by-committee':
        # active_s = al_strategies.QBCStrategy(classifier=model, classifier_args=model.get_params())
        active_s = al_strategies.QBCStrategy(classifier_name=classifier_name)
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