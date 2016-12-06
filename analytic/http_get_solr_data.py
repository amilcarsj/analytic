import urllib2
import json


def get_tids_in_database(database):
    url = "http://206.167.183.217:8983/solr/"+str(database)+"/select?fl=tid&q=tid:*&rows=1000000&wt=json"
    # print url
    get_data = urllib2.urlopen(url).read()

    point_line_object = json.loads(get_data)['response']['docs']
    tids_array = []
    for elem in point_line_object:
        tids_array.append(elem['tid'])
    # print sorted(tids_array)
    return sorted(tids_array)


def create_geometry(array_of_values):
    bidim_array = []
    index = 0
    max_index = len(array_of_values)
    # print max_index
    while index < max_index:
        a = [array_of_values[index], array_of_values[index+1]]
        bidim_array.append(a)
        index += 2

    return bidim_array


def get_geojson_layer(database, tid):

    geojson = {}
    geojson['type'] = 'Feature'
    geojson['tid'] = tid
    geojson['properties'] = {}
    geojson['geometry'] = {}
    geojson['geometry']['type'] = 'LineString'
    geojson['points'] = {}

    url = "http://206.167.183.217:8983/solr/"+str(database)+"/select?q=tid:"+str(tid)+"&rows=1&wt=json"
    # print url
    get_data = urllib2.urlopen(url).read()

    point_line_object = json.loads(get_data)['response']['docs'][0]
    geojson['geometry']['coordinates'] = create_geometry(point_line_object['geometry'])
    # print point_line_object

    properties_keys = point_line_object.keys()
    # print properties_keys

    tf_fields = [str(k)[3:] for k in properties_keys if str(k).startswith('tf_')]
    pf_fields = [str(k)[3:] for k in properties_keys if str(k).startswith('pf_')]

    for tf in tf_fields:
        geojson['properties'][tf] = point_line_object['tf_'+str(tf)]

    for pf in pf_fields:
        geojson['points'][pf] = point_line_object['pf_'+str(pf)]

    # print geojson
    return geojson


def get_all_trajectory_features(database):
    map_tid_feature = {}
    get_data = urllib2.urlopen(
        "http://206.167.183.217:8983/solr/" + database + \
        "/select?&q=tid:*&rows=1000000&wt=json").read()
    all_data_lines = json.loads(get_data)['response']['docs']
    # get keys
    properties_keys = all_data_lines[0].keys()
    tf_fields = [str(k)[3:] for k in properties_keys if str(k).startswith('tf_')]

    for line in all_data_lines:
        # print line
        # data = {}
        # data[line['tid']] = line['tid']

        fields = {}
        for tf in tf_fields:
            fields[tf] = line['tf_'+str(tf)]

        # data[line['tid']] = fields
        # map_tid_feature.append(data)
        map_tid_feature[line['tid']] = fields

    # print map_tid_feature
    return  map_tid_feature


def get_trajectories_geojson(database, tids):
    out = ''
    for tid in tids:
        out += json.dumps(get_geojson_layer(database, tid)) + '\n'
    # print out
    return out

# get_tids_in_database('fishingvessels')
# get_geojson_layer('fishingvessels', 2)
# get_all_trajectory_features('fishingvessels')