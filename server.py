#!/usr/bin/python

from bottle import route, request, response, run, post, get, static_file, hook
from analytic import al_strategies, classify_all, trajectory_manager, http_get_solr_data
import json
import urllib2


@route('/')
def send_index():
    response.body = static_file('index.html', './')
    return response

@route('/css')
def send_css():
    response.body = static_file('style.min.css', './dist/css')
    response.content_type = 'text/css'
    return response


@route('/analytic_js')
def send_analytic_js():
    response.body = static_file('analytic.min.js', './dist/js')
    response.content_type = 'text/javascript'
    return response


@get('/images/<image_name>', method='GET')
def images(image_name):
    response.body = static_file(image_name, root='./images')
    return response



@get('/data/geolife', method='GET')
def geolife_geojson():
    response.headers['Access-Control-Allow-Origin'] = '*'
    # geolife_points = trajectory_manager.load_point_data('../data/geolife_points.geojson')
    # geolife_lines = trajectory_manager.load_line_data('../data/geolife.geojson')
    # out = trajectory_manager.get_random_trajectories('../data/geolife.geojson', '../data/geolife_points.geojson')
    # out = trajectory_manager.get_random_trajectories(geolife_lines, geolife_points)
    out = trajectory_manager.get_random_trajectories('geolife')
    response.body = out
    return response


@get('/data/fishingvessels', method='GET')
def fishing_geojson():
    response.headers['Access-Control-Allow-Origin'] = '*'
    # out = trajectory_manager.get_random_trajectories('./data/fishing/fishing_vessels_ck.geojson', './data/fishing/fishing_ck_point_vessels.geojson')
    # fishingvessels_points = trajectory_manager.load_point_data('../data/fishing_vessels_points.geojson')
    # fishingvessels_lines = trajectory_manager.load_line_data('../data/fishing_vessels.geojson')

    # out = trajectory_manager.get_random_trajectories(fishingvessels_lines, fishingvessels_points)
    out = trajectory_manager.get_random_trajectories('fishingvessels')
    response.body = out
    return response


@get('/data/hurricanes', method='GET')
def hurricanes_geojson():
    response.headers['Access-Control-Allow-Origin'] = '*'
    # hurricanes_lines = trajectory_manager.load_line_data('../data/hurricanes.geojson')
    # hurricanes_points = trajectory_manager.load_point_data('../data/hurricanes_points.geojson')
    # out = trajectory_manager.get_random_trajectories('../data/hurricanes.geojson', '../data/hurricanes_points.geojson')
    # out = trajectory_manager.get_random_trajectories(hurricanes_lines,hurricanes_points)
    out = trajectory_manager.get_random_trajectories('hurricanes')
    response.body = out
    return response


@get('/data/animals', method='GET')
def animals_geojson():
    response.headers['Access-Control-Allow-Origin'] = '*'
    # animals_points = trajectory_manager.load_point_data('../data/animals_points.geojson')
    # animals_lines = trajectory_manager.load_line_data('../data/animals.geojson')
    # out = trajectory_manager.get_random_trajectories('../data/animals.geojson', '../data/animals_points.geojson')
    # out = trajectory_manager.get_random_trajectories(animals_lines, animals_points)
    out = trajectory_manager.get_random_trajectories('animals')
    response.body = out
    return response


@route('/al_run', method='POST')
def al_strategy():
    response.headers['Access-Control-Allow-Origin'] = '*'
    trajectories_to_label = None
    file_name = None
    database = ''
    for l in request.body:
        str_l = str(l)
        #print str_l
        data_to_parse = json.loads(str_l)
        #print data_to_parse
        database = data_to_parse["dataset"]

        trajectories_to_be_labeled = al_strategies.run_al_strategy(
                                      data_to_parse["strategy"],
                                      database,
                                      data_to_parse["classifier"],
                                      data_to_parse["labeled_trajectories"],
                                      int(data_to_parse["time_step"]))
    #print json.dumps(trajectories_to_label)
    # print trajectories_to_label
    # print file_name
    #return json.dumps(trajectories_to_label)
    # lines_file_name, point_file_name = trajectory_manager.get_file_name(data_to_parse["dataset"])
    # out = trajectory_manager.get_trajectories_geojson(trajectory_manager.load_line_data(lines_file_name),\
    #                                                   trajectory_manager.load_point_data(point_file_name),\
    #                                            trajectories_to_label)
    out = http_get_solr_data.get_trajectories_geojson(database, trajectories_to_be_labeled)
    # print out
    response.body = out
    return response


# @route('/classify_all', method='POST')
# def classify():
#     response.headers['Access-Control-Allow-Origin'] = '*'
#     dict_tid_label = None
#     database = ''
#     for l in request.body:
#         str_l = str(l)
#         #print str_l
#         data_to_parse = json.loads(str_l)
#         database = data_to_parse["dataset"]
#         # print data_to_parse
#         lines_file_name, point_file_name = trajectory_manager.get_file_name(data_to_parse["dataset"])
#         dict_tid_label = classify_all.run_classification(data_to_parse["dataset"],
#                                       data_to_parse["classifier"],data_to_parse["labeled_trajectories"])
#     # print 'finished classifying all trajs'
#     # print json.dumps(trajectories_to_label)
#     # return json.dumps(trajectories_to_label)
#     # out = trajectory_manager.get_trajectories_geojson(trajectory_manager.load_line_data(lines_file_name), \
#     #                                                   trajectory_manager.load_point_data(point_file_name), \
#     #                                                   dict_tid_label.keys())
#     # out = trajectory_manager.add_label_to_geojson(out, dict_tid_label)
#     out = ''
#
#     # print dict_tid_label
#     for tid in dict_tid_label:
#         data_object = http_get_solr_data.get_geojson_layer(database, tid)
#         data_object['label'] = dict_tid_label[tid]
#         out += json.dumps(data_object) + '\n'
#     # print out
#     # print 'finished parsing all trajs to geojson'
#     response.body = out
#     return response

@route('/classify_all', method='POST')
def classify():
    response.headers['Access-Control-Allow-Origin'] = '*'
    dict_tid_label = None
    database = ''
    for l in request.body:
        str_l = str(l)
        # print str_l
        data_to_parse = json.loads(str_l)
        database = data_to_parse["dataset"]
        # print data_to_parse
        lines_file_name, point_file_name = trajectory_manager.get_file_name(data_to_parse["dataset"])
        dict_tid_label = classify_all.run_classification(data_to_parse["dataset"],
                                                         data_to_parse["classifier"],
                                                         data_to_parse["labeled_trajectories"])
    out = json.dumps(dict_tid_label)

    response.body = out
    return response


@route('/trajectories_geojson', method='POST')
def get_trajectories_geojson():
    response.headers['Access-Control-Allow-Origin'] = '*'
    database = ''
    # print request
    geo_out = ''
    for l in request.body:
        str_l = str(l)
        # print str_l
        data_to_parse = json.loads(str_l)
        database = data_to_parse["dataset"]
        # print data_to_parse
        for tid in data_to_parse['bag']:
            geojson = http_get_solr_data.get_geojson_layer(database, tid)
            geojson['label'] = data_to_parse['bag'][tid]
            geo_out += json.dumps(geojson)+'\n'

    response.body = geo_out
    return response

run(host='0.0.0.0', port=80, debug=True)
# run(host='127.0.0.1', port=8080, debug=True)
