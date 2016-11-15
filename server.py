#!/usr/bin/python

from bottle import route, request, response, run, post, get, static_file, hook
from analytic import al_strategies, classify_all, trajectory_manager
import json


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
    out = trajectory_manager.get_random_trajectories('./data/geolife/geolife.geojson')
    response.body = out
    return response


@get('/data/fishing', method='GET')
def fishing_geojson():
    response.headers['Access-Control-Allow-Origin'] = '*'
    # out = trajectory_manager.get_random_trajectories('./data/fishing/fishing_vessels_ck.geojson', './data/fishing/fishing_ck_point_vessels.geojson')
    out = trajectory_manager.get_random_trajectories('./data/fishing/fishing_vessels.geojson','./data/fishing/fishing_vessels_points.geojson')
    response.body = out
    return response


@get('/data/hurricanes', method='GET')
def hurricanes_geojson():
    response.headers['Access-Control-Allow-Origin'] = '*'
    out = trajectory_manager.get_random_trajectories('./data/hurricanes/hurricanes.geojson')
    response.body = out
    return response


@route('/al_run', method='POST')
def al_strategy():
    response.headers['Access-Control-Allow-Origin'] = '*'
    trajectories_to_label = None
    file_name = None
    for l in request.body:
        str_l = str(l)
        #print str_l
        data_to_parse = json.loads(str_l)
        #print data_to_parse
        file_name, point_file_name = trajectory_manager.get_file_name(data_to_parse["dataset"])
        trajectories_to_label = al_strategies.run_al_strategy(data_to_parse["strategy"],data_to_parse["dataset"],
                                      data_to_parse["classifier"],data_to_parse["labeled_trajectories"],
                                      int(data_to_parse["time_step"]))
    #print json.dumps(trajectories_to_label)
    # print trajectories_to_label
    # print file_name
    #return json.dumps(trajectories_to_label)
    out = trajectory_manager.get_trajectories_geojson(file_name, point_file_name, trajectories_to_label)
    # print out
    response.body = out
    return response


@route('/classify_all', method='POST')
def classify():
    response.headers['Access-Control-Allow-Origin'] = '*'
    dict_tid_label = None
    for l in request.body:
        str_l = str(l)
        #print str_l
        data_to_parse = json.loads(str_l)
        # print data_to_parse
        file_name, point_file_name = trajectory_manager.get_file_name(data_to_parse["dataset"])
        dict_tid_label = classify_all.run_classification(data_to_parse["dataset"],
                                      data_to_parse["classifier"],data_to_parse["labeled_trajectories"])


    # print json.dumps(trajectories_to_label)
    # return json.dumps(trajectories_to_label)
    out = trajectory_manager.get_trajectories_geojson(file_name, point_file_name, dict_tid_label.keys())
    out = trajectory_manager.add_label_to_geojson(out, dict_tid_label)
    response.body = out
    return response


run(host='0.0.0.0', port=80, debug=True)
# run(host='127.0.0.1', port=8080, debug=True)
