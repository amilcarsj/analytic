#!/usr/bin/python

from bottle import route, request, response, run, post, get, static_file, hook
from analytic import al_strategies, classify_all
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
    response.body = static_file('geolife.geojson', root='./data/geolife/')
    return response


@get('/data/fishing', method='GET')
def fishing_geojson():
    response.body = static_file('fishing_vessels.geojson', root='./data/fishing/')
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response


@get('/data/hurricanes', method='GET')
def hurricanes_geojson():
    response.body = static_file('hurricanes.geojson', root='./data/hurricanes/')
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response


@route('/al_run', method='POST')
def al_strategy():

    trajectories_to_label = None
    for l in request.body:
        str_l = str(l)
        #print str_l
        data_to_parse = json.loads(str_l)
        print data_to_parse
        trajectories_to_label = al_strategies.run_al_strategy(data_to_parse["strategy"],data_to_parse["dataset"],
                                      data_to_parse["classifier"],data_to_parse["labeled_trajectories"],
                                      int(data_to_parse["time_step"]))
    response.headers['Access-Control-Allow-Origin'] = '*'
    print json.dumps(trajectories_to_label)
    return json.dumps(trajectories_to_label)


@route('/classify_all', method='POST')
def classify():

    trajectories_to_label = None
    for l in request.body:
        str_l = str(l)
        #print str_l
        data_to_parse = json.loads(str_l)
        print data_to_parse
        trajectories_to_label = classify_all.run_classification(data_to_parse["dataset"],
                                      data_to_parse["classifier"],data_to_parse["labeled_trajectories"])
    response.headers['Access-Control-Allow-Origin'] = '*'
    print json.dumps(trajectories_to_label)
    return json.dumps(trajectories_to_label)

run(host='0.0.0.0', port=80, debug=True)
