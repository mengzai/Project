#!/usr/bin/python
#-*- coding:utf-8 -*-
# coding:utf-8
import sys
reload(sys)
sys.setdefaultencoding("utf8")
import time
import json                                                                     #导入json模块
import urllib                                                                   #导入urllib模块
from urllib2 import Request, urlopen, URLError, HTTPError
import argparse
import multiprocessing
from time import sleep
from xml.dom.minidom import parse
import xml.dom.minidom
import xml.etree.ElementTree as Etree
from decimal import *
import math
import jieba
import types
parser = argparse.ArgumentParser()
parser.add_argument('--lng')
parser.add_argument('--lat')
parser.add_argument('--ip')
map_type = {u"金融":0,u"酒店":1,u"购物":2,u"休闲娱乐":3}
map_province = {}
def baidu_location(lat,lng):
    result = ''
    getresult = 1
    url = 'http://api.map.baidu.com/place/v2/search?query=酒店$银行$购物中心$休闲娱乐&scope=2&output=xml&location=' + lat + ',' + lng + '&radius=1000&page_size=20&filter=sort_name:distance|sort_rule:1&ak=Y9I8wcnrDG3db4Bg6pyorPVkrHTyLD7z'
    try:
        resultPage = urlopen(url)
    except HTTPError as e:
        print('The server couldn\'t fulfill the request.')
        print('Error code: ', e.code)
        getresult = 0
    except URLError as e:
        print('We failed to reach a server.')
        print('Reason: ', e.reason)
        getresult = 0
    except Exception, e:
        print 'translate error.'
        print e
        getresult = 0
    if getresult:
        resultJason = resultPage.read().decode('utf-8')                #取得翻译的结果，翻译的结果是json格式
        # DOMTree = xml.dom.minidom.parse(resultPage.read())
        # collection = DOMTree.documentElement
        notify_data_tree = Etree.fromstring(resultJason)
        status = notify_data_tree.find("status").text
        if status == "0":
            index_type_num = [0 for i in range (0,len(map_type))]
            hotel_price = 0
            total_num = notify_data_tree.find("total").text
            # print total_num,lat,lng
            for result_index in notify_data_tree.findall("results/result"):
                name = result_index.find('detail_info/tag')
                if (name != None):
                    index_type = name.text.split(';')[0]
                    if map_type.has_key(index_type):
                        # print map_type[index_type],index_type
                        index_type_num[map_type[index_type]] += 1
                        if map_type[index_type] == 1:
                            index_price = result_index.find('detail_info/price')
                            try:
                                index_price_float = float(index_price.text)
                            except Exception, e:
                                print 'float error.'
                                print e
                                continue
                            if index_price_float:
                                hotel_price += index_price_float
                                # print index_price_float
            # print hotel_price,hotel_price / float(index_type_num[map_type[u"金融"]])
            index_content = total_num + ','
            for ii in range(len(index_type_num)):
                index_content += str(index_type_num[ii]) + ','
            if float(index_type_num[map_type[u"酒店"]]):
                index_content += str(hotel_price / float(index_type_num[map_type[u"酒店"]]))
            else:
                index_content += '0.0'
            # print index_content
            result = index_content
            # all_result = notify_data_tree.find("result")[0].children()

            # result.append(notify_data_tree.find("result/location/lat").text)
            # result.append(notify_data_tree.find("result/location/lng").text)
            # result.append(notify_data_tree.find("result/level").text)
    return result
def gaode_location(lat,lng):
    result = ''
    getresult = 1
    url = 'http://restapi.amap.com/v3/geocode/regeo?key=389880a06e3f893ea46036f030c94700&location=' + lng + ',' + lat
    try:
        resultPage = urlopen(url)
    except HTTPError as e:
        print('The server couldn\'t fulfill the request.')
        print('Error code: ', e.code)
        getresult = 0
    except URLError as e:
        print('We failed to reach a server.')
        print('Reason: ', e.reason)
        getresult = 0
    except Exception, e:
        print 'translate error.'
        print e
        getresult = 0
    if getresult:
        resultJason = resultPage.read().decode('utf-8')                #取得翻译的结果，翻译的结果是json格式
        js = None
        try:
            js = json.loads(resultJason)                           #将json格式的结果转换成Python的字典结构
        except Exception, e:
            print 'loads Json error.'
            print e

        key = u"regeocode"
        # print url
        # print js
        if key in js:
            if not (len(js["regeocode"]) == 0):
                # print js["regeocode"]["addressComponent"]["province"].decode('utf8')
                # print js["regeocode"]["addressComponent"]["district"],js["regeocode"]["addressComponent"]["district"] == '[]',type(js["regeocode"]["addressComponent"]["district"]) == list
                if type(js["regeocode"]["addressComponent"]["district"]) == list:
                    district = '-1'
                else:
                    district = js["regeocode"]["addressComponent"]["district"].decode('utf8')
                if type(js["regeocode"]["addressComponent"]["city"]) == list:
                    city = '-1'
                else:
                    city = js["regeocode"]["addressComponent"]["city"].decode('utf8')
                if type(js["regeocode"]["addressComponent"]["province"]) == list:
                    province = '-1'
                else:
                    province = js["regeocode"]["addressComponent"]["province"].decode('utf8')
                result += province + ',' + city + ','  + district
                # result.append(js["regeocode"][0]["addressComponent"])
            # print js["geocodes"][0]["location"]
    return result
def gaode_poi(lat,lng):
    result = ''
    getresult = 1
    url = 'http://restapi.amap.com/v3/place/around?key=389880a06e3f893ea46036f030c94700&output=json&extensions=all&radius=1000&types=酒店&location=' + lng + ',' + lat
    try:
        resultPage = urlopen(url)
    except HTTPError as e:
        print('The server couldn\'t fulfill the request.')
        print('Error code: ', e.code)
        getresult = 0
    except URLError as e:
        print('We failed to reach a server.')
        print('Reason: ', e.reason)
        getresult = 0
    except Exception, e:
        print 'translate error.'
        print e
        getresult = 0
    if getresult:
        resultJason = resultPage.read().decode('utf-8')                #取得翻译的结果，翻译的结果是json格式
        js = None
        try:
            js = json.loads(resultJason)                           #将json格式的结果转换成Python的字典结构
        except Exception, e:
            print 'loads Json error.'
            print e

        key = u"count"
        print url
        # print js
        rating = 0.0
        ordering = 0.0
        price = 0.0
        for ii in js["pois"]:
            # print ii
            # print ii["biz_ext"]
            name = ii["biz_ext"].has_key('rating')
            if name:
                rating += float(ii["biz_ext"]["rating"]) if (type(ii["biz_ext"]["rating"]) != list) else 0.0
            name = ii["biz_ext"].has_key('hotel_ordering')
            if (name):
                ordering += float(ii["biz_ext"]["hotel_ordering"]) if (type(ii["biz_ext"]["hotel_ordering"]) != list) else 0.0
            name = ii["biz_ext"].has_key('lowest_price')
            if name:
                price += float(ii["biz_ext"]["lowest_price"]) if (type(ii["biz_ext"]["lowest_price"]) != list) else 0.0
        if len(js["pois"]):
            rating = rating / float(len(js["pois"]))
            ordering = ordering / float(len(js["pois"]))
            price = price / float(len(js["pois"]))
        # print rating,ordering,price
        result += js["count"] + ',' + str(rating) + ',' + str(ordering) + ',' + str(price)
    return result
def gaode_ip(ip):
    result = ''
    getresult = 1
    url = 'http://restapi.amap.com/v3/ip?output=json&key=389880a06e3f893ea46036f030c94700&ip=' + ip
    try:
        resultPage = urlopen(url)
    except HTTPError as e:
        print('The server couldn\'t fulfill the request.')
        print('Error code: ', e.code)
        getresult = 0
    except URLError as e:
        print('We failed to reach a server.')
        print('Reason: ', e.reason)
        getresult = 0
    except Exception, e:
        print 'translate error.'
        print e
        getresult = 0
    if getresult:
        resultJason = resultPage.read().decode('utf-8')                #取得翻译的结果，翻译的结果是json格式
        js = None
        try:
            js = json.loads(resultJason)                           #将json格式的结果转换成Python的字典结构
        except Exception, e:
            print 'loads Json error.'
            print e

        key = u"count"
        # print url
        # print js
        if type(js["rectangle"]) == list:
            result += '0,0'
        else:
            index_ip_block = js["rectangle"].split(';')
            index_midle_lng = str((float(index_ip_block[0].split(',')[0]) + float(index_ip_block[1].split(',')[0])) / float(2))
            index_midle_lat = str((float(index_ip_block[0].split(',')[1]) + float(index_ip_block[1].split(',')[1])) / float(2))
            # print js["rectangle"]
            # print index_midle_lng + ',' + index_midle_lat
            result += index_midle_lng + ',' + index_midle_lat
        # print js["count"]
    return result
def get_location_ip(ip,flag=1):
    # print address
    result = gaode_ip(ip)
    fail_index_num = 0
    while(result == ''):
        result = gaode_ip(ip)
        fail_index_num += 1
        if fail_index_num > 10:
            print 'always failed more than 20:::::'
            break
    return result
def get_location(lat,lng,flag=1):
    # print address
    result = gaode_location(lat,lng)
    fail_index_num = 0
    while(result == ''):
        result = gaode_location(lat,lng)
        fail_index_num += 1
        if fail_index_num > 10:
            print 'always failed more than 20:::::'
            break
    return result
def get_poi(lat,lng,flag=1):
    # print address
    result = gaode_poi(lat,lng)
    fail_index_num = 0
    while(result == ''):
        result = gaode_poi(lat,lng)
        fail_index_num += 1
        if fail_index_num > 10:
            print 'always failed more than 20:::::'
            break
    return result
def read_date_loc(path_order,lat_pos,lng_pos):
    ip_pos = int(args.ip)
    # f = open('../data/sdk_log_all/split/6')
    f = open('../data/sdk_log_all/all_data_0701.csv')
    content = f.readlines()
    f.close()

    print 'begin',len(content)
    content_all = ''
    content_success = ''
    content_failure = ''
    content_bad = ''
    content_notsame = ''
    content_notdetail_success = ''
    content_city = ''
    testcontent = ''
    failed_num = 0
    not_same_num = 0
    notdetail_success = 0
    ii = 0
    city_flag = 0
    for item in content:
        ii += 1
        if ii == 1:
            continue
        items = item.strip('\n').strip('\r')
        # items = items.replace('\t\t','\t')
        all_data = items.split(',')
        if not (ii % 100):
            print str(ii)
        lat = all_data[lat_pos]
        lng = all_data[lng_pos]
        success_flag = 0
        if (lat != '' and lat != '0' and lng != '' and lng != '0'):
            result = get_location(lat,lng)
            if (result != ',,') and (result != ''):
                content_all += all_data[9] + ',' + result + ',0,' + lng + ',' + lat + '\n'
                success_flag = 1
        if not success_flag:
            # print "ip",all_data[ip_pos],ip_pos
            index_lat_lng = get_location_ip(all_data[ip_pos])
            lat = index_lat_lng.split(',')[1]
            lng = index_lat_lng.split(',')[0]
            result = get_location(lat,lng)
            if (result != ',,') and (result != ''):
                content_all += all_data[9] + ',' + result + ',1,' + lng + ',' + lat + '\n'
            else:
                content_all += all_data[9] + ',-1,-1,-1,-1,-1,-1\n'
            
        # print content_success
    # f.close()
    print 'eeeeeeeeeeeeeeeeddddddddddddddddddddd!!!!!!!!!!!!!!!!!!'
    fp = open('../data/sdk_log_all/split/success_loc_ip_' + path_order + '.txt','w')
    fp.write(content_all)
    fp.close()

    # fp = open('../data/sdk_log_all/split/success_' + path_order + '.txt','w')
    # fp.write(content_success)
    # fp.close()

    # fp = open('../data/sdk_log_all/split/fail_' + path_order + '.txt','w')
    # fp.write(content_failure)
    # fp.close()

    print time.strftime("%I:%M:%S")
    sleep(20)
def read_date(path_order,lat_pos,lng_pos):
    f = open('../data/blackgray/poi/' + path_order + '.txt')
    # f = open('../data/sdk_log_all/yirendai_0725.csv')
    content = f.readlines()
    f.close()

    print 'begin',len(content)
    content_all = ''
    content_success = ''
    content_failure = ''
    content_bad = ''
    content_notsame = ''
    content_notdetail_success = ''
    content_city = ''
    testcontent = ''
    failed_num = 0
    not_same_num = 0
    notdetail_success = 0
    ii = 0
    city_flag = 0
    for item in content:
        ii += 1
        if ii == 1:
            continue
        print ii
        items = item.strip('\n').strip('\r')
        # items = items.replace('\t\t','\t')
        all_data = items.split(',')
        if not (ii % 100):
            print str(ii)
        lat = all_data[lat_pos]
        lng = all_data[lng_pos]
        if (lat != '' and lat != '0' and lng != '' and lng != '0'):
            result = get_poi(lat,lng)
            if (result == ''):
                failed_num += 1
                print 'failed:',lat,lng
                # content_failure += all_data[9] +  '\n'
                content_all += all_data[0] +  ',-1,-1,-1,-1\n'
            else:
                # content_success += all_data[9] + ',' + result + '\n'
                content_all += all_data[0] + ',' + result + '\n'
        else:
            # content_failure += all_data[9] + '\n'
            content_all += all_data[0] + ',-1,-1,-1,-1\n'
            
        # print content_success
    # f.close()
    print 'eeeeeeeeeeeeeeeeddddddddddddddddddddd!!!!!!!!!!!!!!!!!!'
    fp = open('../data/blackgray/poi/success_jiudian_' + path_order + '.txt','w')
    fp.write(content_all)
    fp.close()

    # fp = open('../data/sdk_log_all/split/success_' + path_order + '.txt','w')
    # fp.write(content_success)
    # fp.close()

    # fp = open('../data/sdk_log_all/split/fail_' + path_order + '.txt','w')
    # fp.write(content_failure)
    # fp.close()

    print time.strftime("%I:%M:%S")
    sleep(20)
    # print "failed num:",failed_num
if __name__ == '__main__':
    print time.strftime("%I:%M:%S")
    args = parser.parse_args()
    lat = args.lat
    lng = args.lng
    split_number = 7
    # pool = multiprocessing.Pool(processes=split_number)    # set the processes max number 3
    # for i in (7,0):
    # # for i in range(1,split_number + 1):
    #     result = pool.apply_async(read_date, (str(i),int(lat),int(lng)))
    # pool.close()
    # pool.join()
    # if result.successful():
    #     print 'successful'
    read_date('all_bad_final',int(lat),int(lng))









