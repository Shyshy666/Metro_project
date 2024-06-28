import warnings
warnings.filterwarnings("ignore")
import argparse
import random
import time
import networkx as nx
import pandas as pd
from tqdm import tqdm
from utils import io as utils_io
# from Models.percolation import random_percolation, weight_percolation, PERCOLATION_NUMBER
from Models.edge_percolation import random_percolation, weight_percolation, PERCOLATION_NUMBER, origin_weight_percolation, order_weight1attack_weight2save_percolation, speed_weight1attack_weight2save_percolation, partial_origin_weight_percolation
import json

total_od_list = [[],[]]
day = [9]
for i in range(len(day)):
    for itime in range(6,23):
        metrood = utils_io.get_improve_metro_od(day[i], itime)
        total_od_list[i].append(metrood['wij'].sum())

def random_alpha_percolation_repeat(day, is_improve_alpha, level, repeat):
    p = 1  # 抽取OD概率
    start = 6   # - start: time-start-6
    end = 23    # - end: time-end-23(取不到)
    # 以指定格式打开文件，准备写入结果
    fp = open(r"subway-percolation/data/alpha/edge_alpha/random_alpha{}_day{}_level{}_v{}.txt".format(is_improve_alpha, day, level, repeat), "w")
    g = utils_io.get_improve_network()  # 获取上海图
    percolation_graph = random_percolation(g)  # 对上海图进行随机渗滤处理
    for index in range(start, end):
        start_time = time.time()
        # 获取指定索引的模型OD数据
        metrood = utils_io.get_improve_metro_od(day, index)
        total_od = metrood['wij'].sum()  # 计算OD对总数
        data = [0 for i in range(PERCOLATION_NUMBER)]  # 初始化数据列表
        for i, row in metrood.iterrows():
            # 根据抽取OD概率决定是否对当前OD对进行处理
            if random.random() < p:
                for key, graph in enumerate(percolation_graph):
                    o, d = str(int(row['ostation'])), str(int(row['dstation']))
                    # 检查图中是否存在从o到d的路径
                    if o in graph and d in graph and nx.has_path(graph, o, d):
                        # 根据是否改善α值进行不同处理
                        if is_improve_alpha == 1:
                            if o != d:
                                origin_length = int(nx.shortest_path_length(g, o, d, weight='length'))
                                now_length = int(nx.shortest_path_length(graph, o, d, weight='length'))# 如果当前路径长度增加，则记录其增加量
                                if now_length > origin_length:
                                    data[key] +=  ((origin_length / now_length)** level) * row['wij']
                                else:
                                    data[key] += row['wij']
                        elif is_improve_alpha == 0:
                            data[key] += row['wij']
        # 计算并打印α值
        alpha = sum(data) / (total_od * p)
        print(index, data, alpha, time.time() - start_time)
        # 将结果写入文件
        s = "{},{},{}\n".format(index, data, alpha)
        fp.write(s)
    fp.close()  # 关闭文件

def random_alpha_percolation(day, is_improve_alpha, level):

    p = 1  # 抽取OD概率
    start = 6   # - start: time-start-6
    end = 23    # - end: time-end-23(取不到)
    # 以指定格式打开文件，准备写入结果
    fp = open(r"subway-percolation/data/alpha/edge_alpha/random_alpha{}_day{}_level{}.txt".format(is_improve_alpha, day, level), "w")
    g = utils_io.get_improve_network()  # 获取上海图
    percolation_graph = random_percolation(g)  # 对上海图进行随机渗滤处理
    for index in range(start, end):
        start_time = time.time()
        # 获取指定索引的模型OD数据
        metrood = utils_io.get_improve_metro_od(day, index)
        total_od = metrood['wij'].sum()  # 计算OD对总数
        data = [0 for i in range(PERCOLATION_NUMBER)]  # 初始化数据列表
        for i, row in metrood.iterrows():
            # 根据抽取OD概率决定是否对当前OD对进行处理
            if random.random() < p:
                for key, graph in enumerate(percolation_graph):
                    o, d = str(int(row['ostation'])), str(int(row['dstation']))
                    # 检查图中是否存在从o到d的路径
                    if o in graph and d in graph and nx.has_path(graph, o, d):
                        # 根据是否改善α值进行不同处理
                        if is_improve_alpha == 1:
                            if o != d:
                                origin_length = int(nx.shortest_path_length(g, o, d, weight='length'))
                                now_length = int(nx.shortest_path_length(graph, o, d, weight='length'))# 如果当前路径长度增加，则记录其增加量
                                if now_length > origin_length:
                                    data[key] +=  ((origin_length / now_length)** level) * row['wij']
                                else:
                                    data[key] += row['wij']
                        elif is_improve_alpha == 0:
                            data[key] += row['wij']
        # 计算并打印α值
        alpha = sum(data) / (total_od * p)
        print(index, data, alpha, time.time() - start_time)
        # 将结果写入文件
        s = "{},{},{}\n".format(index, data, alpha)
        fp.write(s)
    fp.close()  # 关闭文件


    
def process_alpha_data(alpha_path):
    f_list = []
    alpha_list = []
    with open(alpha_path) as fp:
        while True:
            line = fp.readline()[:-1]  # 读取一行并去除换行符
            if not line:  # 如果读取到文件末尾，则退出循环
                break
            # 提取F列表
            start_index = line.index("[")
            end_index = line.index("]")
            data_str = line[start_index + 1:end_index]
            data_list = data_str.split(",")
            float_list = [float(x) for x in data_list]
            f_list.append(float_list)
            # 提取alpha
            alpha = float(line[line.rfind(",") + 1:])
            alpha_list.append(alpha)
    return f_list, alpha_list


def indexes_to_remove(method, is_improve, day, level, r):
    alpha_path = utils_io.get_edge_alpha_path(method, is_improve, day, level, r)
    f_list, alpha_list = process_alpha_data(alpha_path)
    time_list = [ii for ii in range(6,23)]
    top_five_indexes_day_dict = {}
    for k in range(len(time_list)):
        f_list_final = [tmp/total_od_list[0][k] for tmp in f_list[k]]
        f_list_final.insert(0, 1)
        differences = [f_list_final[i] - f_list_final[i+1] for i in range(len(f_list_final)-1)]
        top_five_indexes = sorted(range(len(differences)), key=lambda i: differences[i], reverse=True)[:5]
        top_five_indexes_day_dict[time_list[k]] = top_five_indexes
    print('top_five_indexes_day_dict',top_five_indexes_day_dict)
    return top_five_indexes_day_dict


def model_indexes_to_remove(is_weekend, day):
    weekend = {'0': 'workday', '1': 'weekend'}
    alpha_path = 'subway-percolation/data/alpha/edge_alpha/model_BCFLOW_alpha1_{}00{}_level5.txt'.format(weekend[str(is_weekend)], day)
    f_list, alpha_list = process_alpha_data(alpha_path)
    time_list = [ii for ii in range(6,23)]
    top_five_indexes_day_dict = {}
    total_od = json.load(open(r'subway-percolation/data/model_total_od_workday.json'))
    for k in range(len(time_list)):
        f_list_final = [tmp/total_od[str(day)][str(time_list[k])] for tmp in f_list[k]]
        f_list_final.insert(0, 1)
        differences = [f_list_final[i] - f_list_final[i+1] for i in range(len(f_list_final)-1)]
        top_five_indexes = sorted(range(len(differences)), key=lambda i: differences[i], reverse=True)[:5]
        top_five_indexes_day_dict[time_list[k]] = top_five_indexes
    print('top_five_indexes_day_dict',top_five_indexes_day_dict)
    return top_five_indexes_day_dict


def cwb_alpha_percolation(day, is_improve_alpha, method, level, r, save_method):
    start = 6
    end = 23
    g = utils_io.get_improve_network()  # 获取上海交通图
    if method == 'BC':
        CWB = utils_io.get_edge_BC_data()  # 获取真实OD数据下的CWB数据
    elif method == 'WBC':
        CWB = utils_io.get_edge_WBC_data()
    elif method == 'ODBC':
        CWB = utils_io.get_edge_ODBC_data(day)
    elif method == 'ODWBC':
        CWB = utils_io.get_edge_ODWBC_data(day)
    elif method == 'BCFLOW':
        CWB = utils_io.get_edge_flow_data(day)
    # CWB = utils_io.get_model_cwb_data(start, end)  # 获取理论模型下的CWB数据
    fp = open(r"subway-percolation/data/alpha/edge_alpha/{}_alpha{}_day{}_level{}_r{}_save{}.txt".format(method, is_improve_alpha, day, level, r, save_method), "w")
    p = 1  # 随机采样概率
    top_five_indexes_day_dict = indexes_to_remove(method, is_improve_alpha, day, level, r)
    for index in range(start, end):
        start_time = time.time()
        data = [0 for _ in range(PERCOLATION_NUMBER)]  # 初始化数据列表
        percolation_graph = weight_percolation(g, CWB["{}".format(index)], r, save_method, top_five_indexes_day_dict[index])  # 计算权重渗透图
        metrood = utils_io.get_improve_metro_od(day, index)
        total_od = metrood['wij'].sum()  # 计算OD对总数
        # metrood = utils_io.get_model_od(index, 1)  # 获取理论模型下的OD数据
        for i, row in metrood.iterrows():
            if random.random() < p:  # 根据概率决定是否处理当前OD对
                for key, graph in enumerate(percolation_graph):
                    o, d = str(int(row['ostation'])), str(int(row['dstation']))
                    if o in graph and d in graph and nx.has_path(graph, o, d):
                        if is_improve_alpha == 1:
                            if o != d:
                                origin_length = int(nx.shortest_path_length(g, o, d, weight='length'))
                                now_length = int(nx.shortest_path_length(graph, o, d, weight='length'))
                                # 如果当前路径长度增加，则记录其增加量
                                if now_length > origin_length:
                                    data[key] +=  ((origin_length / now_length) ** level) * row['wij']
                                else:
                                    data[key] += row['wij']
                        elif is_improve_alpha == 0:  # 计算基础Alpha值
                            data[key] += row['wij']
                # print(key)
        alpha = sum(data) / (total_od * p)  # 计算Alpha值
        print(index, data, alpha, time.time() - start_time)  # 打印进度信息
        output_line = "{},{},{}\n".format(index, data, alpha)  # 准备输出行
        fp.write(output_line)  # 写入输出文件
    fp.close()  # 关闭输出文件


def origin_cwb_alpha_percolation(day, is_improve_alpha, method, level):
    start = 6
    end = 23
    g = utils_io.get_improve_network()  # 获取上海交通图
    if method == 'BC':
        CWB = utils_io.get_edge_BC_data()  # 获取真实OD数据下的CWB数据
    elif method == 'WBC':
        CWB = utils_io.get_edge_WBC_data()
    elif method == 'ODBC':
        CWB = utils_io.get_edge_ODBC_data(day)
    elif method == 'ODWBC':
        CWB = utils_io.get_edge_ODWBC_data(day)
    elif method == 'BCFLOW':
        CWB = utils_io.get_edge_flow_data(day)
    # CWB = utils_io.get_model_cwb_data(start, end)  # 获取理论模型下的CWB数据
    fp = open(r"subway-percolation/data/alpha/edge_alpha/{}_alpha{}_day{}_level{}.txt".format(method, is_improve_alpha, day, level), "w")
    p = 1  # 随机采样概率
    for index in range(start, end):
        start_time = time.time()
        data = [0 for _ in range(PERCOLATION_NUMBER)]  # 初始化数据列表
        percolation_graph = origin_weight_percolation(g, CWB["{}".format(index)])  # 计算权重渗透图
        metrood = utils_io.get_improve_metro_od(day, index)
        total_od = metrood['wij'].sum()  # 计算OD对总数
        # metrood = utils_io.get_model_od(index, 1)  # 获取理论模型下的OD数据
        for i, row in metrood.iterrows():
            if random.random() < p:  # 根据概率决定是否处理当前OD对
                for key, graph in enumerate(percolation_graph):
                    o, d = str(int(row['ostation'])), str(int(row['dstation']))
                    if o in graph and d in graph and nx.has_path(graph, o, d):
                        if is_improve_alpha == 1:
                            if o != d:
                                origin_length = int(nx.shortest_path_length(g, o, d, weight='length'))
                                now_length = int(nx.shortest_path_length(graph, o, d, weight='length'))
                                # 如果当前路径长度增加，则记录其增加量
                                if now_length > origin_length:
                                    data[key] +=  ((origin_length / now_length) ** level) * row['wij']
                                else:
                                    data[key] += row['wij']
                        elif is_improve_alpha == 0:  # 计算基础Alpha值
                            data[key] += row['wij']
                # print(key)
        alpha = sum(data) / (total_od * p)  # 计算Alpha值
        print(index, data, alpha, time.time() - start_time)  # 打印进度信息
        output_line = "{},{},{}\n".format(index, data, alpha)  # 准备输出行
        fp.write(output_line)  # 写入输出文件
    fp.close()  # 关闭输出文件


def cwb_alpha_weight1attack_weight2saveorder_percolation(day, is_improve_alpha, attackmethod, savemethod, level):
    start = 6
    end = 23
    g = utils_io.get_improve_network()  # 获取上海交通图
    if attackmethod == 'BC':
        CWB = utils_io.get_edge_BC_data()  # 获取真实OD数据下的CWB数据
    elif attackmethod == 'WBC':
        CWB = utils_io.get_edge_WBC_data()
    elif attackmethod == 'ODBC':
        CWB = utils_io.get_edge_ODBC_data(day)
    elif attackmethod == 'ODWBC':
        CWB = utils_io.get_edge_ODWBC_data(day)
    elif attackmethod == 'BCFLOW':
        CWB = utils_io.get_edge_flow_data(day)
    if savemethod == 'BC':
        weight2 = utils_io.get_edge_BC_data()  # 获取真实OD数据下的CWB数据
    elif savemethod == 'WBC':
        weight2 = utils_io.get_edge_WBC_data()
    elif savemethod == 'ODBC':
        weight2 = utils_io.get_edge_ODBC_data(day)
    elif savemethod == 'ODWBC':
        weight2 = utils_io.get_edge_ODWBC_data(day)
    elif savemethod == 'BCFLOW':
        weight2 = utils_io.get_edge_flow_data(day)
    # CWB = utils_io.get_model_cwb_data(start, end)  # 获取理论模型下的CWB数据
    fp = open(r"subway-percolation/data/alpha/edge_alpha/order_attack-{}_save-{}_day{}_level{}.txt".format(attackmethod, savemethod, day, level), "w")
    p = 1  # 随机采样概率
    for index in range(start, end):
        start_time = time.time()
        percolation_graph, now_percolation_number = order_weight1attack_weight2save_percolation(g, CWB["{}".format(index)], weight2["{}".format(index)])  # 计算权重渗透图
        data = [0 for _ in range(now_percolation_number)]  # 初始化数据列表
        metrood = utils_io.get_improve_metro_od(day, index)
        total_od = metrood['wij'].sum()  # 计算OD对总数
        # metrood = utils_io.get_model_od(index, 1)  # 获取理论模型下的OD数据
        for i, row in metrood.iterrows():
            if random.random() < p:  # 根据概率决定是否处理当前OD对
                for key, graph in enumerate(percolation_graph):
                    o, d = str(int(row['ostation'])), str(int(row['dstation']))
                    if o in graph and d in graph and nx.has_path(graph, o, d):
                        if is_improve_alpha == 1:
                            if o != d:
                                origin_length = int(nx.shortest_path_length(g, o, d, weight='length'))
                                now_length = int(nx.shortest_path_length(graph, o, d, weight='length'))
                                # 如果当前路径长度增加，则记录其增加量
                                if now_length > origin_length:
                                    data[key] +=  ((origin_length / now_length) ** level) * row['wij']
                                else:
                                    data[key] += row['wij']
                        elif is_improve_alpha == 0:  # 计算基础Alpha值
                            data[key] += row['wij']
        alpha = sum(data) / (total_od * p)  # 计算Alpha值
        print(index, data, alpha, time.time() - start_time)  # 打印进度信息
        output_line = "{},{},{}\n".format(index, data, alpha)  # 准备输出行
        fp.write(output_line)  # 写入输出文件
    fp.close()  # 关闭输出文件


def cwb_alpha_weight1attack_weight2savespeed_percolation(day, is_improve_alpha, attackmethod, savemethod, level):

    start = 6
    end = 23
    g = utils_io.get_improve_network()  # 获取上海交通图
    if attackmethod == 'BC':
        CWB = utils_io.get_edge_BC_data()  # 获取真实OD数据下的CWB数据
    elif attackmethod == 'WBC':
        CWB = utils_io.get_edge_WBC_data()
    elif attackmethod == 'ODBC':
        CWB = utils_io.get_edge_ODBC_data(day)
    elif attackmethod == 'ODWBC':
        CWB = utils_io.get_edge_ODWBC_data(day)
    elif attackmethod == 'BCFLOW':
        CWB = utils_io.get_edge_flow_data(day)
    if savemethod == 'BC':
        weight2 = utils_io.get_edge_BC_data()  # 获取真实OD数据下的CWB数据
    elif savemethod == 'WBC':
        weight2 = utils_io.get_edge_WBC_data()
    elif savemethod == 'ODBC':
        weight2 = utils_io.get_edge_ODBC_data(day)
    elif savemethod == 'ODWBC':
        weight2 = utils_io.get_edge_ODWBC_data(day)
    elif savemethod == 'BCFLOW':
        weight2 = utils_io.get_edge_flow_data(day)
    # CWB = utils_io.get_model_cwb_data(start, end)  # 获取理论模型下的CWB数据
    fp = open(r"subway-percolation/data/alpha/edge_alpha/speed_attack-{}_save-{}_day{}_level{}.txt".format(attackmethod, savemethod, day, level), "w")
    p = 1  # 随机采样概率
    for index in range(start, end):
        start_time = time.time()
        indexes_to_remove_list = indexes_to_remove(savemethod, 1, day, 5, 0)
        percolation_graph, now_percolation_number = speed_weight1attack_weight2save_percolation(g, CWB["{}".format(index)], weight2["{}".format(index)], indexes_to_remove_list[index])  # 计算权重渗透图
        data = [0 for _ in range(now_percolation_number)]  # 初始化数据列表
        metrood = utils_io.get_improve_metro_od(day, index)
        total_od = metrood['wij'].sum()  # 计算OD对总数
        # metrood = utils_io.get_model_od(index, 1)  # 获取理论模型下的OD数据
        for i, row in metrood.iterrows():
            if random.random() < p:  # 根据概率决定是否处理当前OD对
                for key, graph in enumerate(percolation_graph):
                    o, d = str(int(row['ostation'])), str(int(row['dstation']))
                    if o in graph and d in graph and nx.has_path(graph, o, d):
                        if is_improve_alpha == 1:
                            if o != d:
                                origin_length = int(nx.shortest_path_length(g, o, d, weight='length'))
                                now_length = int(nx.shortest_path_length(graph, o, d, weight='length'))
                                # 如果当前路径长度增加，则记录其增加量
                                if now_length > origin_length:
                                    data[key] +=  ((origin_length / now_length) ** level) * row['wij']
                                else:
                                    data[key] += row['wij']
                        elif is_improve_alpha == 0:  # 计算基础Alpha值
                            data[key] += row['wij']
        alpha = sum(data) / (total_od * p)  # 计算Alpha值
        print(index, data, alpha, time.time() - start_time)  # 打印进度信息
        output_line = "{},{},{}\n".format(index, data, alpha)  # 准备输出行
        fp.write(output_line)  # 写入输出文件
    fp.close()  # 关闭输出文件

def model_cwb_alpha_percolation(is_weekend, day, is_improve_alpha, method, level):
    start = 6
    end = 23
    weekend = {'0':'workday', '1':'weekend'}
    g = utils_io.get_improve_network()  # 获取上海交通图
    CWB = utils_io.get_model_bcflow_data(is_weekend, day)
    fp = open(r"subway-percolation/data/alpha/edge_alpha/model_{}_alpha{}_{}00{}_level{}.txt".format(method, is_improve_alpha, weekend[str(is_weekend)], day, level), "w")
    p = 1  # 随机采样概率
    for index in range(start, end):
        start_time = time.time()
        data = [0 for _ in range(PERCOLATION_NUMBER)]  # 初始化数据列表
        percolation_graph = origin_weight_percolation(g, CWB["{}".format(index)])  # 计算权重渗透图
        metrood = utils_io.get_model_improve_metro_od(is_weekend, day, index)
        total_od = metrood['wij'].sum()  # 计算OD对总数
        for i, row in metrood.iterrows():
            if random.random() < p:  # 根据概率决定是否处理当前OD对
                for key, graph in enumerate(percolation_graph):
                    o, d = str(int(row['ostation'])), str(int(row['dstation']))
                    if o in graph and d in graph and nx.has_path(graph, o, d):
                        if is_improve_alpha == 1:
                            if o != d:
                                origin_length = int(nx.shortest_path_length(g, o, d, weight='length'))
                                now_length = int(nx.shortest_path_length(graph, o, d, weight='length'))
                                # 如果当前路径长度增加，则记录其增加量
                                if now_length > origin_length:
                                    data[key] +=  ((origin_length / now_length) ** level) * row['wij']
                                else:
                                    data[key] += row['wij']
                        elif is_improve_alpha == 0:  # 计算基础Alpha值
                            data[key] += row['wij']
                # print(key)
        alpha = sum(data) / (total_od * p)  # 计算Alpha值
        print(index, data, alpha, time.time() - start_time)  # 打印进度信息
        output_line = "{},{},{}\n".format(index, data, alpha)  # 准备输出行
        fp.write(output_line)  # 写入输出文件
    fp.close()  # 关闭输出文件

def partial_origin_cwb_alpha_percolation(day, is_improve_alpha, method, level):
    # PERCOLATION_NUMBER = 50,50条链路分成50份
    start = 6
    end = 23
    g = utils_io.get_improve_network()  # 获取上海交通图
    if method == 'ODWBC':
        CWB = utils_io.get_edge_ODWBC_data(day)
    elif method == 'BCFLOW':
        CWB = utils_io.get_edge_flow_data(day)
    # CWB = utils_io.get_model_cwb_data(start, end)  # 获取理论模型下的CWB数据
    fp = open(r"subway-percolation/data/alpha/edge_alpha/partial_{}_alpha{}_day{}_level{}.txt".format(method, is_improve_alpha, day, level), "w")
    p = 1  # 随机采样概率
    for index in range(start, end):
        start_time = time.time()
        data = [0 for _ in range(50)]  # 初始化数据列表
        percolation_graph = partial_origin_weight_percolation(g, CWB["{}".format(index)])  # 计算权重渗透图
        metrood = utils_io.get_improve_metro_od(day, index)
        total_od = metrood['wij'].sum()  # 计算OD对总数
        # metrood = utils_io.get_model_od(index, 1)  # 获取理论模型下的OD数据
        for i, row in metrood.iterrows():
            if random.random() < p:  # 根据概率决定是否处理当前OD对
                for key, graph in enumerate(percolation_graph):
                    o, d = str(int(row['ostation'])), str(int(row['dstation']))
                    if o in graph and d in graph and nx.has_path(graph, o, d):
                        if is_improve_alpha == 1:
                            if o != d:
                                origin_length = int(nx.shortest_path_length(g, o, d, weight='length'))
                                now_length = int(nx.shortest_path_length(graph, o, d, weight='length'))
                                # 如果当前路径长度增加，则记录其增加量
                                if now_length > origin_length:
                                    data[key] +=  ((origin_length / now_length) ** level) * row['wij']
                                else:
                                    data[key] += row['wij']
                        elif is_improve_alpha == 0:  # 计算基础Alpha值
                            data[key] += row['wij']
                # print(key)
        alpha = sum(data) / (total_od * p)  # 计算Alpha值
        print(index, data, alpha, time.time() - start_time)  # 打印进度信息
        output_line = "{},{},{}\n".format(index, data, alpha)  # 准备输出行
        fp.write(output_line)  # 写入输出文件
    fp.close()  # 关闭输出文件

def cwb_alpha_weight1attack_weight2modelsave_ps1_percolation(day, is_improve_alpha, attackmethod, level):
    # 工作日
    start = 6
    end = 23
    # is_weekend = 0 
    g = utils_io.get_improve_network()  # 获取上海交通图
    if attackmethod == 'BC':
        CWB = utils_io.get_edge_BC_data()  # 获取真实OD数据下的CWB数据
    elif attackmethod == 'WBC':
        CWB = utils_io.get_edge_WBC_data()
    elif attackmethod == 'ODWBC':
        CWB = utils_io.get_edge_ODWBC_data(day)
    elif attackmethod == 'BCFLOW':
        CWB = utils_io.get_edge_flow_data(day)
    weight2 = utils_io.get_model_bcflow_data(0, 0)
    # CWB = utils_io.get_model_cwb_data(start, end)  # 获取理论模型下的CWB数据
    fp = open(r"subway-percolation/data/alpha/edge_alpha/ps1_modelsave-attack{}_day{}_level{}.txt".format(attackmethod, day, level), "w")
    p = 1  # 随机采样概率
    for index in range(start, end):
        start_time = time.time()
        percolation_graph, now_percolation_number = order_weight1attack_weight2save_percolation(g, CWB["{}".format(index)], weight2["{}".format(index)])  # 计算权重渗透图
        data = [0 for _ in range(now_percolation_number)]  # 初始化数据列表
        metrood = utils_io.get_improve_metro_od(day, index)
        total_od = metrood['wij'].sum()  # 计算OD对总数
        # metrood = utils_io.get_model_od(index, 1)  # 获取理论模型下的OD数据
        for i, row in metrood.iterrows():
            if random.random() < p:  # 根据概率决定是否处理当前OD对
                for key, graph in enumerate(percolation_graph):
                    o, d = str(int(row['ostation'])), str(int(row['dstation']))
                    if o in graph and d in graph and nx.has_path(graph, o, d):
                        if is_improve_alpha == 1:
                            if o != d:
                                origin_length = int(nx.shortest_path_length(g, o, d, weight='length'))
                                now_length = int(nx.shortest_path_length(graph, o, d, weight='length'))
                                # 如果当前路径长度增加，则记录其增加量
                                if now_length > origin_length:
                                    data[key] +=  ((origin_length / now_length) ** level) * row['wij']
                                else:
                                    data[key] += row['wij']
                        elif is_improve_alpha == 0:  # 计算基础Alpha值
                            data[key] += row['wij']
        alpha = sum(data) / (total_od * p)  # 计算Alpha值
        print(index, data, alpha, time.time() - start_time)  # 打印进度信息
        output_line = "{},{},{}\n".format(index, data, alpha)  # 准备输出行
        fp.write(output_line)  # 写入输出文件
    fp.close()  # 关闭输出文件
    
def cwb_alpha_weight1attack_modelsave_ps2_percolation(day, is_improve_alpha, attackmethod, level):
    start = 6
    end = 23
    g = utils_io.get_improve_network()  # 获取上海交通图
    if attackmethod == 'BC':
        CWB = utils_io.get_edge_BC_data()  # 获取真实OD数据下的CWB数据
    elif attackmethod == 'WBC':
        CWB = utils_io.get_edge_WBC_data()
    elif attackmethod == 'ODBC':
        CWB = utils_io.get_edge_ODBC_data(day)
    elif attackmethod == 'ODWBC':
        CWB = utils_io.get_edge_ODWBC_data(day)
    elif attackmethod == 'BCFLOW':
        CWB = utils_io.get_edge_flow_data(day)
    weight2 = utils_io.get_model_bcflow_data(0, 0)
    # CWB = utils_io.get_model_cwb_data(start, end)  # 获取理论模型下的CWB数据
    fp = open(r"subway-percolation/data/alpha/edge_alpha/ps2_modelsave-attack{}_day{}_level{}.txt".format(attackmethod, day, level), "w")
    p = 1  # 随机采样概率
    for index in range(start, end):
        start_time = time.time()
        indexes_to_remove_list = model_indexes_to_remove(0, 0)
        percolation_graph, now_percolation_number = speed_weight1attack_weight2save_percolation(g, CWB["{}".format(index)], weight2["{}".format(index)], indexes_to_remove_list[index])  # 计算权重渗透图
        data = [0 for _ in range(now_percolation_number)]  # 初始化数据列表
        metrood = utils_io.get_improve_metro_od(day, index)
        total_od = metrood['wij'].sum()  # 计算OD对总数
        # metrood = utils_io.get_model_od(index, 1)  # 获取理论模型下的OD数据
        for i, row in metrood.iterrows():
            if random.random() < p:  # 根据概率决定是否处理当前OD对
                for key, graph in enumerate(percolation_graph):
                    o, d = str(int(row['ostation'])), str(int(row['dstation']))
                    if o in graph and d in graph and nx.has_path(graph, o, d):
                        if is_improve_alpha == 1:
                            if o != d:
                                origin_length = int(nx.shortest_path_length(g, o, d, weight='length'))
                                now_length = int(nx.shortest_path_length(graph, o, d, weight='length'))
                                # 如果当前路径长度增加，则记录其增加量
                                if now_length > origin_length:
                                    data[key] +=  ((origin_length / now_length) ** level) * row['wij']
                                else:
                                    data[key] += row['wij']
                        elif is_improve_alpha == 0:  # 计算基础Alpha值
                            data[key] += row['wij']
        alpha = sum(data) / (total_od * p)  # 计算Alpha值
        print(index, data, alpha, time.time() - start_time)  # 打印进度信息
        output_line = "{},{},{}\n".format(index, data, alpha)  # 准备输出行
        fp.write(output_line)  # 写入输出文件
    fp.close()  # 关闭输出文件