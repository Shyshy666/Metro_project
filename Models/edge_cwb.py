from utils import io as utils_io
import json
from tqdm import tqdm
import networkx as nx
# 考虑了换乘的最短跳数
def edge_BC():
    start = 6
    end = 23
    g = utils_io.get_improve_network()
    # 初始化每天的加权平均中心性字典，初始值为None
    all_CWB = dict.fromkeys(range(start, end), None)
    for time in tqdm(range(start, end)):
        # 不加权：结构上的边介数中心性（考虑了换乘的最短跳数）
        CWB = nx.edge_betweenness_centrality(g)
        CWB = {str(key): value for key, value in CWB.items()}
        all_CWB[time] = CWB
    with open(r'data\centrality\edge_centrality\edge_BC.json', 'w') as json_file:
        json.dump(all_CWB, json_file)
    return all_CWB
# 考虑了换乘的最短行程
def edge_WBC():
    start = 6
    end = 23
    g = utils_io.get_improve_network()
    # 初始化每天的加权平均中心性字典，初始值为None
    all_CWB = dict.fromkeys(range(start, end), None)
    for time in tqdm(range(start, end)):
        # 不加权：结构上的边介数中心性（考虑了换乘的最短跳数）
        CWB = nx.edge_betweenness_centrality(g, weight='length')
        CWB = {str(key): value for key, value in CWB.items()}
        all_CWB[time] = CWB
    with open(r'data\centrality\edge_centrality\edge_WBC.json', 'w') as json_file:
        json.dump(all_CWB, json_file)
    return all_CWB

def find_adjacent_number(numbers, target_tuple):
    target = target_tuple[0]
    adjacent = target_tuple[1]
    for i in range(len(numbers) - 1):
        if numbers[i] == target:
            if i > 0 and numbers[i - 1] == adjacent:
                return True
            elif i < len(numbers) - 1 and numbers[i + 1] == adjacent:
                return True
    return False
def edge_ODBC(day, all_short_paths):
    start = 6
    end = 23
    g = utils_io.get_improve_network()
    # 初始化每天的加权平均中心性字典，初始值为None
    all_CWB = dict.fromkeys(range(start, end), None)
    for time in tqdm(range(start, end)):
        # 初始化每个节点的加权平均中心性值为0
        CWB = dict.fromkeys(g.edges, 0)
        flow_dict = utils_io.get_improve_flow_dict_data(day, time)
        # 遍历所有节点，计算其加权平均中心性
        for v in g.edges:
            for i in g.nodes:
                for j in g.nodes:
                    if i != j:
                        vs = all_short_paths[i][j]
                        temp = 0
                        vs_length = len(vs)
                        # 统计当前节点v在所有从i到j的路径中的出现次数
                        for vi in vs:
                            # if v in vi:
                            if find_adjacent_number(vi, v):
                                temp += 1
                        # 根据路径出现次数和流量，计算节点v的加权平均中心性
                        if (float(i), float(j)) in flow_dict and (float(j), float(i)) in flow_dict:
                            CWB[v] += temp * (flow_dict[(float(i), float(j))] + flow_dict[(float(j), float(i))]) / vs_length
                        elif (float(j), float(i)) in flow_dict:
                            CWB[v] += temp * flow_dict[(float(j), float(i))] / vs_length
                        elif (float(i), float(j)) in flow_dict:
                            CWB[v] += temp * flow_dict[(float(i), float(j))] / vs_length
                        # CWB[v] = CWB[v] / total_od
        CWB = {str(key): value for key, value in CWB.items()}
        all_CWB[time] = CWB
        with open(r'data\centrality\edge_centrality\ODBC_{}.json'.format(day), 'w') as json_file:
            json.dump(all_CWB, json_file)
    return all_CWB

def edge_ODWBC(day, all_short_paths):
    start = 6
    end = 23
    g = utils_io.get_improve_network()
    # 初始化每天的加权平均中心性字典，初始值为None
    all_CWB = dict.fromkeys(range(start, end), None)
    for time in tqdm(range(start, end)):
        # 初始化每个节点的加权平均中心性值为0
        CWB = dict.fromkeys(g.edges, 0)
        flow_dict = utils_io.get_improve_flow_dict_data(day, time)
        metrood = utils_io.get_improve_metro_od(day, time)
        total_od = metrood['wij'].sum()
        # 遍历所有节点，计算其加权平均中心性
        for v in g.edges:
            for i in g.nodes:
                for j in g.nodes:
                    if i != j:
                        vs = all_short_paths[i][j]
                        temp = 0
                        vs_length = len(vs)
                        # 统计当前节点v在所有从i到j的路径中的出现次数
                        for vi in vs:
                            # if v in vi:
                            if find_adjacent_number(vi, v):
                                temp += 1
                        # 根据路径出现次数和流量，计算节点v的加权平均中心性
                        if (float(i), float(j)) in flow_dict and (float(j), float(i)) in flow_dict:
                            CWB[v] += temp * (flow_dict[(float(i), float(j))] + flow_dict[(float(j), float(i))]) / vs_length
                        elif (float(j), float(i)) in flow_dict:
                            CWB[v] += temp * flow_dict[(float(j), float(i))] / vs_length
                        elif (float(i), float(j)) in flow_dict:
                            CWB[v] += temp * flow_dict[(float(i), float(j))] / vs_length
                        # CWB[v] = CWB[v] / total_od
        CWB = {str(key): value for key, value in CWB.items()}
        all_CWB[time] = CWB
        with open(r'data\centrality\edge_centrality\ODWBC_{}.json'.format(day), 'w') as json_file:
            json.dump(all_CWB, json_file)
    return all_CWB

def edge_flow(day, all_short_paths, all_path_length):
    start = 6
    end = 23
    g = utils_io.get_improve_network()
    # 初始化每天的加权平均中心性字典，初始值为None
    all_CWB = dict.fromkeys(range(start, end), None)
    for time in tqdm(range(start, end)):
        # 初始化每个节点的加权平均中心性值为0
        CWB = dict.fromkeys(g.edges, 0)
        flow_dict = utils_io.get_improve_flow_dict_data(day, time)
        metrood = utils_io.get_improve_metro_od(day, time)
        total_od = metrood['wij'].sum()
        # 遍历所有节点，计算其加权平均中心性
        for v in g.edges:
            for i in g.nodes:
                for j in g.nodes:
                    if i != j:
                        vs = all_short_paths[i][j]
                        temp = 0
                        vs_length = len(vs)
                        # 统计当前节点v在所有从i到j的路径中的出现次数
                        for vi in vs:
                            # if v in vi:
                            if find_adjacent_number(vi, v):
                                temp += 1
                        # 根据路径出现次数和流量，计算节点v的加权平均中心性
                        if (float(i), float(j)) in flow_dict and (float(j), float(i)) in flow_dict:
                            CWB[v] += ((temp * (flow_dict[(float(i), float(j))] + flow_dict[(float(j), float(i))]) / vs_length) / all_path_length['{}_{}'.format(i,j)]) * g.edges[v]['length']
                        elif (float(j), float(i)) in flow_dict:
                            CWB[v] += ((temp * flow_dict[(float(j), float(i))] / vs_length) / all_path_length['{}_{}'.format(i,j)]) * g.edges[v]['length']
                        elif (float(i), float(j)) in flow_dict:
                            CWB[v] += ((temp * flow_dict[(float(i), float(j))] / vs_length) / all_path_length['{}_{}'.format(i,j)]) * g.edges[v]['length']
                        # CWB[v] = CWB[v] / total_od
        CWB = {str(key): value for key, value in CWB.items()}
        all_CWB[time] = CWB
        with open(r'data\centrality\edge_centrality\flow_{}.json'.format(day), 'w') as json_file:
            json.dump(all_CWB, json_file)
    return all_CWB




def edge_weekend_model_flow(day, all_short_paths, all_path_length):
    start = 6
    end = 23
    g = utils_io.get_improve_network()
    # 初始化每天的加权平均中心性字典，初始值为None
    all_CWB = dict.fromkeys(range(start, end), None)
    for time in tqdm(range(start, end)):
        # 初始化每个节点的加权平均中心性值为0
        CWB = dict.fromkeys(g.edges, 0)
        flow_dict = utils_io.get_modelweekend_improve_flow_dict_data(day, time)
        metrood = utils_io.get_modelweekend_improve_metro_od(day, time)
        total_od = metrood['wij'].sum()
        # 遍历所有节点，计算其加权平均中心性
        for v in g.edges:
            for i in g.nodes:
                for j in g.nodes:
                    if i != j:
                        vs = all_short_paths[i][j]
                        temp = 0
                        vs_length = len(vs)
                        # 统计当前节点v在所有从i到j的路径中的出现次数
                        for vi in vs:
                            # if v in vi:
                            if find_adjacent_number(vi, v):
                                temp += 1
                        # 根据路径出现次数和流量，计算节点v的加权平均中心性
                        if (float(i), float(j)) in flow_dict and (float(j), float(i)) in flow_dict:
                            CWB[v] += ((temp * (flow_dict[(float(i), float(j))] + flow_dict[(float(j), float(i))]) / vs_length) / all_path_length['{}_{}'.format(i,j)]) * g.edges[v]['length']
                        elif (float(j), float(i)) in flow_dict:
                            CWB[v] += ((temp * flow_dict[(float(j), float(i))] / vs_length) / all_path_length['{}_{}'.format(i,j)]) * g.edges[v]['length']
                        elif (float(i), float(j)) in flow_dict:
                            CWB[v] += ((temp * flow_dict[(float(i), float(j))] / vs_length) / all_path_length['{}_{}'.format(i,j)]) * g.edges[v]['length']
                        # CWB[v] = CWB[v] / total_od
        CWB = {str(key): value for key, value in CWB.items()}
        all_CWB[time] = CWB
        with open(r'subway-percolation/data/centrality/edge_centrality/modelflow_weekend_00{}.json'.format(day), 'w') as json_file:
            json.dump(all_CWB, json_file)
    return all_CWB

def edge_workday_model_flow(day, all_short_paths, all_path_length):
    start = 6
    end = 23
    g = utils_io.get_improve_network()
    # 初始化每天的加权平均中心性字典，初始值为None
    all_CWB = dict.fromkeys(range(start, end), None)
    for time in tqdm(range(start, end)):
        # 初始化每个节点的加权平均中心性值为0
        CWB = dict.fromkeys(g.edges, 0)
        flow_dict = utils_io.get_modelworkday_improve_flow_dict_data(day, time)
        metrood = utils_io.get_modelworkday_improve_metro_od(day, time)
        total_od = metrood['wij'].sum()
        # 遍历所有节点，计算其加权平均中心性
        for v in g.edges:
            for i in g.nodes:
                for j in g.nodes:
                    if i != j:
                        vs = all_short_paths[i][j]
                        temp = 0
                        vs_length = len(vs)
                        # 统计当前节点v在所有从i到j的路径中的出现次数
                        for vi in vs:
                            # if v in vi:
                            if find_adjacent_number(vi, v):
                                temp += 1
                        # 根据路径出现次数和流量，计算节点v的加权平均中心性
                        if (float(i), float(j)) in flow_dict and (float(j), float(i)) in flow_dict:
                            CWB[v] += ((temp * (flow_dict[(float(i), float(j))] + flow_dict[(float(j), float(i))]) / vs_length) / all_path_length['{}_{}'.format(i,j)]) * g.edges[v]['length']
                        elif (float(j), float(i)) in flow_dict:
                            CWB[v] += ((temp * flow_dict[(float(j), float(i))] / vs_length) / all_path_length['{}_{}'.format(i,j)]) * g.edges[v]['length']
                        elif (float(i), float(j)) in flow_dict:
                            CWB[v] += ((temp * flow_dict[(float(i), float(j))] / vs_length) / all_path_length['{}_{}'.format(i,j)]) * g.edges[v]['length']
                        # CWB[v] = CWB[v] / total_od
        CWB = {str(key): value for key, value in CWB.items()}
        all_CWB[time] = CWB
        with open(r'subway-percolation/data/centrality/edge_centrality/modelflow_workday_00{}.json'.format(day), 'w') as json_file:
            json.dump(all_CWB, json_file)
    return all_CWB