import pandas as pd
import networkx as nx
import json
import pickle
def time_od_pickle2(day, time):
    """OD数据已统计次数时

    Args:
        day (_type_): 日期
        time (_type_): 时间
    """
    # df = pd.read_csv(r"subway-percolation/data/time_od_final/mertrood_DAY{}_TIME{}.csv".format(day, time))
    df = pd.read_csv(r"subway-percolation/data/model_od/mertrood_workday{}_time_{}.0.csv".format(day, time))
    count_dict = {(row['ostation'], row['dstation']): row['wij'] for _, row in df.iterrows()}
    with open(r'subway-percolation/data/model_od/mertrood_workday{}_time_{}.0.p'.format(day, time), 'wb') as f:
        pickle.dump(count_dict, f)

def get_improve_network():
    return nx.read_graphml(r"subway-percolation/data/network/graph-v1.graphml")

def get_improve_metro_od(day, time):
    return pd.read_csv(r"subway-percolation/data/time_od_final/mertrood_DAY{}_TIME{}.csv".format(day, time), encoding='utf-8')
def get_modelweekend_improve_metro_od(day, time):
    return pd.read_csv(r"subway-percolation/data/model_od/mertrood_weekend00{}_time_{}.0.csv".format(day, time), encoding='utf-8')
def get_modelworkday_improve_metro_od(day, time):
    return pd.read_csv(r"subway-percolation/data/model_od/mertrood_workday00{}_time_{}.0.csv".format(day, time), encoding='utf-8')
def get_model_improve_metro_od(is_weekend, day, time):
    if is_weekend:
        return pd.read_csv(r"subway-percolation/data/model_od/mertrood_weekend00{}_time_{}.0.csv".format(day, time), encoding='utf-8')
    else:
        return pd.read_csv(r"subway-percolation/data/model_od/mertrood_workday00{}_time_{}.0.csv".format(day, time), encoding='utf-8')



def get_improve_all_pairs_short_paths():
    return json.load(open(r"subway-percolation/data/new_G_all_pairs.json"))

def get_improve_flow_dict_data(day, time):
    return pickle.load(open(r'subway-percolation/data/time_od_final/mertrood_DAY{}_TIME{}.p'.format(day, time), "rb"))
def get_modelweekend_improve_flow_dict_data(day, time):
    return pickle.load(open(r'subway-percolation/data/model_od/mertrood_weekend00{}_time_{}.0.p'.format(day, time), "rb"))
def get_modelworkday_improve_flow_dict_data(day, time):
    return pickle.load(open(r'subway-percolation/data/model_od/mertrood_workday00{}_time_{}.0.p'.format(day, time), "rb"))

def get_improve_cwb_data(day):
    return json.load(open(r"subway-percolation/data/centrality/improve_cwb_{}.json".format(day)))
def get_improve_bc_data():
    return json.load(open(r"subway-percolation/data/centrality/improve_BC.json"))
def get_improve_dc_data():
    return json.load(open(r"subway-percolation/data/centrality/improve_DC.json"))
def get_improve_cc_data():
    return json.load(open(r"subway-percolation/data/centrality/improve_CC.json"))

def get_improve_WBC_data(day):
    return json.load(open(r"subway-percolation/data/centrality/new_G_WBC_{}.json".format(day)))
def get_improve_node_flow_data(day):
    return json.load(open(r"subway-percolation/data/centrality/improve_node_flow_{}.json".format(day)))

def get_alpha_path(method, is_improve, day):
    return r'subway-percolation/data/alpha/new_G/{}_new_G_alpha_improve{}_day{}.txt'.format(method, is_improve, day)

def get_origin_alpha_path(method, is_improve, day):
    return r'subway-percolation/data/alpha/origin_G/{}_alpha_improve{}_day{}.txt'.format(method, is_improve, day)

def get_data_node_ij_now_shortest_path_length(o,d):
    with open(r'subway-percolation/data/node_ij_now_shortest_path_length.json', 'r') as f:
        origin_ij_shortest_paths = json.load(f)
    return origin_ij_shortest_paths['{}_{}'.format(int(o),int(d))]



# -----------------------------------------------------------------
def get_edge_BC_data():
    return json.load(open(r"subway-percolation/data/centrality/edge_centrality/edge_BC.json"))

def get_edge_WBC_data():
    return json.load(open(r"subway-percolation/data/centrality/edge_centrality/edge_WBC.json"))

def get_edge_ODBC_data(day):
    return json.load(open(r"subway-percolation/data/centrality/edge_centrality/ODBC_{}.json".format(day)))

def get_edge_ODWBC_data(day):
    return json.load(open(r"subway-percolation/data/centrality/edge_centrality/ODWBC_{}.json".format(day)))

def get_edge_flow_data(day):
    return json.load(open(r"subway-percolation/data/centrality/edge_centrality/flow_{}.json".format(day)))


def get_edge_alpha_path(method, is_improve, day, level, r):
    return r'subway-percolation/data/alpha/edge_alpha/{}_alpha{}_day{}_level{}_r{}.txt'.format(method, is_improve, day, level, r)


# -------------------------------------------------------------------------------
def get_model_bcflow_data(is_weekend, day):
    if is_weekend:
        return json.load(open(r"subway-percolation/data/centrality/edge_centrality/modelflow_weekend_00{}.json".format(day)))
    else:
        return json.load(open(r"subway-percolation/data/centrality/edge_centrality/modelflow_workday_00{}.json".format(day)))
    
