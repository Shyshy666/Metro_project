import io
import pandas as pd
import networkx as nx
import json
import pickle
import random

def get_node_ij_origin_shortest_path_length():
    G = io.get_shanghai_graph()
    shortest_paths = {}
    for i in G.nodes():
        for j in G.nodes():
            if i != j:
                path_length = nx.shortest_path_length(G, i, j)
                key_str = f"{i}_{j}"
                shortest_paths[key_str] = path_length

    # 将结果保存为 JSON 文件
    with open('node_ij_origin_shortest_path_length.json', 'w') as f:
        json.dump(shortest_paths, f, indent=4)



def get_network_improve():
    """将换乘站分为好几个虚拟站点,并将实际轨道距离作为权重，
    结果保存在data/network/graph-v1.graphml中
    """
    g = nx.read_graphml("network_graph.graphml")
    edge_length = {}
    for (source_node, target_node) in g.edges:
        edge_attribute = g[source_node][target_node].get("length", None)
        edge_length[(source_node, target_node)] = edge_attribute
    
    G = nx.Graph()
    # 读取Excel文件
    df = pd.read_csv("node0428-v2.csv")
    station_id = df['id']
    station_name = df['station_name']
    line = df['line']
    for i in range(len(station_id)):
        node = station_id[i]
        G.add_node(node, station_name=str(station_name[i]), line=int(line[i]))

    grouped_df = df.groupby('line')
    for line_name, group in grouped_df:
        node_id = group['id']
        pairs = list(zip(node_id, node_id[1:]))  # 注意：这会忽略每个线路的最后一个节点，因为它没有后续节点配对
        G.add_edges_from(pairs)

    for u in G.nodes():
        for v in G.nodes():
            if u != v:
                uu = G.nodes[u]['station_name']
                vv = G.nodes[v]['station_name']
                if (u,v) in G.edges():
                    if (uu,vv) in edge_length.keys():
                        G.edges[(u, v)]['length'] = edge_length[(uu,vv)]
                    elif (vv,uu) in edge_length.keys():
                        G.edges[(u, v)]['length'] = edge_length[(vv,uu)]
                    else:
                        print((uu,vv))
                else:
                    if uu == vv:
                        G.add_edge(u, v)
                        G.edges[(u, v)]['length'] = 5000
    return G
def remove_edges_without_attribute(G, attribute_key):
    """
    删除图G中缺少指定属性键的边。

    参数:
    - G: NetworkX图对象。
    - attribute_key: 字符串，要检查的边属性键。
    """
    edges_to_remove = [(u, v) for u, v, data in G.edges(data=True) if attribute_key not in data]
    for edge in edges_to_remove:
        G.remove_edge(*edge)
    print(f"移除了{len(edges_to_remove)}条缺少'{attribute_key}'属性的边。")
def main_get_network_improve():
    G = get_network_improve()
    remove_edges_without_attribute(G, 'length')
    G.add_edge(125,120,length=2626.9868739876315)
    G.add_edge(329,325,length=5420.870956959996)
    nx.write_graphml(G, "graph-v1.graphml")

def transfer_metrood(day, time):
    """将metrood改写成节点id值的形式,注意这个函数在main函数中可运行

    Args:
        day (_type_): 日期
        time (_type_): 时间
    """
    with open(r'subway-percolation/data/station_id.json', "r", encoding="utf-8") as json_file:
        station_id = json.load(json_file)
    df = pd.read_csv(r"subway-percolation/data/time_od_final/od_nono_DAY{}_TIME{}.csv".format(day, time))
    
    new_ostation = []
    new_dstation = []
    new_wij = []
    for k,row in df.iterrows():
        new_ostation.append(random.choice(station_id[row['ostation']]))
        new_dstation.append(random.choice(station_id[row['dstation']]))
        new_wij.append(row['wij'])
    new_df = pd.DataFrame(columns=['new_ostation', 'new_dstation'])
    new_df['ostation'] = new_ostation
    new_df['dstation'] = new_dstation
    new_df['wij'] = new_wij
    new_df.to_csv(r"subway-percolation/data/time_od_final/mertrood_DAY{}_TIME{}.csv".format(day, time), index=False)

def transfer_model_metrood(is_weekend, day, time):
    """将metrood改写成节点id值的形式,注意这个函数在main函数中可运行

    Args:
        day (_type_): 日期
        time (_type_): 时间
    """
    weekend = {'0':'workday', '1':'weekend'}
    with open(r'subway-percolation/data/station_id.json', "r", encoding="utf-8") as json_file:
        station_id = json.load(json_file)
    df = pd.read_csv(r"subway-percolation/data/model_od/{}00{}_time_{}.0.csv".format(weekend[str(is_weekend)], day, time))
    
    new_ostation = []
    new_dstation = []
    new_wij = []
    for k,row in df.iterrows():
        if row['ostation'] in station_id.keys() and row['dstation'] in station_id.keys():
            new_ostation.append(random.choice(station_id[row['ostation']]))
            new_dstation.append(random.choice(station_id[row['dstation']]))
            new_wij.append(row['wij'])
    new_df = pd.DataFrame(columns=['new_ostation', 'new_dstation'])
    new_df['ostation'] = new_ostation
    new_df['dstation'] = new_dstation
    new_df['wij'] = new_wij
    new_df.to_csv(r"subway-percolation/data/model_od/mertrood_{}00{}_time_{}.0.csv".format(weekend[str(is_weekend)], day, time), index=False)

def model_od_data_transfer_first(is_weekend, day):
    weekend = {'0':'workday', '1':'weekend'}
    # 读取CSV文件
    df = pd.read_csv('subway-percolation/data/model_od/{}00{}_withhour.csv'.format(weekend[str(is_weekend)], day), encoding='gbk')  # 替换 'your_file.csv' 为你的文件名
    # 更改列名
    df.columns = ['index','ostation', 'dstation', 'time']

    grouped = df.groupby('time')

    for time_value, group in grouped:
        # 构造每个文件的名称
        filename = f'subway-percolation/data/model_od/{weekend[str(is_weekend)]}00{day}_time_{time_value}.csv'
        group_sizes = group.groupby(['ostation', 'dstation']).size().reset_index(name='wij')
        # 保存每个分组到CSV
        group_sizes.to_csv(filename, index=False)  # index=False表示不保存行索引
        # 打印保存操作，以便确认
        print(f"Saved group with time {time_value} to {filename}")

def time_od_pickle2(is_weekend, day, time):
    """OD数据已统计次数时

    Args:
        day (_type_): 日期
        time (_type_): 时间
    """
    weekend = {'0':'workday', '1':'weekend'}
    # df = pd.read_csv(r"subway-percolation/data/time_od_final/mertrood_DAY{}_TIME{}.csv".format(day, time))
    df = pd.read_csv(r"subway-percolation/data/model_od/mertrood_{}00{}_time_{}.0.csv".format(weekend[str(is_weekend)], day, time))
    count_dict = {(row['ostation'], row['dstation']): row['wij'] for _, row in df.iterrows()}
    with open(r'subway-percolation/data/model_od/mertrood_{}00{}_time_{}.0.p'.format(weekend[str(is_weekend)], day, time), 'wb') as f:
        pickle.dump(count_dict, f)


def main_transfer_metrood():      
    # for day in range(1,10):
    #     for is_weekend in [0,1]:
    #         model_od_data_transfer_first(is_weekend, day)
        
    # for day in range(1,10):  
    #     for time in range(6, 23):
    #         for is_weekend in [0,1]:
    #             transfer_model_metrood(is_weekend, day, time)
    
    for day in range(1,10):
        for is_weekend in [0,1]:
            for time in range(6,23):
                time_od_pickle2(is_weekend, day, time)

main_transfer_metrood()




def inconsistent_number_of_paths(G):
    count = 0
    count1 = 0
    for i in G.nodes():
        for j in G.nodes():
            if i != j and nx.has_path(G, i, j):
                count1 += 1
                shortest_paths1 = nx.shortest_path(G, source=i, target=j)
                shourtest_paths2 = nx.shortest_path(G, source=i, target=j, weight='length')
                if shortest_paths1 != shourtest_paths2:
                    count += 1
    print('不一致的路径数：',count)
    print('总路径数：',count1)





def get_all_pairs():
    g = io.get_improve_network()
    all_pairs = nx.all_pairs_shortest_path(g)
    with open(r'data/new_G_all_pairs.json', 'w') as json_file:
        json.dump(dict(all_pairs), json_file)