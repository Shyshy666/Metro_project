import random
from utils import io as utils_io

PERCOLATION_NUMBER = 20

def split_list(l, n):
    """
    将列表 l 平均分割成 n 个子列表。

    参数:
    l -- 待分割的列表。
    n -- 分割成的子列表数量。

    返回值:
    result -- 分割后的子列表组成的列表。
    
    输出值示例：
    sublists = split_list(example_list, 3)
    [[1, 2, 3], [4, 5], [6, 7]]
    """
    # 计算每个子列表大致应该包含的元素数和剩余的不均分元素数
    k = len(l) // n
    m = len(l) % n
    result = []
    i = 0
    for j in range(n):
        # 对于剩余的不均分元素，逐步添加到子列表中
        if j < m:
            result.append(l[i:i + k + 1])
            i += k + 1
        else:
            # 均分的元素直接按照计算好的大小添加到子列表中
            result.append(l[i:i + k])
            i += k
    return result

def random_percolation(g):
    # 复制输入的图对象，以避免修改原始图
    g_copy = g.copy()
    # 获取图中的所有边
    edges_list = list(g.edges())
    edges_number = g.number_of_edges()
    # 创建一个边索引列表，并对其进行随机打乱
    node_shuffle_index = list(range(edges_number))
    random.shuffle(node_shuffle_index)
    # 将打乱后的边索引分成若干组，每组长度为PERCOLATION_NUMBER
    result = split_list(node_shuffle_index[:20], PERCOLATION_NUMBER)
    # 用于存储每次渗滤操作后剩余图的列表
    after_percolation_graph = []
    for i in result:
        # 根据分组信息，移除图中的边
        removal_edges = [edges_list[v] for v in i]
        g_copy.remove_edges_from(removal_edges)
        # 将移除边后的图对象添加到结果列表中
        after_percolation_graph.append(g_copy.copy())
    return after_percolation_graph


def weight_percolation(g, weight, r, save_method, indexes_to_remove):
    # 复制图结构以避免修改原始图
    g_copy = g.copy()
    
    # 根据边权重从大到小排序，并获取边的索引列表
    edge_index = [i[0] for i in sorted(weight.items(), reverse=True, key=lambda e: e[1])]
    
    # 将边索引列表分割为若干子列表，每个子列表长度为PERCOLATION_NUMBER
    # result = split_list(edge_index[r:r+20], PERCOLATION_NUMBER)
    edge_20 = edge_index[:20]
    if save_method == 'order':
        edge_remove = edge_20[5:20]
    elif save_method == 'speed':
        edge_remove = [edge_20[i] for i in range(len(edge_20)) if i not in indexes_to_remove]
    result = split_list(edge_remove, PERCOLATION_NUMBER)
    
    # 存储渗透后的图结构
    after_percolation_graph = []
    for i in result:
        # 从图中移除当前子列表中的节点
        i = [eval(item) for item in i]
        g_copy.remove_edges_from(i)
        # 将渗透后的图结构添加到列表中
        after_percolation_graph.append(g_copy.copy())
    
    return after_percolation_graph


def origin_weight_percolation(g, weight):
    # 复制图结构以避免修改原始图
    g_copy = g.copy()
    
    # 根据边权重从大到小排序，并获取边的索引列表
    edge_index = [i[0] for i in sorted(weight.items(), reverse=True, key=lambda e: e[1])]
    
    # 将边索引列表分割为若干子列表，每个子列表长度为PERCOLATION_NUMBER
    # result = split_list(edge_index[r:r+20], PERCOLATION_NUMBER)
    result = split_list(edge_index[:20], PERCOLATION_NUMBER)
    
    # 存储渗透后的图结构
    after_percolation_graph = []
    for i in result:
        # 从图中移除当前子列表中的节点
        i = [eval(item) for item in i]
        g_copy.remove_edges_from(i)
        # 将渗透后的图结构添加到列表中
        after_percolation_graph.append(g_copy.copy())
    
    return after_percolation_graph


def order_weight1attack_weight2save_percolation(g, weight1, weight2):
    # 复制图结构以避免修改原始图
    g_copy = g.copy()
    
    # 根据边权重从大到小排序，并获取边的索引列表
    edge_index1 = [i[0] for i in sorted(weight1.items(), reverse=True, key=lambda e: e[1])]
    edge_index2 = [i[0] for i in sorted(weight2.items(), reverse=True, key=lambda e: e[1])]
    
    origin_attack_nodes = edge_index1[:20]
    save_nodes = edge_index2[:5]
    now_attack_nodes = [x for x in origin_attack_nodes if x not in save_nodes]
    now_percolation_number = len(now_attack_nodes)
    # 将边索引列表分割为若干子列表，每个子列表长度为PERCOLATION_NUMBER
    
    result = split_list(now_attack_nodes, now_percolation_number)
    
    # 存储渗透后的图结构
    after_percolation_graph = []
    for i in result:
        # 从图中移除当前子列表中的节点
        i = [eval(item) for item in i]
        g_copy.remove_edges_from(i)
        # 将渗透后的图结构添加到列表中
        after_percolation_graph.append(g_copy.copy())
    
    return after_percolation_graph, now_percolation_number

def speed_weight1attack_weight2save_percolation(g, weight1, weight2, indexes_to_remove):
    # 复制图结构以避免修改原始图
    g_copy = g.copy()
    
    # 根据边权重从大到小排序，并获取边的索引列表
    edge_index1 = [i[0] for i in sorted(weight1.items(), reverse=True, key=lambda e: e[1])]
    edge_index2 = [i[0] for i in sorted(weight2.items(), reverse=True, key=lambda e: e[1])]
    
    # 将边索引列表分割为若干子列表，每个子列表长度为PERCOLATION_NUMBER
    # result = split_list(edge_index[r:r+20], PERCOLATION_NUMBER)
    origin_attack_nodes = edge_index1[:20]
    save_nodes = []
    for i in indexes_to_remove:
        save_nodes.append(edge_index2[i])
    now_attack_nodes = [x for x in origin_attack_nodes if x not in save_nodes]
    now_percolation_number = len(now_attack_nodes)
    result = split_list(now_attack_nodes, now_percolation_number)
    
    # 存储渗透后的图结构
    after_percolation_graph = []
    for i in result:
        # 从图中移除当前子列表中的节点
        i = [eval(item) for item in i]
        g_copy.remove_edges_from(i)
        # 将渗透后的图结构添加到列表中
        after_percolation_graph.append(g_copy.copy())
    
    return after_percolation_graph, now_percolation_number




def partial_origin_weight_percolation(g, weight):
    # PERCOLATION_NUMBER = 50,100条链路分成50份
    # 复制图结构以避免修改原始图
    g_copy = g.copy()
    
    # 根据边权重从大到小排序，并获取边的索引列表
    edge_index = [i[0] for i in sorted(weight.items(), reverse=True, key=lambda e: e[1])]
    
    # 将边索引列表分割为若干子列表，每个子列表长度为PERCOLATION_NUMBER
    # result = split_list(edge_index[r:r+20], PERCOLATION_NUMBER)
    result = split_list(edge_index[:100], 50)
    
    # 存储渗透后的图结构
    after_percolation_graph = []
    for i in result:
        # 从图中移除当前子列表中的节点
        i = [eval(item) for item in i]
        g_copy.remove_edges_from(i)
        # 将渗透后的图结构添加到列表中
        after_percolation_graph.append(g_copy.copy())
    
    return after_percolation_graph