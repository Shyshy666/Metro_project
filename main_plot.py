import warnings
warnings.filterwarnings("ignore")
import networkx as nx
import pandas as pd
from utils import io as utils_io
from Models.edge_percolation import random_percolation, weight_percolation, PERCOLATION_NUMBER
import json
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import ScalarFormatter


global total_od_list
total_od_list = [[],[]]
day = [9,5]
for i in range(len(day)):
    for itime in range(6,23):
        metrood = utils_io.get_improve_metro_od(day[i], itime)
        total_od_list[i].append(metrood['wij'].sum())

global other_total_od_list
other_total_od_list = json.load(open(r"data\other_total_od1.json"))

def process_alpha_data(alpha_path):
    # alpha_path = utils_io.get_edge_alpha_path(method, is_improve, day, level, r)
    f_list = []
    alpha_list = []
    alpha_path = r'{}'.format(alpha_path)
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
def calculate_alpha(alpha_path, day, total_od_list):
    totalod = other_total_od_list[str(day)]
    f_list, alpha_list = process_alpha_data(alpha_path)
    time_list = [ii for ii in range(6,23)]
    alpha_1_list = []
    f_list_final_list = []
    itime = [i for i in range(6,23)]
    for k in range(len(time_list)):
        f_list_final = [tmp/totalod[str(itime[k])] for tmp in f_list[k]]
        f_list_final.insert(0, 1)
        alpha = sum(f_list_final) / len(f_list_final)
        alpha_1_list.append(alpha)
        f_list_final_list.append(f_list_final)
    return alpha_1_list, f_list_final_list
def plot_total_od():
    plt.figure(figsize=(6, 4))
    bar_width = 0.4  # Adjust this value as needed
    xx = [i for i in range(6,23)]
    plt.bar(xx, total_od_list[0], width=bar_width, color='#D69D98', label='Day9')
    plt.bar([x + bar_width for x in xx], total_od_list[1], width=bar_width, color='#6699CC', label='Day5')
    plt.xlabel('Time')
    plt.ylabel('Total Flow')
    plt.xticks(xx)
    plt.legend()
    plt.savefig(r'data\fig\total_od.pdf', dpi=600)
    plt.show()

    
def gini(x):
    total = 0
    x = np.array(x)
    for i, xi in enumerate(x[:-1], 1):
        total += np.sum(np.abs(xi - x[i:]))
    return total / (len(x) ** 2 * np.mean(x))

def centrality_transform(day):
    
    flow = json.load(open(r'data\centrality\edge_centrality\flow_{}.json'.format(day)))
    odwbc = json.load(open(r'data\centrality\edge_centrality\ODWBC_{}.json'.format(day)))
    
    data = odwbc
    
    itime = [i for i in range(6,23)]
    
    if day == 9:
        total_od_day = total_od_list[0]
    elif day == 5:
        total_od_day = total_od_list[1]
    
    total_dict = {}
    for kk in range(0,len(itime)):
        time_dict = {}
        for k,value in data[str(itime[kk])].items():
            time_dict[k] = value / total_od_day[kk]
        total_dict[str(itime[kk])] = time_dict
    

    with open(r'data\centrality\edge_centrality\odwbc_norm_{}.json'.format(day), 'w') as json_file:
        json.dump(total_dict, json_file)
    
    data = flow
    
    itime = [i for i in range(6,23)]
    
    total_dict = {}
    for kk in range(0,len(itime)):
        time_dict = {}
        for k,value in data[str(itime[kk])].items():
            time_dict[k] = value / total_od_day[kk]
        total_dict[str(itime[kk])] = time_dict
    with open(r'data\centrality\edge_centrality\flow_norm_{}.json'.format(day), 'w') as json_file:
        json.dump(total_dict, json_file)     
        

def data_centrality_var():
    hour = [i for i in range(6,23)]
    bc_var, wbc_var = [], []
    data = pd.DataFrame()
    bc = json.load(open(r'data\centrality\edge_centrality\edge_BC.json'))
    wbc = json.load(open(r'data\centrality\edge_centrality\edge_WBC.json'))
    data['time'] = hour

    for k, v in bc.items():
        bc_var.append(np.var(np.array(list(v.values()))))
    for k, v in wbc.items():
        wbc_var.append(np.var(np.array(list(v.values()))))
    data['bc_var'] = bc_var
    data['wbc_var'] = wbc_var
        
    for day in [5,9]:
        flow_var, odwbc_var = [], []
        flow = json.load(open(r'data\centrality\edge_centrality\flow_norm_{}.json'.format(day)))
        odwbc = json.load(open(r'data\centrality\edge_centrality\odwbc_norm_{}.json'.format(day)))
        for k, v in flow.items():
            flow_var.append(np.var(np.array(list(v.values()))))
        for k, v in odwbc.items():
            odwbc_var.append(np.var(np.array(list(v.values()))))
        data['flow_var'+str(day)] = flow_var
        data['odwbc_var'+str(day)] = odwbc_var
    data.to_csv(r'data\plot_data\centrality_var.csv', index=False)

def data_centrality_gini():
    hour = [i for i in range(6,23)]
    bc_var, wbc_var = [], []
    data = pd.DataFrame()
    bc = json.load(open(r'data\centrality\edge_centrality\edge_BC.json'))
    wbc = json.load(open(r'data\centrality\edge_centrality\edge_WBC.json'))
    data['time'] = hour

    for k, v in bc.items():
        bc_var.append(gini(np.array(list(v.values()))))
    for k, v in wbc.items():
        wbc_var.append(gini(np.array(list(v.values()))))
    data['bc_var'] = bc_var
    data['wbc_var'] = wbc_var
        
    for day in [5,9]:
        flow_var, odwbc_var = [], []
        flow = json.load(open(r'data\centrality\edge_centrality\flow_norm_{}.json'.format(day)))
        odwbc = json.load(open(r'data\centrality\edge_centrality\odwbc_norm_{}.json'.format(day)))
        for k, v in flow.items():
            flow_var.append(gini(np.array(list(v.values()))))
        for k, v in odwbc.items():
            odwbc_var.append(gini(np.array(list(v.values()))))
        data['flow_var'+str(day)] = flow_var
        data['odwbc_var'+str(day)] = odwbc_var
    data.to_csv(r'data\plot_data\centrality_gini.csv', index=False)


def plot_centrality_var():
    data = pd.read_csv(r'data\plot_data\centrality_var.csv')
    xx = data['time']
    bc_y = data['bc_var']
    wbc_y = data['wbc_var']
    odwbc_y9 = data['flow_var9']
    bcf_y9 = data['odwbc_var9']
    odwbc_y5 = data['flow_var5']
    bcf_y5 = data['odwbc_var5']
    
    # 设置图形尺寸
    plt.figure(figsize=(8, 6))

    # 第一个子图：Day9数据的BCwoddis Var
    plt.subplot(2, 2, 1)
    plt.plot(xx, bcf_y9, color='#BA3E45', linewidth=2, marker='o', mfc='#BA3E45', mew=2,label='Day9')  # 实线样式，黄色
    plt.xlabel('Time')  # x轴标签
    plt.ylabel(r'$BC_{w,od}^{dis}$ Var')  # y轴标签
    plt.xticks(range(6, 23, 2))
    plt.legend()  # 显示图例
    plt.tight_layout()  # 调整布局以防止标签被截断
    # 将 y 轴坐标刻度显示为科学计数法，并设置指数部分的长度
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, 3))  # 将指数部分的长度限制在3位数以内
    plt.gca().yaxis.set_major_formatter(formatter)

    # 第二个子图：Day5数据的BCwoddis Var
    plt.subplot(2, 2, 2)
    plt.plot(xx, bcf_y5, color='#3A4B6E', linewidth=2, marker='s', mfc='#3A4B6E', mew=2,label='Day5')  # 实线样式，黄色
    plt.xlabel('Time')  # x轴标签
    plt.ylabel(r'$BC_{w,od}^{dis}$ Var')  # y轴标签
    plt.xticks(range(6, 23, 2))
    plt.legend()  # 显示图例
    plt.tight_layout()  # 调整布局以防止标签被截断
    # 将 y 轴坐标刻度显示为科学计数法，并设置指数部分的长度
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, 3))  # 将指数部分的长度限制在3位数以内
    plt.gca().yaxis.set_major_formatter(formatter)

    # 第三个子图：Day9数据的BCwod Var
    plt.subplot(2, 2, 3)
    plt.plot(xx, odwbc_y9, color='#BA3E45', linewidth=2, marker='o', mfc='#BA3E45', mew=2,label='Day9')  # 实线样式，绿色
    plt.xlabel('Time')  # x轴标签
    plt.ylabel('$BC_{w,od}$ Var')  # y轴标签
    plt.xticks(range(6, 23, 2))
    plt.legend()  # 显示图例
    plt.tight_layout()  # 调整布局以防止标签被截断
    # 将 y 轴坐标刻度显示为科学计数法，并设置指数部分的长度
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, 3))  # 将指数部分的长度限制在3位数以内
    plt.gca().yaxis.set_major_formatter(formatter)

    # 第四个子图：Day5数据的BCwod Var
    plt.subplot(2, 2, 4)
    plt.plot(xx, odwbc_y5, color='#3A4B6E', linewidth=2, marker='s', mfc='#3A4B6E', mew=2,label='Day5')  # 实线样式，绿色
    plt.xlabel('Time')  # x轴标签
    plt.ylabel('$BC_{w,od}$ Var')  # y轴标签
    plt.xticks(range(6, 23, 2))
    plt.legend()  # 显示图例
    plt.tight_layout()  # 调整布局以防止标签被截断
    # 将 y 轴坐标刻度显示为科学计数法，并设置指数部分的长度
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, 3))  # 将指数部分的长度限制在3位数以内
    plt.gca().yaxis.set_major_formatter(formatter)

    plt.savefig(r'data\fig\bc_var_subplots.pdf')
    plt.show()
  
def plot_centrality_gini():
    data = pd.read_csv(r'data\plot_data\centrality_gini.csv')
    xx = data['time']
    bc_y = data['bc_var']
    wbc_y = data['wbc_var']
    odwbc_y9 = data['flow_var9']
    bcf_y9 = data['odwbc_var9']
    odwbc_y5 = data['flow_var5']
    bcf_y5 = data['odwbc_var5']
    
    plt.figure(figsize=(4, 4))  # 设置图形尺寸
    plt.plot(xx, bcf_y9, color='#BA3E45', linewidth=2, marker='o', mfc='#BA3E45', mew=2,label='Day9')  # 实线样式，黄色
    plt.plot(xx, bcf_y5, color='#3A4B6E', linewidth=2, marker='s', mfc='#3A4B6E', mew=2,label='Day5')  # 实线样式，黄色
    
    plt.xlabel('Time')  # x轴标签
    plt.ylabel(r'$BC_{w,od}^{dis}$ Var')  # y轴标签
    plt.xticks(range(6, 23, 2))
    plt.legend()  # 显示图例
    plt.tight_layout()  # 调整布局以防止标签被截断
    plt.savefig(r'data\fig\bcflow_var_9_5.pdf')
    plt.show()
    
    plt.figure(figsize=(4, 4))  # 设置图形尺寸
    plt.plot(xx, odwbc_y9, color='#BA3E45', linewidth=2, marker='o', mfc='#BA3E45', mew=2,label='Day9')  # 实线样式，绿色
    plt.plot(xx, odwbc_y5, color='#3A4B6E', linewidth=2, marker='s', mfc='#3A4B6E', mew=2,label='Day5')  # 实线样式，绿色

    plt.xlabel('Time')  # x轴标签
    plt.ylabel('$BC_{w,od}$ Var')  # y轴标签
    plt.xticks(range(6, 23, 2))
    plt.legend()  # 显示图例
    plt.tight_layout()  # 调整布局以防止标签被截断
    plt.savefig(r'data\fig\odwbc_var_9_5.pdf')
    plt.show()

def plot_centrality_3d(day, itime):
    # bc = json.load(open(r'data\centrality\edge_centrality\edge_BC.json'))
    wbc = json.load(open(r'data\centrality\edge_centrality\edge_WBC.json'))
    flow = json.load(open(r'data\centrality\edge_centrality\flow_norm_{}.json'.format(day)))
    odwbc = json.load(open(r'data\centrality\edge_centrality\odwbc_norm_{}.json'.format(day)))

    # 获取每个中心性的值
    wbc_values = list(wbc[str(itime)].values())
    flow_values = list(flow[str(itime)].values())
    odwbc_values = list(odwbc[str(itime)].values())

    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    axes[0].scatter(wbc_values, flow_values)
    axes[0].set_xlabel('wbc Values')
    axes[0].set_ylabel('flow Values')
    axes[0].set_title('BC Values vs BCW Values')


    axes[1].scatter(odwbc_values, flow_values)
    axes[1].set_xlabel('odwbc Values')
    axes[1].set_ylabel('flow Values')
    axes[1].set_title('BC Values vs BCWOD Values')
    
    axes[2].scatter(wbc_values, odwbc_values)
    axes[2].set_xlabel('wbc_values')
    axes[2].set_ylabel('odwbc_values')
    axes[2].set_title('BC Values vs BCWOD Values')

    plt.tight_layout()
    plt.show()

def centrality_rank(day, rank):
    bc = utils_io.get_edge_BC_data()
    wbc = utils_io.get_edge_WBC_data()
    odwbc = utils_io.get_edge_ODWBC_data(9)
    BCflow = utils_io.get_edge_flow_data(9)
    
    data = pd.DataFrame()
    bc_sort = [i[0] for i in sorted(bc[str(6)].items(), reverse=True, key=lambda e: e[1])]
    wbc_sort = [i[0] for i in sorted(wbc[str(6)].items(), reverse=True, key=lambda e: e[1])]
    data['bc'] = bc_sort[:rank]
    data['wbc'] = wbc_sort[:rank]
    for itime in range(6,23):
        odwbc_sort = [i[0] for i in sorted(odwbc[str(itime)].items(), reverse=True, key=lambda e: e[1])]
        data['odwbc'+str(itime)] = odwbc_sort[:rank]
    for itime in range(6,23):
        BCflow_sort = [i[0] for i in sorted(BCflow[str(itime)].items(), reverse=True, key=lambda e: e[1])]
        data['BCflow'+str(itime)] = BCflow_sort[:rank]
    data.to_csv(r'data\plot_data\centrality_rank{}_day{}.csv'.format(rank,day), index=False)
    # 计算重合率
    # 计算重合率
    # 统计每个指标的链路
    
def plot_centrality_overlap_rate(day,rank):
    data = pd.read_csv(r'data\plot_data\centrality_rank{}_day{}.csv'.format(rank,day))
    selected_data = data.iloc[:, -34:-18]
    selected_data = data
    data_dict = selected_data.to_dict(orient='list')
 
    # 计算重合率
    overlap_rates = np.zeros((len(data_dict), len(data_dict)))
    for i, (index1, links1) in enumerate(data_dict.items()):
        for j, (index2, links2) in enumerate(data_dict.items()):
            intersection = len(set(links1) & set(links2))
            union = len(set(links1) | set(links2))
            overlap_rates[i, j] = intersection / union
            
    sub_matrix = overlap_rates[1:2, -17:]
    average_overlap_rate = np.mean(sub_matrix)

    print("重合率矩阵所有数值的平均值:", average_overlap_rate)
    
    # 绘制热力图
    plt.imshow(overlap_rates, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Overlap Rate')
    plt.xticks(np.arange(len(data_dict)), list(data_dict.keys()), rotation=45)
    plt.yticks(np.arange(len(data_dict)), list(data_dict.keys()))
    plt.title('Overlap Rates between Top Twenty Links of Different Centrality Measures')
    plt.xlabel('Centrality Measure')
    plt.ylabel('Centrality Measure')
    plt.show()

def indexes_to_remove(day):
    
    alpha_path = r'data\alpha\edge_alpha\BCFLOW_alpha1_day{}_level5.txt'.format(day)
    _, f_list_final_list = calculate_alpha(alpha_path, day, total_od_list)
    time_list = [ii for ii in range(6,23)]
    top_five_indexes_day_dict = {}
    
    for k in range(len(time_list)):
        differences = [f_list_final_list[k][i] - f_list_final_list[k][i+1] for i in range(len(f_list_final_list[k])-1)]
        top_five_indexes = sorted(range(len(differences)), key=lambda i: differences[i], reverse=True)[:5]
        top_five_indexes_day_dict[time_list[k]] = top_five_indexes
    
    flow = json.load(open(r"data\centrality\edge_centrality\flow_norm_{}.json".format(day)))
    
    BCflow_sort = [i[0] for i in sorted(flow[str(itime)].items(), reverse=True, key=lambda e: e[1])]
    
    
    for k,value in top_five_indexes_day_dict.items():
        top_five_indexes_day_dict[k] = [BCflow_sort[i] for i in value]
    print('top_five_indexes_day_dict',top_five_indexes_day_dict)
    return top_five_indexes_day_dict

def calculate_model_alpha_1(alpha_path, day, is_weekend):
    weekend = {'0':'workday', '1':'weekend'}
    model_total_od = json.load(open(r"data\model_total_od_{}.json".format(weekend[str(is_weekend)])))
    totalod = model_total_od[str(day)]
    f_list, _ = process_alpha_data(alpha_path)
    time_list = [ii for ii in range(6,23)]
    alpha_1_list = []
    f_list_final_list = []
    itime = [i for i in range(6,23)]
    for k in range(len(time_list)):
        f_list_final = [tmp/totalod[str(itime[k])] for tmp in f_list[k]]
        f_list_final.insert(0, 1)
        alpha = sum(f_list_final) / len(f_list_final)
        alpha_1_list.append(alpha)
        f_list_final_list.append(f_list_final)
    return alpha_1_list, f_list_final_list

def model_indexes_to_remove(is_weekend, day):
    weekend = {'0':'workday', '1':'weekend'}
    alpha_path = r'data\alpha\edge_alpha\model_BCFLOW_alpha1_{}00{}_level5.txt'.format(weekend[str(is_weekend)],day)
    _, f_list_final_list = calculate_model_alpha_1(alpha_path, day, is_weekend)
    time_list = [ii for ii in range(6,23)]
    top_five_indexes_day_dict = {}
    for k in range(len(time_list)):
        differences = [f_list_final_list[k][i] - f_list_final_list[k][i+1] for i in range(len(f_list_final_list[k])-1)]
        top_five_indexes = sorted(range(len(differences)), key=lambda i: differences[i], reverse=True)[:5]
        top_five_indexes_day_dict[time_list[k]] = top_five_indexes
    # flow = json.load(open(r"\data\centrality\edge_centrality\modelflow_{}_00{}.json".format(weekend[str(is_weekend)],day)))
    flow = json.load(open(r'data\centrality\edge_centrality\modelflow_{}_00{}.json'.format(weekend[str(is_weekend)], day))) 
    BCflow_sort = [i[0] for i in sorted(flow[str(itime)].items(), reverse=True, key=lambda e: e[1])]
    for k,value in top_five_indexes_day_dict.items():
        top_five_indexes_day_dict[k] = [BCflow_sort[i] for i in value]
    print('top_five_indexes_day_dict',top_five_indexes_day_dict)
    return top_five_indexes_day_dict

def data_ps1_ps2_rank5_real_model_data():
    # """real_data_weekend_ps1_rank5
    # """
    # rank = 5
    # data = pd.DataFrame()
    # for itime in range(6,23):
    #     for day in [5,6,11,12]:
    #         flow = json.load(open(r"data\centrality\edge_centrality\flow_norm_{}.json".format(day)))
    #         BCflow_sort = [i[0] for i in sorted(flow[str(itime)].items(), reverse=True, key=lambda e: e[1])]
    #         print('BCflow_sort[:rank]', BCflow_sort[:rank])
    #         data['Day'+str(day)+'_'+'Time'+str(itime)] = BCflow_sort[:rank]
    # data.to_csv(r'data\plot_data\weekend_ps1.csv', index=False)
    
    # """real_data_weekday_ps1_rank5
    # """
    # rank = 5
    # data = pd.DataFrame()
    # for itime in range(6,23):
    #     for day in [7,8,9,10]:
    #         flow = json.load(open(r"data\centrality\edge_centrality\flow_norm_{}.json".format(day)))
    #         BCflow_sort = [i[0] for i in sorted(flow[str(itime)].items(), reverse=True, key=lambda e: e[1])]
    #         data['Day'+str(day)+'_'+'Time'+str(itime)] = BCflow_sort[:rank]
    # data.to_csv(r'data\plot_data\weekday_ps1.csv', index=False)
    
    # """real_data_weekend_ps2_rank5
    # """
    # total_dict = {}
    # for day in [5,6,11,12]:
    #     top_five_indexes_day_dict = indexes_to_remove(day)
    #     total_dict[day] = top_five_indexes_day_dict
    # data = pd.DataFrame()
    # for itime in range(6,23):
    #     for day in [5,6,11,12]:       
    #         data['{}_{}'.format(day,itime)] = total_dict[day][itime]
    # data.to_csv(r'data\plot_data\weekend_ps2.csv', index=False)

    # """real_data_weekday_ps2_rank5
    # """
    # total_dict = {}
    # for day in [7,8,9,10]:
    #     top_five_indexes_day_dict = indexes_to_remove(day)
    #     total_dict[day] = top_five_indexes_day_dict
    # data = pd.DataFrame()
    # for itime in range(6,23):
    #     for day in [7,8,9,10]:          
    #         data['{}_{}'.format(day,itime)] = total_dict[day][itime]
    # data.to_csv(r'data\plot_data\weekday_ps2.csv', index=False)
    
    # """model_data_weekend_ps1_rank5
    # """ 
    # rank = 5
    # data = pd.DataFrame()
    # for itime in range(6,23):
    #     for day in range(10):
    #         flow = json.load(open(r"data\centrality\edge_centrality\modelflow_weekend_00{}.json".format(day)))
    #         BCflow_sort = [i[0] for i in sorted(flow[str(itime)].items(), reverse=True, key=lambda e: e[1])]
    #         data['Day'+str(day)+'_'+'Time'+str(itime)] = BCflow_sort[:rank]
    # data.to_csv(r'data\plot_data\model_weekend_ps1.csv', index=False)
    
    # """model_data_weekday_ps1_rank5
    # """
    # rank = 5
    # data = pd.DataFrame()
    # for itime in range(6,23):
    #     for day in range(10):
    #         flow = json.load(open(r"data\centrality\edge_centrality\modelflow_workday_00{}.json".format(day)))
    #         BCflow_sort = [i[0] for i in sorted(flow[str(itime)].items(), reverse=True, key=lambda e: e[1])]
    #         data['Day'+str(day)+'_'+'Time'+str(itime)] = BCflow_sort[:rank]
    # data.to_csv(r'data\plot_data\model_weekday_ps1.csv', index=False)
    
    # """model_data_weekend_ps2_rank5
    # """
    # total_dict = {}
    # for day in range(10):
    #     top_five_indexes_day_dict = model_indexes_to_remove(1, day)
    #     total_dict[day] = top_five_indexes_day_dict
    # data = pd.DataFrame()
    # for itime in range(6,23):
    #     for day in range(10):     
    #         data['{}_{}'.format(day,itime)] = total_dict[day][itime]
    # data.to_csv(r'data\plot_data\model_weekend_ps2.csv', index=False)

    # """real_data_weekday_ps2_rank5
    # """
    # total_dict = {}
    # for day in range(10):
    #     top_five_indexes_day_dict = model_indexes_to_remove(0, day)
    #     total_dict[day] = top_five_indexes_day_dict
    # data = pd.DataFrame()
    # for itime in range(6,23):
    #     for day in range(10):        
    #         data['{}_{}'.format(day,itime)] = total_dict[day][itime]
    # data.to_csv(r'data\plot_data\model_weekday_ps2.csv', index=False)
    
    
    """real+model_data_weekend_ps1_rank5
    """
    rank = 5
    data = pd.DataFrame()
    for itime in range(6,23):
        for day in [5,6,11,12]:
            flow = json.load(open(r"data\centrality\edge_centrality\flow_norm_{}.json".format(day)))
            BCflow_sort = [i[0] for i in sorted(flow[str(itime)].items(), reverse=True, key=lambda e: e[1])]
            print('BCflow_sort[:rank]', BCflow_sort[:rank])
            data['Day'+str(day)+'_'+'Time'+str(itime)] = BCflow_sort[:rank]
        for day in range(10):
            flow = json.load(open(r"data\centrality\edge_centrality\modelflow_weekend_00{}.json".format(day)))
            BCflow_sort = [i[0] for i in sorted(flow[str(itime)].items(), reverse=True, key=lambda e: e[1])]
            data['Model'+str(day)+'_'+'Time'+str(itime)] = BCflow_sort[:rank]
    data.to_csv(r'data\plot_data\real_and_model_weekend_ps1.csv', index=False)
    
    """real+model_data_workday_ps1_rank5
    """
    rank = 5
    data = pd.DataFrame()
    for itime in range(6,23):
        for day in [7,8,9,10]:
            flow = json.load(open(r"data\centrality\edge_centrality\flow_norm_{}.json".format(day)))
            BCflow_sort = [i[0] for i in sorted(flow[str(itime)].items(), reverse=True, key=lambda e: e[1])]
            print('BCflow_sort[:rank]', BCflow_sort[:rank])
            data['Day'+str(day)+'_'+'Time'+str(itime)] = BCflow_sort[:rank]
        for day in range(10):
            flow = json.load(open(r"data\centrality\edge_centrality\modelflow_workday_00{}.json".format(day)))
            BCflow_sort = [i[0] for i in sorted(flow[str(itime)].items(), reverse=True, key=lambda e: e[1])]
            data['Model'+str(day)+'_'+'Time'+str(itime)] = BCflow_sort[:rank]
    data.to_csv(r'data\plot_data\real_and_model_workday_ps1.csv', index=False)
    
    
    
    """real+model_data_weekend_ps2_rank5
    """
    total_dict = {}
    for day in [5,6,11,12]:
        top_five_indexes_day_dict = indexes_to_remove(day)
        total_dict[day] = top_five_indexes_day_dict
    total_dict1 = {}
    for day in range(10):
        top_five_indexes_day_dict = model_indexes_to_remove(1, day)
        total_dict1[day] = top_five_indexes_day_dict
    data = pd.DataFrame()
    for itime in range(6,23):
        for day in [5,6,11,12]:       
            data['{}_{}'.format(day,itime)] = total_dict[day][itime]
        for day in range(10):     
            data['model{}_{}'.format(day,itime)] = total_dict1[day][itime]
    data.to_csv(r'data\plot_data\real_and_model_weekend_ps2.csv', index=False)

    """real+model_data_workday_ps2_rank5
    """
    total_dict = {}
    for day in [7,8,9,10]:
        top_five_indexes_day_dict = indexes_to_remove(day)
        total_dict[day] = top_five_indexes_day_dict
    total_dict1 = {}
    for day in range(10):
        top_five_indexes_day_dict = model_indexes_to_remove(1, day)
        total_dict1[day] = top_five_indexes_day_dict
    data = pd.DataFrame()
    for itime in range(6,23):
        for day in [7,8,9,10]:       
            data['{}_{}'.format(day,itime)] = total_dict[day][itime]
        for day in range(10):     
            data['model{}_{}'.format(day,itime)] = total_dict1[day][itime]
    data.to_csv(r'data\plot_data\real_and_model_workday_ps2.csv', index=False)
    


    # """real_data_weekday_ps2_rank5
    # """
    # total_dict = {}
    # for day in [7,8,9,10]:
    #     top_five_indexes_day_dict = indexes_to_remove(day)
    #     total_dict[day] = top_five_indexes_day_dict
    # data = pd.DataFrame()
    # for itime in range(6,23):
    #     for day in [7,8,9,10]:          
    #         data['{}_{}'.format(day,itime)] = total_dict[day][itime]
    # data.to_csv(r'data\plot_data\weekday_ps2.csv', index=False)
    


    # """real_data_weekday_ps2_rank5
    # """
    # total_dict = {}
    # for day in range(10):
    #     top_five_indexes_day_dict = model_indexes_to_remove(0, day)
    #     total_dict[day] = top_five_indexes_day_dict
    # data = pd.DataFrame()
    # for itime in range(6,23):
    #     for day in range(10):        
    #         data['{}_{}'.format(day,itime)] = total_dict[day][itime]
    # data.to_csv(r'data\plot_data\model_weekday_ps2.csv', index=False)
    

def plot_flow_5_centrality_overlap_rate(rank):
    # rank=5
    # data = pd.read_csv(r'data\plot_data\weekend_ps1.csv')
    # data = pd.read_csv(r'data\plot_data\weekdayps1.csv')
    
    # data = pd.read_csv(r'data\plot_data\weekend_ps2.csv')
    # data = pd.read_csv(r'data\plot_data\weekday_ps2.csv')
    
    # data = pd.read_csv(r'data\plot_data\real_and_model_weekend_ps1.csv')
    # data = pd.read_csv(r'data\plot_data\real_and_model_workday_ps1.csv')
    # data = pd.read_csv(r'data\plot_data\real_and_model_weekend_ps2.csv')
    data = pd.read_csv(r'data\plot_data\real_and_model_workday_ps2.csv')
    
    
    # selected_data = data.iloc[:, -34:-18]
    selected_data = data
    data_dict = selected_data.to_dict(orient='list')
    with open(r'data\academic_data/real_and_model_workday_ps2.json', 'w', encoding='utf-8') as json_file:
        json.dump(data_dict, json_file, ensure_ascii=False, indent=4)
 
    # 计算重合率
    overlap_rates = np.zeros((len(data_dict), len(data_dict)))
    for i, (index1, links1) in enumerate(data_dict.items()):
        for j, (index2, links2) in enumerate(data_dict.items()):
            intersection = len(set(links1) & set(links2))
            union = len(set(links1) | set(links2))
            overlap_rates[i, j] = intersection / union
    
    # 将矩阵转换为DataFrame
    overlap_df = pd.DataFrame(overlap_rates, index=data_dict.keys(), columns=data_dict.keys())
    overlap_df.to_csv(r'data\academic_data/overlap_rates-real_and_model_workday_ps2.csv', index=True)
    
            
    # sub_matrix = overlap_rates[1:2, -17:]
    # average_overlap_rate = np.mean(sub_matrix)
    # print("重合率矩阵所有数值的平均值:", average_overlap_rate)
    
    # 绘制热力图
    plt.imshow(overlap_rates, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Overlap Rate')

    labels = [i for i in range(6, 23) for _ in range(14)]
    x_labels = np.arange(0, len(labels), 14)
    y_labels = np.arange(0, len(labels), 14)

    # 直接使用x_labels和y_labels的值，而不是用它们去索引labels
    # plt.xticks(x_labels, [labels[i] for i in x_labels], rotation=15)
    plt.xticks(x_labels, [labels[i] for i in x_labels])
    # plt.tick_params(axis='x', labelbottom=False)  # 隐藏底部的默认标签
    # plt.tick_params(axis='x', labeltop=True)  # 显示顶部的标签
    plt.yticks(y_labels, [labels[i] for i in y_labels])
       
    # plt.savefig(r'data\fig\ps2_weekend_overlap_time{}.pdf'.format(rank),dpi=600)
    # plt.savefig(r'data\fig\ps2_weekday_overlap_time{}.pdf'.format(rank),dpi=600)
    
    # plt.savefig(r'data\fig\real_and_model_weekend_ps1.pdf',dpi=600)
    plt.savefig(r'data\fig\real_and_model_workday_ps2.pdf',dpi=600)
    plt.show()
def centrality_transform_v1(day):
    flow = json.load(open(r'data\centrality\edge_centrality\flow_{}.json'.format(day)))
    data = flow
    itime = [i for i in range(6,23)]
    total_od_day = other_total_od_list[str(day)]
    
    total_dict = {}
    
    for itime in range(6,23):
        time_dict = {}
        for k,value in data[str(itime)].items():
            time_dict[k] = value / total_od_day[str(itime)]
        total_dict[str(itime)] = time_dict
    
    with open(r'data\centrality\edge_centrality\flow_norm_{}.json'.format(day), 'w') as json_file:
        json.dump(total_dict, json_file)
      
        
def data_centrality_var_alpha_flow_5_12():
    data = pd.DataFrame()
    for day in [i for i in range(5,13)]:
        centrality = json.load(open(r"data\centrality\edge_centrality\flow_norm_{}.json".format(day)))
        alpha_path = r'data\alpha\edge_alpha\BCFLOW_alpha1_day{}_level5.txt'.format(day)  
        flow_var = []
        for k, v in centrality.items():
            flow_var.append(np.var(np.array(list(v.values()))))
        data['var_day{}'.format(day)] = flow_var
          
        alpha_1_list, f_list_final_list = calculate_alpha(alpha_path, day, total_od_list)
        data['alpha_day{}'.format(day)] = alpha_1_list
    data.to_csv(r'data\plot_data\6_12_alpha_var.csv')


def data_alpha_level0_9():
    data = pd.DataFrame()
    for day in [5,9]:
        for method in ['ODWBC', 'BC', 'WBC', 'BCFLOW','random']:
        # for method in ['ODWBC', 'BC', 'WBC', 'BCFLOW']:
            is_improve = 1
            level = 5
            alpha_1_list,_ = calculate_alpha(r'data\alpha\edge_alpha\{}_alpha{}_day{}_level{}.txt'.format(method,is_improve,day,level), day, total_od_list)
            data[method+'_alpha'+'_level'+str(level)+'_day'+str(day)] = [sum(alpha_1_list) / len(alpha_1_list)]
    data.to_csv(r'data\plot_data\data_ave_alpha_day_9_5.csv', index=False)

def alpha_level5_average():
    method = ['BC', 'WBC', 'ODWBC', 'BCFLOW','random']
    data = pd.DataFrame()
    for day in [9,5]:
        for index in range(len(method)):
            is_improve = 1
            level = 5
            alpha_1_list,_ = calculate_alpha(r'data\alpha\edge_alpha\{}_alpha{}_day{}_level{}.txt'.format(method[index],is_improve,day,level), day, total_od_list)
            data[method[index]+str(day)] = [np.mean(alpha_1_list)]
            print(np.mean(alpha_1_list))
    data.to_csv(r'data\plot_data\alpha_level5_average.csv', index=False)
       
def plot_alpha_level0_9(day):
    xx = [ii for ii in range(6,23)]
    colors = ['#83b8ec', '#f28b82','#bc4338',  '#0f6ecc', '#18b9a3']
    legend_dict = {'WBC': r'$BC_{w}$', 'BC': r'$BC$', 'ODWBC': r'$BC_{w,od}$', 'BCFLOW': r'$BC_{w,od}^{dis}$', 'random': r'$RD$'}
    method = ['BC', 'WBC', 'ODWBC', 'BCFLOW','random']
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(15, 10))
    axes = [ax1, ax2, ax3, ax4, ax5, ax6]
    
    for index in range(len(method)):
        is_improve = 1
        level = 0
        alpha_1_list,_ = calculate_alpha(r'data\alpha\edge_alpha\{}_alpha{}_day{}_level{}.txt'.format(method[index],is_improve,day,level), day, total_od_list)
        ax1.plot(xx, alpha_1_list, marker='D',color=colors[index], label=legend_dict[method[index]])
    ax1.legend()
    # ax1.set_yticks(np.arange(0.6, 0.96, 0.05))
    ax1.set_xticks(np.arange(6, 23, 2))
    ax1.set_xlabel('Time')
    ax1.set_ylabel(r'$\alpha$')
    ax1.set_title(r'$\beta$=0')

    for index in range(len(method)):
        is_improve = 1
        level = 1
        alpha_1_list,_ = calculate_alpha(r'data\alpha\edge_alpha\{}_alpha{}_day{}_level{}.txt'.format(method[index],is_improve,day,level), day, total_od_list)
        ax2.plot(xx, alpha_1_list, marker='D',color=colors[index], label=legend_dict[method[index]])
    # ax2.legend()
    # ax2.set_yticks(np.arange(0.6, 0.91, 0.05))
    ax2.set_xticks(np.arange(6, 23, 2))
    ax2.set_xlabel('Time')
    ax2.set_ylabel(r'$\alpha$')
    ax2.set_title(r'$\beta$={}'.format(level))

    for index in range(len(method)):
        is_improve = 1
        level = 3
        alpha_1_list,_ = calculate_alpha(r'data\alpha\edge_alpha\{}_alpha{}_day{}_level{}.txt'.format(method[index],is_improve,day,level), day, total_od_list)
        ax3.plot(xx, alpha_1_list, marker='D',color=colors[index], label=legend_dict[method[index]])
    # ax3.legend()
    # ax3.set_yticks(np.arange(0.6, 0.86, 0.05))
    ax3.set_xticks(np.arange(6, 23, 2))
    ax3.set_xlabel('Time')
    ax3.set_ylabel(r'$\alpha$')
    ax3.set_title(r'$\beta$={}'.format(level))


    for index in range(len(method)):
        is_improve = 1
        level = 5
        alpha_1_list,_ = calculate_alpha(r'data\alpha\edge_alpha\{}_alpha{}_day{}_level{}.txt'.format(method[index],is_improve,day,level), day, total_od_list)
        ax4.plot(xx, alpha_1_list, marker='D',color=colors[index], label=legend_dict[method[index]])
    # ax4.legend()
    # ax4.set_yticks(np.arange(0.6, 0.81, 0.05))
        print(method[index], alpha_1_list)
    ax4.set_xticks(np.arange(6, 23, 2))
    ax4.set_xlabel('Time')
    ax4.set_ylabel(r'$\alpha$')
    ax4.set_title(r'$\beta$={}'.format(level))

    for index in range(len(method)):
        is_improve = 1
        level = 7
        alpha_1_list,_ = calculate_alpha(r'data\alpha\edge_alpha\{}_alpha{}_day{}_level{}.txt'.format(method[index],is_improve,day,level), day, total_od_list)
        ax5.plot(xx, alpha_1_list, marker='D',color=colors[index], label=legend_dict[method[index]])
    # ax5.legend()
    # ax5.set_yticks(np.arange(0.55, 0.81, 0.05))
    ax5.set_xticks(np.arange(6, 23, 2))
    ax5.set_xlabel('Time')
    ax5.set_ylabel(r'$\alpha$')
    ax5.set_title(r'$\beta$={}'.format(level))

    for index in range(len(method)):
        is_improve = 1
        level = 9
        alpha_1_list,_ = calculate_alpha(r'data\alpha\edge_alpha\{}_alpha{}_day{}_level{}.txt'.format(method[index],is_improve,day,level), day, total_od_list)
        ax6.plot(xx, alpha_1_list, marker='D',color=colors[index], label=legend_dict[method[index]])
    # ax6.legend()
    # ax6.set_yticks(np.arange(0.55, 0.81, 0.05))
    ax6.set_xticks(np.arange(6, 23, 2))
    ax6.set_xlabel('Time')
    ax6.set_ylabel(r'$\alpha$')
    ax6.set_title(r'$\beta$={}'.format(level))
    plt.savefig(r'data\fig\alpha_level0_9_day{}.pdf'.format(day), dpi=600)
    plt.show()


def plot_f_ODWBC_level0_9(day):
    colors = ['#50b918','#2d511a','#456268','#05c0eb','#3853bc','#6038bc']
    level = [0,1,3,5,9]
    xx = [ii for ii in range(21)]
    for ilevel in range(len(level)):
        alpha_path = r'data\alpha\edge_alpha\ODWBC_alpha1_day{}_level{}.txt'.format(day, level[ilevel])
        _,f = calculate_alpha(alpha_path, day, total_od_list)
        plt.plot(xx, f[2], marker='o', linestyle='-', color=colors[ilevel], label=r'$\beta$'+'='+str(level[ilevel]))
        plt.fill_between(xx, f[2], color=colors[ilevel], alpha=0.3)
    plt.xticks(range(0,21,2))
    plt.yticks(np.arange(0,1.1,0.1))
    plt.xlim([0,20])
    plt.ylim([0.4,1])
    plt.grid()
    plt.xlabel(r'$\rho$')
    plt.ylabel(r'$F$')
    plt.legend()
    plt.savefig(r'data\fig\f_ODWBC_level0_9.pdf', dpi=600)
    plt.show()


def plot_f_BCFLOW_level0_9(day):
    colors = ['#50b918','#2d511a','#456268','#05c0eb','#3853bc','#6038bc']
    level = [0,1,3,5,9]
    xx = [ii for ii in range(21)]
    for ilevel in range(len(level)):
        alpha_path = r'data\alpha\edge_alpha\BCFLOW_alpha1_day9_level{}.txt'.format(level[ilevel])
        _,f = calculate_alpha(alpha_path, day, total_od_list)
        plt.plot(xx, f[2], marker='o', linestyle='-', color=colors[ilevel], label=r'$\beta$'+'='+str(level[ilevel]))
        plt.fill_between(xx, f[2], color=colors[ilevel], alpha=0.3)
    plt.xticks(range(0,21,2))
    plt.yticks(np.arange(0,1.1,0.1))
    plt.xlim([0,20])
    plt.ylim([0.4,1])
    plt.xlabel(r'$\rho$')
    plt.ylabel(r'$F$')
    plt.grid()
    plt.legend()
    plt.savefig(r'data\fig\f_BCFLOW_level0_9.pdf', dpi=600)
    plt.show()


def model_save_ave_alpha(day):
    fig = plt.figure(figsize=(5, 4))
    colors = ['#50b918','#2d511a','#456268','#05c0eb','#3853bc','#6038bc']
    attackmethod = ['BC','WBC','ODWBC','BCFLOW']
    xx = [ii for ii in range(17)]
    for attack in attackmethod:
        alpha_path = r'data\alpha\edge_alpha\ps1_modelsave-attack{}_day9_level5.txt'.format(attack)
        alpha_list,f = calculate_alpha(alpha_path, day, total_od_list)
        plt.plot(xx, alpha_list)
    plt.xlim([0,20])
    plt.ylim([0,1])
    plt.xlabel(r'$\rho$')
    plt.ylabel(r'$F$')
    plt.legend()

    # plt.savefig(r'data\fig\appendix_fig2.pdf', dpi=600)
    plt.show()

def plot_alpha_attack():
    itime = [i for i in range(6,23)]
    data = pd.read_csv(r'data\plot_data\alpha_level5_save_BC_WBC_ODWBC_BCFLOW.csv')
    xx = data['time']
    # for method in ['BC', 'WBC', 'ODWBC', 'BCFLOW']:
    #     method_order = data['{}_order'.format(method)]
    #     method_speed = data['{}_speed'.format(method)]
    #     method_none = data['{}_none'.format(method)]
        
    BC_order = data['BC_order']
    BC_speed = data['BC_speed']
    BC_none = data['BC_none']        
    WBC_order = data['WBC_order']
    WBC_speed = data['WBC_speed']
    WBC_none = data['WBC_none']
    ODWBC_order = data['ODWBC_order']
    ODWBC_speed = data['ODWBC_speed']
    ODWBC_none = data['ODWBC_none']
    BCFLOW_order = data['BCFLOW_order']
    BCFLOW_speed = data['BCFLOW_speed']
    BCFLOW_none = data['BCFLOW_none']
    
    plt.plot(xx, BC_none, color='#D69D98', linewidth=2, marker='o', mfc='#D69D98', mew=2,label='$BC$_none')  # 实线样式，蓝色
    plt.plot(xx, BC_order, color='#9FBBD5', linewidth=2, marker='s', mfc='#9FBBD5', mew=2,label='$BC$_order')  # 实线样式，红色
    plt.plot(xx, BC_speed, color='#BA3E45', linewidth=2, marker='^', mfc='#BA3E45', mew=2,label='$BC$_speed')  # 实线样式，绿色
    # plt.plot(xx, bcf_y, color='#3A4B6E', linewidth=2, marker='D', mfc='#3A4B6E', mew=2,label='$BC_{w,od}^{dis}$')  # 实线样式，黄色
    plt.xlabel('Time')  # x轴标签
    plt.ylabel(r'$\alpha$')  # y轴标签
    plt.legend()  # 显示图例
    plt.tight_layout()  # 调整布局以防止标签被截断
    plt.show()
    
    plt.plot(xx, WBC_none, color='#D69D98', linewidth=2, marker='o', mfc='#D69D98', mew=2,label='$BC_{w}$_none')  # 实线样式，蓝色
    plt.plot(xx, WBC_order, color='#9FBBD5', linewidth=2, marker='s', mfc='#9FBBD5', mew=2,label='$BC_{w}$_order')  # 实线样式，红色
    plt.plot(xx, WBC_speed, color='#BA3E45', linewidth=2, marker='^', mfc='#BA3E45', mew=2,label='$BC_{w}$_speed')  # 实线样式，绿色
    # plt.plot(xx, bcf_y, color='#3A4B6E', linewidth=2, marker='D', mfc='#3A4B6E', mew=2,label='$BC_{w,od}^{dis}$')  # 实线样式，黄色
    plt.xlabel('Time')  # x轴标签
    plt.ylabel(r'$\alpha$')  # y轴标签
    plt.legend()  # 显示图例
    plt.tight_layout()  # 调整布局以防止标签被截断
    plt.show()
    
    plt.plot(xx, ODWBC_none, color='#D69D98', linewidth=2, marker='o', mfc='#D69D98', mew=2,label='$BC_{w,od}$_none')  # 实线样式，蓝色
    plt.plot(xx, ODWBC_order, color='#9FBBD5', linewidth=2, marker='s', mfc='#9FBBD5', mew=2,label='$BC_{w,od}$_order')  # 实线样式，红色
    plt.plot(xx, ODWBC_speed, color='#BA3E45', linewidth=2, marker='^', mfc='#BA3E45', mew=2,label='$BC_{w,od}$_speed')  # 实线样式，绿色
    # plt.plot(xx, bcf_y, color='#3A4B6E', linewidth=2, marker='D', mfc='#3A4B6E', mew=2,label='$BC_{w,od}^{dis}$')  # 实线样式，黄色
    plt.xlabel('Time')  # x轴标签
    plt.ylabel(r'$\alpha$')  # y轴标签
    plt.legend()  # 显示图例
    plt.tight_layout()  # 调整布局以防止标签被截断
    plt.show()
    
    plt.plot(xx, BCFLOW_none, color='#D69D98', linewidth=2, marker='o', mfc='#D69D98', mew=2,label='$BC$')  # 实线样式，蓝色
    plt.plot(xx, BCFLOW_order, color='#9FBBD5', linewidth=2, marker='s', mfc='#9FBBD5', mew=2,label='$BC_{w}$')  # 实线样式，红色
    plt.plot(xx, BCFLOW_speed, color='#BA3E45', linewidth=2, marker='^', mfc='#BA3E45', mew=2,label='$BC_{w,od}$')  # 实线样式，绿色
    # plt.plot(xx, bcf_y, color='#3A4B6E', linewidth=2, marker='D', mfc='#3A4B6E', mew=2,label='$BC_{w,od}^{dis}$')  # 实线样式，黄色
    plt.xlabel('Time')  # x轴标签
    plt.ylabel(r'$\alpha$')  # y轴标签
    # plt.title('Plot Title')  # 图形标题
    # plt.grid(True)  # 显示网格线
    plt.legend()  # 显示图例
    plt.tight_layout()  # 调整布局以防止标签被截断
    plt.show()
    
def plot_alpha_level0_4():
    data = pd.read_csv(r'data\plot_data\alpha0_level1_BC_WBC_ODWBC_BCFLOW_random.csv')
    xx = data['time']
    bc_y = data['BC']
    wbc_y = data['WBC']
    odwbc_y = data['ODWBC']
    bcf_y = data['BCFLOW']
    random_y = data['random']
    
    # plt.figure(figsize=(8, 6))  # 设置图形尺寸
    # plt.plot(xx, bc_y, color='#D69D98', linewidth=2, marker='o', mfc='#D69D98', mew=2,label='$BC$')  # 实线样式，蓝色
    plt.plot(xx, wbc_y, color='#9FBBD5', linewidth=2, marker='s', mfc='#9FBBD5', mew=2,label='$BC_{w}$')  # 实线样式，红色
    plt.plot(xx, odwbc_y, color='#BA3E45', linewidth=2, marker='^', mfc='#BA3E45', mew=2,label='$BC_{w,od}$')  # 实线样式，绿色
    plt.plot(xx, bcf_y, color='#3A4B6E', linewidth=2, marker='D', mfc='#3A4B6E', mew=2,label='$BC_{w,od}^{dis}$')  # 实线样式，黄色
    plt.plot(xx, random_y, color='#68BD48', linewidth=2, marker='D', mfc='#68BD48', mew=2,label='$RD$')  # 实线样式，黄色
    
    plt.plot()
    
    plt.xlabel('Time')  # x轴标签
    plt.ylabel(r'$\alpha$')  # y轴标签
    # plt.title('Plot Title')  # 图形标题
    # plt.grid(True)  # 显示网格线
    plt.legend()  # 显示图例
    plt.tight_layout()  # 调整布局以防止标签被截断
    plt.show()

def pearson_alpha_var():
    alpha_data = pd.read_csv(r'data\plot_data\alpha_level0_9.csv')
    centrality_var = pd.read_csv(r'data\plot_data\centrality_var.csv')
    centrality_var = centrality_var.iloc[:, 1:]
    
    data = pd.concat([alpha_data, centrality_var], axis=1)
    print(data.shape)
    # 计算相关系数
    correlation_matrix = data.corr()

    # 创建一个自定义的cmap，以便在热力图中控制颜色
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # 用seaborn的heatmap绘制热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, cmap=cmap, annot=False, fmt=".2f", linewidths=.5)

    # 设置标题和轴标签
    plt.title('Pearson Correlation Matrix')
    plt.xlabel('Columns')
    plt.ylabel('Columns')

    # 显示图形
    plt.show()


def plot_attack_save():
    attackmethod = ['BC', 'WBC', 'ODWBC']
    fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
    axes = [ax1, ax2, ax3, ax4]
    colors = ['#d34565', '#d3c745','#67d345',  '#45a2d3', '#d55ff1']
    colors = ['#83b8ec', '#f28b82','#bc4338',  '#0f6ecc', '#18b9a3']
    legend_dict = ['UP', 'PS1','PS2',r'PS1-$BC_{w,od}^{dis}$',r'PS2-$BC_{w,od}^{dis}$']
    title_dict = {'WBC': r'$BC_{w}$', 'BC': r'$BC$', 'ODWBC': r'$BC_{w,od}$', 'BCFLOW': r'$BC_{w,od}^{dis}$', 'random': r'$RD$'}
    
    xx = [ii for ii in range(6,23)]
    day = 9
    for i in range(len(attackmethod)):
        origin_data = r'data\alpha\edge_alpha\{}_alpha1_day{}_level5.txt'.format(attackmethod[i], day)
        print(origin_data)
        save_order = r'data\alpha\edge_alpha\{}_alpha1_day{}_level5_r0_saveorder.txt'.format(attackmethod[i], day)
        save_speed = r'data\alpha\edge_alpha\{}_alpha1_day{}_level5_r0_savespeed.txt'.format(attackmethod[i], day)
        save_flow_order = r'data\alpha\edge_alpha\order_attack-{}_save-BCFLOW_day{}_level5.txt'.format(attackmethod[i], day)
        save_flow_speed = r'data\alpha\edge_alpha\speed_attack-{}_save-BCFLOW_day{}_level5.txt'.format(attackmethod[i], day)
        alpha_origin_data ,_ = calculate_alpha(origin_data, day, total_od_list)
        alpha_save_order ,_ = calculate_alpha(save_order, day, total_od_list)
        alpha_save_speed ,_ = calculate_alpha(save_speed, day, total_od_list)
        alpha_save_flow_order ,_ = calculate_alpha(save_flow_order, day, total_od_list)
        alpha_save_flow_speed ,_ = calculate_alpha(save_flow_speed, day, total_od_list)
        axes[i].plot(xx, alpha_origin_data, marker='D',color=colors[0], label=legend_dict[0])
        axes[i].plot(xx, alpha_save_order, marker='D',color=colors[1], label=legend_dict[1])
        axes[i].plot(xx, alpha_save_speed, marker='D',color=colors[2], label=legend_dict[2])
        axes[i].plot(xx, alpha_save_flow_order, marker='D',color=colors[3], label=legend_dict[3])
        axes[i].plot(xx, alpha_save_flow_speed, marker='D',color=colors[4], label=legend_dict[4])
        axes[i].set_title(title_dict[attackmethod[i]])
        axes[i].set_xticks(range(6,23,2))
        
        
    attackmethod = 'BCFLOW'
    origin_data = r'data\alpha\edge_alpha\{}_alpha1_day{}_level5.txt'.format(attackmethod, day)
    save_order = r'data\alpha\edge_alpha\{}_alpha1_day{}_level5_r0_saveorder.txt'.format(attackmethod, day)
    save_speed = r'data\alpha\edge_alpha\{}_alpha1_day{}_level5_r0_savespeed.txt'.format(attackmethod, day)
    alpha_origin_data ,_ = calculate_alpha(origin_data, day, total_od_list)
    alpha_save_order ,_ = calculate_alpha(save_order, day, total_od_list)
    alpha_save_speed ,_ = calculate_alpha(save_speed, day, total_od_list)
    axes[3].plot(xx, alpha_origin_data, marker='D',color=colors[0], label=legend_dict[0])
    axes[3].plot(xx, alpha_save_order, marker='D',color=colors[1], label=legend_dict[1])
    axes[3].plot(xx, alpha_save_speed, marker='D',color=colors[2], label=legend_dict[2])
    axes[3].set_title(title_dict[attackmethod])
    axes[3].set_xticks(range(6,23,2))
    axes[3].set_yticks(np.arange(0.60,0.78,0.025))

    axes[0].legend(loc='upper left', fontsize=8)
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel(r'$\alpha$')
    # axes[1].legend(loc='upper left', fontsize=8)
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel(r'$\alpha$')
    # axes[2].legend(loc='upper left', fontsize=8)
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel(r'$\alpha$')
    # axes[3].legend(loc='upper left', fontsize=8)
    axes[3].set_xlabel('Time')
    axes[3].set_ylabel(r'$\alpha$')

    plt.savefig(r'data\fig\attack_save_alpha.pdf', dpi=600)
    plt.show()

def data_average_attack_save():
    attackmethod = ['BC', 'WBC', 'ODWBC']
    legend_dict = ['UP', 'PS1','PS2',r'PS1-$BC_{w,od}^{dis}$',r'PS2-$BC_{w,od}^{dis}$']
    title_dict = {'WBC': r'$BC_{w}$', 'BC': r'$BC$', 'ODWBC': r'$BC_{w,od}$', 'BCFLOW': r'$BC_{w,od}^{dis}$', 'random': r'$RD$'}
    data = pd.DataFrame()
    day = 9
    for i in range(len(attackmethod)):
        origin_data = r'data\alpha\edge_alpha\{}_alpha1_day{}_level5.txt'.format(attackmethod[i], day)
        save_order = r'data\alpha\edge_alpha\{}_alpha1_day{}_level5_r0_saveorder.txt'.format(attackmethod[i], day)
        save_speed = r'data\alpha\edge_alpha\{}_alpha1_day{}_level5_r0_savespeed.txt'.format(attackmethod[i], day)
        save_flow_order = r'data\alpha\edge_alpha\order_attack-{}_save-BCFLOW_day{}_level5.txt'.format(attackmethod[i], day)
        save_flow_speed = r'data\alpha\edge_alpha\speed_attack-{}_save-BCFLOW_day{}_level5.txt'.format(attackmethod[i], day)
        alpha_origin_data ,_ = calculate_alpha(origin_data, day, total_od_list)
        alpha_save_order ,_ = calculate_alpha(save_order, day, total_od_list)
        alpha_save_speed ,_ = calculate_alpha(save_speed, day, total_od_list)
        alpha_save_flow_order ,_ = calculate_alpha(save_flow_order, day, total_od_list)
        alpha_save_flow_speed ,_ = calculate_alpha(save_flow_speed, day, total_od_list)
        data[attackmethod[i]+legend_dict[0]] = [np.mean(alpha_origin_data)]
        data[attackmethod[i]+legend_dict[1]] = [np.mean(alpha_save_order)]
        data[attackmethod[i]+legend_dict[2]] = [np.mean(alpha_save_speed)]
        data[attackmethod[i]+legend_dict[3]] = [np.mean(alpha_save_flow_order)]
        data[attackmethod[i]+legend_dict[4]] = [np.mean(alpha_save_flow_speed)]

    attackmethod = 'BCFLOW'
    origin_data = r'data\alpha\edge_alpha\{}_alpha1_day{}_level5.txt'.format(attackmethod, day)
    save_order = r'data\alpha\edge_alpha\{}_alpha1_day{}_level5_r0_saveorder.txt'.format(attackmethod, day)
    save_speed = r'data\alpha\edge_alpha\{}_alpha1_day{}_level5_r0_savespeed.txt'.format(attackmethod, day)
    alpha_origin_data ,_ = calculate_alpha(origin_data, day, total_od_list)
    alpha_save_order ,_ = calculate_alpha(save_order, day, total_od_list)
    alpha_save_speed ,_ = calculate_alpha(save_speed, day, total_od_list)
    data[attackmethod+legend_dict[0]] = [np.mean(alpha_origin_data)]
    data[attackmethod+legend_dict[1]] = [np.mean(alpha_save_order)]
    data[attackmethod+legend_dict[2]] = [np.mean(alpha_save_speed)]

    data.to_csv(r'data\plot_data\average_attack_save.csv')

def calculate_model_alpha(alpha_path, is_weekend):
    weekend = {'0':'workday', '1':'weekend'}
    model_total_od_list = json.load(open(r"data\model_total_od_{}.json".format(weekend[str(is_weekend)])))
    totalod = model_total_od_list[str(is_weekend)]
    f_list, alpha_list = process_alpha_data(alpha_path)
    time_list = [ii for ii in range(6,23)]
    alpha_1_list = []
    f_list_final_list = []
    itime = [i for i in range(6,23)]
    for k in range(len(time_list)):
        f_list_final = [tmp/totalod[str(itime[k])] for tmp in f_list[k]]
        f_list_final.insert(0, 1)
        alpha = sum(f_list_final) / len(f_list_final)
        alpha_1_list.append(alpha)
        f_list_final_list.append(f_list_final)
    return alpha_1_list, f_list_final_list
def data_model_alpha():
    xx = [i for i in range(6,23)]
    for is_weekend in [0,1]:
        alpha_1_list,_ = calculate_model_alpha(r'data\alpha\edge_alpha\model_BCFLOW_alpha1_weekend{}_level5.txt'.format(is_weekend),is_weekend)
        print(is_weekend, alpha_1_list)
        plt.plot(xx,alpha_1_list,label=is_weekend)
    plt.legend()
    plt.show()
