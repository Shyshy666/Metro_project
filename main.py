import warnings
warnings.filterwarnings("ignore")
import argparse
import random
import time
import networkx as nx
import pandas as pd
from tqdm import tqdm
from utils import io as utils_io
from utils import visualization as utils_visualization
from Models import edge_cwb
from Models import edge_alpha
import json
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns


# -------------------------------------------------------------
day = 9
is_improve_alpha = 1
method = 'BCFLOW'
level = 3
edge_alpha.partial_origin_cwb_alpha_percolation(day, is_improve_alpha, method, level)

# -------------------------------------------------------------

all_path_length = json.load(open(r"subway-percolation/data/node_ij_now_shortest_path_length.json"))
all_short_paths = json.load(open(r"subway-percolation/data/all_pair/all_pairs_journey.json"))
for day in range(1, 4):
    edge_cwb.edge_workday_model_flow(day, all_short_paths, all_path_length)

# -------------------------------------------------------------
level = 5
is_improve_alpha = 1
is_weekend = 0
method = 'BCFLOW'
for day in range(1, 4):
    edge_alpha.model_cwb_alpha_percolation(is_weekend, day, is_improve_alpha, method, level)

# -------------------------------------------------------------
level = 1
is_improve_alpha = 1
day = 9
for repeat in range(3):
    edge_alpha.random_alpha_percolation_repeat(day, is_improve_alpha, level, repeat)


# -------------------------------------------------------------
level = 7
method = 'BCFLOW'
is_improve_alpha = 1
day = 9
edge_alpha.origin_cwb_alpha_percolation(day, is_improve_alpha, method, level)

