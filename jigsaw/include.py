import argparse
import cv2
import os
import numpy as np
import json
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
from sklearn.neighbors import KDTree
import networkx as nx

# add your path here
# on h050018
data_dir = r'/wgdisk/st0008/hzh/workspace/tgs/input'
# on my laptop
# data_dir = r'/home/hzh/MachineLearning/equinor/tgs/input'
