import os
import pdb
import random
import math
import pickle
import torch
import time 
from tqdm import tqdm

in_path = 'wordnet-mlj12-definitions.txt'
out_path = 'my_entity2text.txt'

descriptions = {}
with open(in_path, 'r', encoding='utf8') as fil:
	for line in fil.readlines():
		idx, name, description = line.strip('\n').split('\t')
		name = ' '.join([ s for s in name.split('_')[:-2] if s != ''])
		descriptions[idx] = name +' : ' + description 

with open(out_path, 'w', encoding='utf8') as fil:
	for k, v in descriptions.items():
		#pdb.set_trace()
		fil.write('{}\t{}\n'.format(k, v))
		