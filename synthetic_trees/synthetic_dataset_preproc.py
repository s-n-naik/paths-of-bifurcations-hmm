
import pandas as pd
import numpy as np
import os, glob
import matplotlib.pyplot as plt
import torch
import networkx as nx
from tqdm.notebook import tqdm
import time


import pandas as pd
import numpy as np
import os, json
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import time
import argparse
import random
# ### Data Loading

def _process_synthetic_tree_data(args):
    '''
    Iterate over individual trees, rename columns and generate paths per graph in networkx
    
    '''
    #synthetic data path
    all_trees = glob.glob(args.folder_path+'/tree*.csv')
    print("# Trees", len(all_trees))
    print(all_trees[0])

    
    path_info_per_person_all = []

    for i in tqdm(range(len(all_trees)), desc='Iterating individual tree csv files'):
        path = all_trees[i]
        tree_df = pd.read_csv(path)
        # remove unneccary cols
        tree_df.drop(columns=[col for col in tree_df.columns if 'unnamed' in col.lower()], inplace=True)
        # remove scaffold
        tree_df = tree_df.loc[tree_df.parent_id.str.len() > 2]
        # get idno + gt cluster generated from
        path_info = path.split('/')[-1].strip('.csv').strip('tree_').split('_')
        # print(path_info)
        idno= int(path_info[0])
        tree_df['idno'] = idno
        tree_df['gt'] = args.gt_cluster
        # get leaf flag
        tree_df['leaf_flag'] = tree_df.p_emit < 0.5
        tree_df['no_children'] = ~tree_df.child_id.isin(tree_df.parent_id.unique())
        # get parent of parent to make graph
        tree_df.child_id = tree_df.child_id.str.strip('[]').str.split(',')
        tree_df_exploded = tree_df[['parent_id', 'child_id']].explode(column=['child_id'])
        tree_df_exploded.child_id = tree_df_exploded.child_id.str.strip("' '")
        tree_df = tree_df.merge(tree_df_exploded.rename({'parent_id':'parent_of_parent', 'child_id':'parent_id'}, axis=1), on='parent_id', how='left')
        # get leaf nodes
        leave1 = tree_df.loc[tree_df.leaf_flag].parent_id
        leave2 = tree_df.loc[tree_df.no_children].parent_id
        if len(leave1) == 0:
            leaves = leave2
        else:
            leaves = leave1
        root = tree_df.loc[tree_df.parent_of_parent.isnull()].parent_id.item()
        tree_df = tree_df.loc[~tree_df.parent_of_parent.isnull()]
        g = nx.DiGraph()
        nodes =tree_df.parent_id.to_list()
        edges = list(zip(tree_df.parent_of_parent, tree_df.parent_id))
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        
        for leaf in leaves:
            path =list(nx.shortest_simple_paths(g,root,leaf))
            path_info = tree_df.loc[tree_df.parent_id.isin(path[0])].copy()
            path_info['N_M'] = str(idno)+"_"+str(leaf)
            path_info_per_person_all.append(path_info)


    tree_df_synthetic = pd.concat(path_info_per_person_all, axis=0)
    # process angles
    print("Processing angles")
    try:
        tree_df_synthetic.rename(columns={'parent_of_parent':'startbpid', 'parent_id':'endbpid'}, inplace=True)
    except:
        print(tree_df_synthetic.columns)
    tree_df_synthetic = _process_synthetic_angles(tree_df_synthetic)
    print("Saving dataset")
    
    tree_df_synthetic.to_csv(args.save_path)
    print('Saved to', args.save_path)
    return tree_df_synthetic

def _process_synthetic_angles(df_map):
    '''
    Converts angles to inner + Scales by 180 so in range 0-1 for inference
    
    '''
    df_map[['alpha1_abs','alpha2_abs','phi_abs']] = np.abs(df_map[['alpha1','alpha2','phi']].copy().values)
    # relabel so aloha1 always smaller than alpha2
    df_map['alpha1_smaller'] = np.min(df_map[['alpha1_abs','alpha2_abs']].values, axis=1)
    df_map['alpha2_larger'] = np.max(df_map[['alpha1_abs','alpha2_abs']].values, axis=1)
    df_map[['alpha1_scaled', 'alpha2_scaled', 'phi_scaled']] = df_map[['alpha1_smaller','alpha2_larger','phi_abs']].values/180
    df_map['alpha2_scaled'] = np.abs(df_map['alpha2_scaled'].values)
    
    return df_map


    
    
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Generation of Synthetic Trees")

    parser.add_argument("--folder_path", default=os.getcwd(), type=str)
    parser.add_argument("--stopping", default=True, type=bool)
    parser.add_argument("--gt_cluster", default=0, type=int)
    parser.add_argument("--save_path",
                        default= os.getcwd()+"/total_path_dataset.csv",
                        type=str)
    args = parser.parse_args('')

    tree_df_synthetic = _process_synthetic_tree_data(args)
