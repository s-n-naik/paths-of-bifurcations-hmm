import os, glob
from collections import defaultdict
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import argparse
from pprint import pprint
from datetime import date
from sklearn.model_selection import train_test_split



def load_path(args: argparse.Namespace, data: dict):
    '''
    param args: arguments for script
    param data: dictionary to store experiment data + information

    Load path dataset df from args.tree_cohort_path folder
    Load gt_info dictionary from args.tree_cohort_path folder
    
    Check there are no null values in key columns of df

    returns df: # path dataset loaded for experiment
    return data: dictionary with gt_info recorded
    '''
    path = args.tree_cohort_path
    df = pd.read_csv(path+'/total_path_dataset.csv')
    if args.synthetic:
        gt_info = torch.load(path + '/gt_info.pkl')
        data['gt_info'] = gt_info

    # Check for nulls in key columms
    key_cols = ['idno', 'startbpid', 'endbpid', 'N_M', 'alpha1_scaled', 'alpha2_scaled', 'phi_scaled']
    if df[key_cols].isnull().sum().sum() > 0:
        print('Number of rows incl nulls', len(df))
        df.dropna(inplace=True)
        print('Number of rows without nulls', len(df))
    else:
        print('No null values in dataset')
    print('Processing {}, # max / mean paths per tree = {}/{}. NOTE: mkeep = {}'.format(args.tree_cohort_path,
                                                                           df.groupby('idno').agg({'N_M':'nunique'}).max().item(),
                                                                       df.groupby('idno').agg({'N_M':'nunique'}).mean().item(),
                                                                       args.m_keep
                                                                       
                                                                       ))
    data['df_info'] = {'max_paths':df.groupby('idno').agg({'N_M':'nunique'}).max().item(),
                       'mean_paths': df.groupby('idno').agg({'N_M':'nunique'}).mean().item(),
                       'std_paths': df.groupby('idno').agg({'N_M':'nunique'}).std().item(),
                       'mean_len_paths': df.groupby('N_M').agg({'endbpid':'count'}).mean().item(),
                       'std_len_paths': df.groupby('N_M').agg({'endbpid':'count'}).std().item(),
                       'max_len_paths': df.groupby('N_M').agg({'endbpid':'count'}).max().item(),
                       'max_nodes': df.groupby('idno').agg({'endbpid':'nunique'}).max().item(),
                       'mean_nodes': df.groupby('idno').agg({'endbpid':'nunique'}).mean().item(),
                       'std_nodes': df.groupby('idno').agg({'endbpid':'nunique'}).std().item(),
                       }

    return df, data




def filter_M_paths_per_person(M_keep, df):
    '''
    At random, shuffles all paths and keeps M_keep of the paths per tree
    If there are < M_keep paths, there are zeros in the matrix. These are filtered out as the length is 0 in T_i_df later
    returns df_map: Path dataset with up to M_keep paths per tree
    '''
    groups = df.groupby('idno')
    chains_all = []
    for idno, group in groups:
        group = group.sample(len(group))
        chains = group.N_M.drop_duplicates().to_list()
        if len(chains) > M_keep:
            chains_keep = chains[:M_keep]
    
        else:
            chains_keep = chains
        chains_all.extend(list(chains_keep))

    df_map = df.loc[df.N_M.isin(chains_all)]
    print('Original df len=',len(df),'Df map len=' ,len(df_map))
    return df_map


def get_sequence_lengths(df_map, M_keep):
    '''
    Computes sequence lengths for each N_M chain in the dataset that is kept.
    If there is no data in a place - zeros in the lengths
    Returns T_i_df (idno, N_M, T_i) Dataframe and T_i_final which is the array version
    
    '''

    T_i_df = df_map[['idno', 'N_M', 'endbpid']].groupby([pd.Grouper('idno'), pd.Grouper('N_M')]).agg(lambda x:len(x)).reset_index().rename({'endbpid':'T_i'}, axis=1)
    M_df = df_map[['idno', 'endbpid']].groupby('idno').agg('count')
    max_T_i = T_i_df.T_i.max()
    print('Max path length', max_T_i)
    T_i_final = torch.zeros(T_i_df.idno.nunique(), M_keep)
    T_i_chains = T_i_df.groupby('idno').agg(lambda x:x.to_list())
    for n in range(T_i_final.shape[0]):
        chain = T_i_chains.iloc[n].T_i
        T_i_final[n,:len(chain)] = torch.tensor(chain)

    return T_i_df, T_i_final



def _get_mask(T_i_final,T_i_df):
    '''
    Returns binary mask (# people , # paths, length of path) with 1s and 0s if there is an observation at that point or not/
    
    '''
    max_T_i = T_i_df.T_i.max()
    mask_all = torch.ones_like(T_i_final.unsqueeze(2))
    mask_all = mask_all.repeat(1,1,max_T_i) # N,M,T
    # T_i_final is shape NxM
    t_to_filter = torch.arange(max_T_i).unsqueeze(0).unsqueeze(0).repeat(mask_all.shape[0], mask_all.shape[1], 1)
    mask_filter = T_i_final.unsqueeze(2).repeat(1,1,max_T_i)
    mask_all[torch.where(t_to_filter >= mask_filter)]=0
    return mask_all



def get_angles_spatial_leaf_arrays(df_map, mask_all, T_i_final):
    '''
    Returns Torch arrays:(1) Y_all = angle observations  or 0s if no data (# people , # paths, length of path, dim=3)
            (2) D_all = spatial feature observations or 0s if no data (# people , # paths, length of path, dim=4)
            (3) e_all = leaf flag binary if a leaf otherwise false, one true per path, all other false
            (4) rules_all = rule used to generate each step in the synthetic data (# people , # paths, length of path), 0s if none 
            (5) gt_all = cluster ground truth 
            (6) indic_all = str(idno_endbpid) for each entry in rules_all to map back to original graph
            Note: all have 0s where no data, must apply mask to it to make srue youre not overcounting a class or datapoint 0 with the 'no data' zeros.
    
    '''
    Y_all = torch.zeros_like(mask_all).unsqueeze(3).repeat(1,1,1,3)
    D_all = torch.zeros_like(mask_all).unsqueeze(3).repeat(1,1,1,4)
    e_all = torch.zeros_like(mask_all)
    rules_all = torch.zeros_like(mask_all)
    indic_all = np.zeros_like(mask_all).astype(object)
    grouped_df = df_map.groupby([pd.Grouper('idno')])
    gt_all = torch.zeros(len(grouped_df))
 
    pbar = tqdm(list(enumerate(grouped_df)),bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', desc='Generating Arrays')
    for i, ((idno), group) in pbar:
        pbar.set_description(f"i = {idno}")
        if  set(['gt']).issubset(set(df_map.columns)):
            # get ground truth if it is there (for synthetic data only)
            gt_all[i] = group['gt'].unique().item()
            
        nm_group_df = group.groupby('N_M')
        for j, (name, nm_group) in list(enumerate(nm_group_df)):
            # print('here',Y_all[i,j,:int(T_i_final[i,j]),:].shape,torch.tensor(nm_group[['alpha1_scaled', 'alpha2_scaled', 'phi_scaled']].values.shape))
            Y_all[i,j,:int(T_i_final[i,j]),:] = torch.tensor(nm_group[['alpha1_scaled', 'alpha2_scaled', 'phi_scaled']].values)
            
            if set(['d1', 'd2', 'd3', 'd4']).issubset(set(df_map.columns)):
            
                D_all[i,j,:int(T_i_final[i,j]),:] = torch.tensor(nm_group[['d1', 'd2', 'd3', 'd4']].values)
            
            if  set(['leaf_flag']).issubset(set(df_map.columns)):
                e_all[i,j,:int(T_i_final[i,j])] = ~torch.tensor(nm_group[['leaf_flag']].astype(bool).values).squeeze()
            if  set(['rule']).issubset(set(df_map.columns)):
                rules_all[i,j,:int(T_i_final[i,j])] = torch.tensor(nm_group[['rule']].astype(int).values).squeeze()
                indic_all[i,j,:int(T_i_final[i,j])] = nm_group.idno.astype(str).values + '_' + nm_group.endbpid.astype(str).values
    return Y_all, D_all, e_all, gt_all, rules_all, indic_all




# In[224]:
def _train_test_split(test_size,df_map,T_i_df, Y_all, T_i_final, rules_all, mask_all, sklearn, save_dir, indic_all=None,data={}
                     ):

    # Split # ppl into train - test
    train_idx, test_idx = train_test_split(np.arange(Y_all.shape[0]), test_size=test_size, shuffle=True)
    print("# train", len(train_idx), '# test', len(test_idx))
    
    
    M_keep = Y_all.shape[1]
    # Get sequences for the train ppl
    sequences_train = Y_all[train_idx].reshape(len(train_idx)*Y_all.shape[1], Y_all.shape[2], Y_all.shape[3]) # shape = (ppl x paths), length, dim
    lengths_train = T_i_final[train_idx]# shape = ppl ,paths, length
    rules_train = rules_all[train_idx].reshape(len(train_idx)*rules_all.shape[1], rules_all.shape[2]) # shape = (ppl x paths), length
    mask_train = mask_all[train_idx].reshape(len(train_idx)*mask_all.shape[1], mask_all.shape[2]) # shape = (ppl x paths), length
    
    # Get sequences for the test ppl
    sequences_test = Y_all[test_idx].reshape(len(test_idx)*Y_all.shape[1], Y_all.shape[2], Y_all.shape[3])                          
    lengths_test = T_i_final[test_idx]
    rules_test = rules_all[test_idx].reshape(len(test_idx)*rules_all.shape[1], rules_all.shape[2])
    mask_test = mask_all[test_idx].reshape(len(test_idx)*mask_all.shape[1], mask_all.shape[2])
    if indic_all is not None:
        indic_all_train = indic_all[train_idx].reshape(len(train_idx)*indic_all.shape[1], indic_all.shape[2])
        indic_all_test = indic_all[test_idx].reshape(len(test_idx)*indic_all.shape[1], indic_all.shape[2])
    else:
        indic_all_train = None
        indic_all_test = None
    # Get the idnos for the train-test split for later
    idnos_train = df_map.idno.unique()[train_idx]
    idnos_test =  df_map.idno.unique()[test_idx]
    
    
    
    data['train'] = {'sequences':sequences_train, 
                      'sequence_lengths': lengths_train.flatten().int(), 
                      'mask': mask_train,
                      'rules': rules_train.int(),
                      'idnos':idnos_train,
                      'idx': train_idx, 'idno_endbpid':indic_all_train}
    
    data['test'] = {'sequences':sequences_test ,
                     'sequence_lengths': lengths_test.flatten().int(),  
                     'mask': mask_test,
                     'rules': rules_test.int(),
                     'idnos':idnos_test,
                     'idx': test_idx,
                     'idno_endbpid':indic_all_test}
    

    if sklearn:
        # Gets same train-test data into format for sklearn models (no zeros)
        X_all_train = df_map.loc[df_map.idno.isin(idnos_train)][['alpha1_scaled', 'alpha2_scaled', 'phi_scaled']].values
        lengths_all_train =  T_i_df.loc[T_i_df.idno.isin(idnos_train)].T_i.values
        rules_all_train  = df_map.loc[df_map.idno.isin(idnos_train)]['rule'].values

        X_all_test = df_map.loc[df_map.idno.isin(idnos_test)][['alpha1_scaled', 'alpha2_scaled', 'phi_scaled']].values
        lengths_all_test =  T_i_df.loc[T_i_df.idno.isin(idnos_test)].T_i.values
        rules_all_test  = df_map.loc[df_map.idno.isin(idnos_test)]['rule'].values

        
        # Add more data
        data['train']['sklearn_sequences'] = X_all_train
        data['train']['sklearn_sequence_lengths'] = lengths_all_train
        data['train']['sklearn_rules'] = rules_all_train
        data['test']['sklearn_sequences'] = X_all_test
        data['test']['sklearn_sequence_lengths'] = lengths_all_test
        data['test']['sklearn_rules'] = rules_all_test
        data['orig_df'] = df_map
        data['M_keep'] = M_keep
        
        
    # Save data
    if save_dir is not None:
        print('Saving data for later')
        torch.save(data, save_dir[0] +f'/experiment_data_{save_dir[1]}.pt')
        print("Saved data",save_dir[0] +f'/experiment_data_{save_dir[1]}.pt') 
    return data







def main(args):
    data = {}
    data['tree_cohort_path'] = args.tree_cohort_path
    df, data = load_path(args, data)

    if args.m_keep is None:
        m_keep = df.groupby('idno').agg({'N_M':'nunique'}).max().item()
        args.m_keep = m_keep
        print('Set m_keep to max # paths/tree in dataset = {}'.format(args.m_keep))
    
    # filter/pad paths per tree to m_keep
    df_map = filter_M_paths_per_person(args.m_keep, df)
    # compute sequence lengths
    T_i_df, T_i_final = get_sequence_lengths(df_map, args.m_keep)
    # compute tensors for pyro training (obs, rules, d, etc.)
    mask_all = _get_mask(T_i_final,T_i_df)
    Y_all, D_all, e_all, gt_all, rules_all, indic_all =  get_angles_spatial_leaf_arrays(df_map, mask_all, T_i_final)
    # train-test split

    save_dir = (args.save_dir,
                args.save_key)


    data_out = _train_test_split(args.test_size,
                             df_map,
                             T_i_df,
                             Y_all, 
                             T_i_final, 
                             rules_all, 
                             mask_all, 
                             args.sklearn,
                             save_dir=save_dir,
                             indic_all=indic_all,
                             data=data
                             )
    



    
    
    pprint(data)


if __name__ == '__main__':
    pass
    today = date.today()
    d4 = today.strftime("%b-%d-%Y")



    parser = argparse.ArgumentParser(
        description="Preprocessing total_path_dataset.csv file for inference"
    )
    parser.add_argument(
        "--tree_cohort_path",
        type=str,
        help="Absolute path to location of total_path_dataset.csv file generated for cohort of trees",
    )
    parser.add_argument(
        "--save_dir",
        default=os.getcwd(),
        type=str,
        help="Absolute path to location of where to save experiment results",
    )
    parser.add_argument("--m_keep", default=None, type=int,help="# Paths in every tree (drop paths / pad to size)")
    parser.add_argument("--synthetic", default=True, type=bool,help="Is the tree cohort synthetic data (with ground truth) or not?")

    parser.add_argument("--sklearn", default=True, type=bool,help="Prepare data for use with hmmlearn or not?")
    parser.add_argument("--test_size", default=0.2, type=float,help="Proportion of trees to be allocated to the test set")
    parser.add_argument("--save_key", default=d4, type=str,help="save key for data & final experiment results")

    args = parser.parse_args()

    assert 'total_path_dataset.csv' in os.listdir(args.tree_cohort_path), f'There is no path-processed data in {args.tree_cohort_path}'
    if args.synthetic:
        assert 'gt_info.pkl' in os.listdir(args.tree_cohort_path), f'There is no gt_info in {args.tree_cohort_path}'

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        print('Made save dir {}'.format(args.save_dir) )
    print(args)
    main(args)