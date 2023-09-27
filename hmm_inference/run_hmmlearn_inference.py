import numpy as np
import glob, os
import pandas as pd
import torch
from hmmlearn.hmm import GaussianHMM as hmm_model
from utils import _score_hmm, _compute_hmm_similarity, get_oracle_threshold
import argparse
from datetime import date

def train_hmm(args):
    '''
    Loads dataset in exp_data_path (produced using process_for_inference with args.sklearn= True
    Trains an n_component GaussianHMM with diagonal covariance on the data provided across random states in args.seeds
    Saves trained models, test scores and assignments for test data into args.save_dir
    
    '''
    
    data = torch.load(args.exp_data_path)
    key = args.save_key
    final_test_scores_list = []
    models = []
    n_params_list = []
    assignments_all = []
    bic_list = []
    seed_list = []
    adj_rand_path_list = []
    adj_rand_majority_list = []
    oracle_thresh_list = []
    X_all_train = data['train']['sklearn_sequences']
    lengths_all_train = data['train']['sklearn_sequence_lengths']
    X_all_test = data['test']['sklearn_sequences']
    lengths_all_test = data['test']['sklearn_sequence_lengths']
    print('Starting path {}. Train shape = {}, Test shape = {}'.format(data['tree_cohort_path'], X_all_train.shape, X_all_test.shape))
    
    for seed in list(args.seeds):
        seed_list.append(seed)
        testing_model = hmm_model(n_components=args.n_components, covariance_type = 'diag', random_state=seed)
        testing_model.fit(X_all_train, lengths_all_train)

        likelihood, assignments = testing_model.score_samples(X_all_test, lengths_all_test)
        log_N = np.log(len(lengths_all_train))
        bic = testing_model.bic(X_all_test, lengths_all_test)
        n_params = sum(testing_model._get_n_fit_scalars_per_param().values())
        
        final_test_scores_list.append(likelihood)
        assignments_all.append(assignments)
        models.append(testing_model)
        bic_list.append(bic)
        n_params_list.append(n_params)
        
            
    results_dict = {
        'seed':seed_list,
        'tree_cohort': key,
        'final_test_score':final_test_scores_list, 
        'assignments_all':assignments_all,
        'bic_list':bic_list,
        'models':models, 
        'n_params':n_params_list
    }
    
    
    torch.save(results_dict,args.save_dir + f'final_results_hmmlearn_{key}.pt')
    return results_dict


if __name__ == '__main__':
    today = date.today()
    d4 = today.strftime("%b-%d-%Y")

    save_key_default = f'results_hmmlearn_{d4}'

    parser = argparse.ArgumentParser(
        description="Running HMM Learn EM Baum-Welch"
    )
    parser.add_argument(
        "--exp_data_path",
        type=str,
        help="Absolute path to location of experiment data dictionary produced by process_for_inference.py",
    )
    
    parser.add_argument('--seeds', type=list, default=[i for i in range(3)])
    parser.add_argument('--n_components', type=int, default=2)
    
    parser.add_argument("--save_dir",default=os.getcwd(), type=str)
    parser.add_argument("--save_key",default=save_key_default, type=str)
    
    args = parser.parse_args()
    train_hmm(args)