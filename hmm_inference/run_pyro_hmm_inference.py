
import os, glob
from collections import defaultdict
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch_geometric as geometric
import argparse
from pprint import pprint
from datetime import date
import functools
from collections import Counter

# pyro packages
import pyro
from pyro import poutine
from pyroapi import distributions as dist
from pyro.util import ignore_jit_warnings
from pyro.infer.autoguide import AutoDelta
from pyro.infer import SVI, JitTraceEnum_ELBO, TraceEnum_ELBO, TraceTMC_ELBO, ELBO, infer_discrete
from pyro.ops.indexing import Vindex
from pyroapi import handlers, infer, optim, pyro, pyro_backend

import sklearn
from sklearn.metrics import adjusted_rand_score

from pyro_models import model_3
from utils import _score_hmm, _compute_hmm_similarity, get_oracle_threshold



smoke_test = ('CI' in os.environ)
assert pyro.__version__.startswith('1.8.4')

pyro.enable_validation(True)

global models
models = {
    name[len("model_") :]: model
    for name, model in globals().items()
    if name.startswith("model_")
}

pprint(models)


def classifier(sequences,lengths,trained_guide,trained_model, temperature=0):
    '''
    Returns MAP estimates of rules along each path in pyro

    '''
    model_data = trained_guide()
    cond_model = poutine.condition(trained_model, model_data)
    
    inferred_model = infer_discrete(
        cond_model, temperature=temperature, first_available_dim=-3
    )  # avoid conflict with data plate
    trace = poutine.trace(inferred_model).get_trace(sequences, lengths)
    return_keys =[key for key in trace.nodes.keys() if 'x_' in key]
    
    out_assignments = {key: trace.nodes[key]["value"] for key in return_keys}
    assignments_all = torch.concat([item for name, item in out_assignments.items()], axis=1)
    
    return  assignments_all





def evaluate_model(data, trained_guide, trained_model, args, results_dict={}):
    '''
    Takes in trained model and computes 
    (1) MAP rule assignments: computed per path + per node (majority vote)
    (2) If synthetic: adjusted rand score on per path and per node (majority vote) 
    (3) If synthetic gt available: computes hmm similarity score
    (4) If synthetic and if 2-rule system: computes oracle threshold (between rule1 and rule 2)
         and computes assignments based on this + adj rand per path + per node
    
    returns results_dict updated to include evaluation metrics + assignments
    '''
    # save down guide params that will be used for eval
    results_dict['trained_guide_params_eval'] = trained_guide()
    print('Guide params used',results_dict['trained_guide_params_eval'])


    # MAP rule assignments on test set
    sequences = data["test"]["sequences"].to(args.device)
    lengths = data["test"]["sequence_lengths"].to(args.device)
    mask = data["test"]['mask'].to(args.device)
    assignments_all = classifier(sequences, lengths, trained_guide, trained_model)
    
    
    
    df_test = pd.DataFrame({
                 'idno_endbpid': data['test']['idno_endbpid'][mask.bool().cpu().numpy()], 'assignments_all':assignments_all[mask.bool()].cpu().numpy()})
    print('# rules predicted', len(df_test))
    
    if args.synthetic:
        df_test['rules'] = data['test']['rules'][mask.bool()].cpu().numpy()

    # Adjusted Rand score - per path 
    sequences_for_scoring = data['test']['sequences'][mask.bool()].reshape(-1, 3).cpu().numpy()
    df_test['alpha1'] = sequences_for_scoring[:,0]*180
    df_test['alpha2'] = sequences_for_scoring[:,1]*180
    df_test['phi'] = sequences_for_scoring[:,2]*180
    
    df_test = df_test.loc[df_test.idno_endbpid.str.len()>1]
    if args.synthetic:
        adj_rand_all_paths = adjusted_rand_score(df_test.rules.values, df_test.assignments_all.values)
        print('Adj rand all', adj_rand_all_paths)
    else:
        adj_rand_all_paths = None
    
    
    # Adjusted Rand score - per node (majority voting)
    if args.synthetic:
        grouped = df_test.groupby('idno_endbpid').agg({'rules':lambda x:list(x)[0],
                                                    'assignments_all': lambda x:[Counter(list(x)).most_common()[0][0], len(x), x],
                                                    'alpha1':lambda x:list(x)[0],
                                                    'alpha2':lambda x:list(x)[0],
                                                    'phi':lambda x:list(x)[0]

                                                    })
    else:
        grouped = df_test.groupby('idno_endbpid').agg({'assignments_all': lambda x:[Counter(list(x)).most_common()[0][0], len(x), x],
                                                    'alpha1':lambda x:list(x)[0],
                                                    'alpha2':lambda x:list(x)[0],
                                                    'phi':lambda x:list(x)[0]

                                                    })


    grouped['majority_score'] = grouped.assignments_all.apply(lambda x:x[0])
    grouped['num_scores'] = grouped.assignments_all.apply(lambda x:x[1])

    if args.synthetic:
        adj_rand_majority =adjusted_rand_score(grouped.rules.values, grouped.majority_score.values)
        print('Adj rand majority', adj_rand_majority)
    else:
        adj_rand_majority = None


    if args.synthetic:
        # compute oracle threshold if a 2-rule system
        if (data['gt_info']['mu'].shape[0] ==2):
            if torch.all(data['gt_info']['mu'][0] == data['gt_info']['mu'][0]):
                results_dict['oracle'] = None
            else:
            
                cutoff = get_oracle_threshold(data)

                grouped['binary_sep_consensus']=((grouped[['alpha1', 'alpha2', 'phi']] < cutoff).sum(axis=1)==3).astype(int)
                grouped['binary_sep_avg']=((grouped[['alpha1', 'alpha2', 'phi']] < cutoff).sum(axis=1)>1.5).astype(int)
                per_node_oracle_consensus_rand_score = adjusted_rand_score(grouped.rules.values, grouped.binary_sep_consensus.values)
                per_node_oracle_avg_rand_score = adjusted_rand_score(grouped.rules.values, grouped.binary_sep_avg.values)
                print('Adj rand oracle per_node consensus ={}, average={}'.format( per_node_oracle_consensus_rand_score, per_node_oracle_avg_rand_score))

                df_test['binary_sep_consensus']=((df_test[['alpha1', 'alpha2', 'phi']] < cutoff).sum(axis=1)==3).astype(int)
                df_test['binary_sep_avg']=((df_test[['alpha1', 'alpha2', 'phi']] < cutoff).sum(axis=1)>1.5).astype(int)
                per_path_oracle_consensus_rand_score = adjusted_rand_score(df_test.rules.values, df_test.binary_sep_consensus.values)
                per_path_oracle_avg_rand_score = adjusted_rand_score(df_test.rules.values, df_test.binary_sep_avg.values)
                print('Adj rand oracle per_path consensus ={}, average={}'.format( per_path_oracle_consensus_rand_score, per_path_oracle_avg_rand_score))

                results_dict['oracle'] = {
                    'threshold': cutoff,
                    'path':
                    {
                    'consensus':per_path_oracle_consensus_rand_score, 
                    'avg':per_path_oracle_avg_rand_score
                    },
                    'node': 
                    {
                    'consensus':per_node_oracle_consensus_rand_score, 
                    'avg':per_node_oracle_avg_rand_score
                    }

                }
        else:
            results_dict['oracle'] = None

    else:
        results_dict['oracle'] = None
    
    pprint(results_dict['oracle'])
    results_dict['df_with_assignments_per_path'] = df_test
    results_dict['df_grouped_majority_votes'] = grouped
    results_dict['adj_rand_score_majority'] = adj_rand_majority
    results_dict['adj_rand_score_all_paths'] = adj_rand_all_paths

    results_dict['assignments_test'] = assignments_all.cpu().numpy()

    
    if args.synthetic:
        # compute hmm similarity score
        gt_info = data['gt_info']
        hmm_score, Q_matrix, Se_matrix= _score_hmm(gt_info, trained_guide)
        print("HMM score", hmm_score, Q_matrix, Se_matrix)
        results_dict['gt_hmm_similarity'] = hmm_score
        results_dict['gt_info'] = gt_info
        results_dict['gt_hmm_Q'] = Q_matrix
        results_dict['gt_hmm_Se'] = Se_matrix



    return results_dict



def run_inference(data,args):
    '''
    Runs inference in pyro using MAP Baum Welch
    param data: dict - input data (train / test keys)
    param args: args 
    returns model: trained pyro model
    returns guide: trained pyro guide
    returns data_out: dictionary contains results of inference
    
    '''
    print(args)
    model = models[args.model]
    print(
        "Training {} on {} sequences".format(
            model.__name__, len(data["train"]["sequences"])
        )
    )
    sequences = data["train"]["sequences"].to(args.device)
    lengths = data["train"]["sequence_lengths"].to(args.device)
    
    num_observations = float(lengths.sum())
    pyro.set_rng_seed(args.seed)
    pyro.clear_param_store()
    
    model = models[args.model]

    guide = AutoDelta(
        handlers.block(model, expose_fn=lambda msg: msg["name"].startswith("probs_"))
    )

    # Bind non-PyTorch parameters 
    model = functools.partial(model, args=args)
    guide = functools.partial(guide, args=args)

    # Enumeration requires a TraceEnum elbo and declaring the max_plate_nesting.
    optimiser = optim.Adam({"lr": args.learning_rate})
    

    Elbo = infer.TraceEnum_ELBO
        
    max_plate_nesting = 2
    elbo = Elbo(
        max_plate_nesting=max_plate_nesting,
        strict_enumeration_warning=True,
        jit_options={"time_compilation": True},
    )
    svi = infer.SVI(model, guide, optimiser, elbo)

    # We'll train on small minibatches.
    # return info from run
    
    losses = []
    tm_list, mu_list,std_list, delta_tm_list, delta_mu_list, delta_std_list = [], [], [],[], [],[]
    epoch_stopped=args.num_steps
    mu_old = 0
    tm_old = 0
    std_old = 0
    pbar = tqdm(range(args.num_steps), desc='Training model')
    for step in pbar:
        
        losses.append(svi.step(sequences, lengths,args=args))
            
        tm_new = guide.func.median()['probs_x'].detach().cpu()
        mu_new = guide.func.median()['probs_y'].detach().cpu()
        std_new = guide.func.median()['probs_z'].detach().cpu()
        
        mu_list.append(mu_new)
        std_list.append(std_new)
        tm_list.append(tm_new)
        
        change_tm = torch.abs(tm_new-tm_old).sum().item()
        change_mu = torch.abs(mu_new-mu_old).sum().item()
        change_std = torch.abs(std_new-std_old).sum().item()
        
        delta_tm_list.append(change_tm)
        delta_mu_list.append(change_mu)
        delta_std_list.append(change_std)
        
        pbar.set_description("Exp ({: >5d},{: >5d}) Step: {: >5d}\tLoss: {:.2f}\tDelta_params: {:.2f},{:.2f},{:.2f} ".format(
            args.R, 
            args.seed,
            step, 
            losses[-1] / num_observations,
            change_mu,
            change_std, 
            change_tm))
        
        mu_old = mu_new
        tm_old = tm_new
        std_old = std_new
    
        if args.early_stopping:
            if step > 200:
                if np.mean(losses[-10:]) > np.mean(losses[-20:-10]):
                    print("Losses have converged at step", step)
                    epoch_stopped = step
                    break
    

    # Evaluate  model on the test dataset.
    print(
        "Evaluating on {} test sequences".format(len(data["test"]["sequences"]))
    )
    sequences = data["test"]["sequences"].to(args.device)
    lengths = data["test"]["sequence_lengths"].to(args.device)
    mask = data["test"]['mask'].to(args.device)
    num_observations = float(lengths.sum())
    
    # Process full test set in one batch
    print("Num obs test", num_observations)
    
    test_loss = elbo.loss(
        model,
        guide,
        sequences,
        lengths,
        batch_size=sequences.shape[0],
        include_prior=False,
    )
    print("test loss = {}".format(test_loss / num_observations))

    # Compute model capacity
    capacity = sum(
        value.reshape(-1).size(0) for value in pyro.get_param_store().values()
    )
    print("model_{} capacity = {} parameters".format(args.model, capacity))
    
    # Compute bic
    bic_manual = -2*test_loss + capacity*torch.log(lengths.sum())
    print("bic score for capacity {}, num obs {} = {}".format(capacity, lengths.sum(), bic_manual))

    # Compute assignments
    

    data_out = {'model_capacity': capacity, 
                'final_test_score':test_loss,
                'losses': losses,
                'mu':mu_list, 
                'std': std_list, 
                'tm': tm_list,
                'delta_mu': delta_mu_list, 
                'delta_tm': delta_tm_list, 
                'delta_std': delta_std_list,
                'bic_manual':bic_manual, 
                'last_epoch':epoch_stopped}
    
    data_out = evaluate_model(data, guide, model, args, results_dict=data_out)
    
    return model, guide, data_out




def main(args):
    data = torch.load(args.exp_data_path)
    print('Loaded data for experiment from {}'.format(args.exp_data_path))
        
    key = args.save_key
    scores = []
    results_dicts_all= [] 
    for dim in args.hidden_dims:
        for seed in args.seeds:
            print(f"Starting dim {dim}, seed {seed}")
            # args = parser.parse_args('')
            args.hidden_dim=dim
            args.seed=seed
            # check batch size is smaller / equal to length of data train
            min_batch_size = data['train']['sequence_lengths'].shape[0]
            if args.batch_size > min_batch_size:
                args.batch_size = min_batch_size
            
            trained_model, trained_guide, results_dict = run_inference(data, args)

            #Save results
            torch.save(trained_model, args.save_dir +f'/trained_model_{args.hidden_dim}_{args.seed}_{key}.pt')
            torch.save(trained_guide, args.save_dir +f'/trained_guide_{args.hidden_dim}_{args.seed}_{key}.pt')
            torch.save(results_dict, args.save_dir +f'/results_dict_{args.hidden_dim}_{args.seed}_{key}.pkl')

            results_dict['data'] = data
            results_dict['guide_path'] =args.save_dir + f'/trained_guide_{args.hidden_dim}_{args.seed}_{key}.pt'
            results_dict['model_path'] = args.save_dir +f'/trained_model_{args.hidden_dim}_{args.seed}_{key}.pt'
            
            results_dicts_all.append((args.hidden_dim, args.seed, results_dict))

    torch.save(results_dicts_all,  args.save_dir +f'/final_results_{key}.pkl')
  
    print('Saved results for experiment {} in {}'.format(key, args.save_dir))


















if __name__ == '__main__':
    
    today = date.today()
    d4 = today.strftime("%b-%d-%Y")

    save_key_default = f'results_{d4}'

    parser = argparse.ArgumentParser(
        description="Running Pyro MAP Baum-Welch"
    )
    parser.add_argument(
        "--exp_data_path",
        type=str,
        help="Absolute path to location of experiment data dictionary produced by process_for_inference.py",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="3",
        type=str,
        help="one of: {}".format(", ".join(sorted(models.keys()))),
    )
    parser.add_argument(
        "-R",
        "--hidden_dim",
        default=2,
        type=int,
        help="# hidden Markov states in model",
    )

    parser.add_argument(
        "--device",
        type=str,
        default='cuda:1',
        help="Device name cuda:X or cpu",
    )
    parser.add_argument(
        "--R",
        type=str,
        default=2,
        help="Number of hidden Markov states in model",
    )

    parser.add_argument("-n", "--num-steps", default=1000, type=int)
 
    parser.add_argument("--seed",default=0, type=int)
    parser.add_argument("--batch_size",default=500, type=int)
    parser.add_argument("--learning_rate",default=0.05, type=float)
    parser.add_argument("--early_stopping",default=True, type=bool)
    parser.add_argument("--include_prior",default=True, type=bool)
    parser.add_argument('--hidden_dims', type=list, default=[2])
    parser.add_argument('--seeds', type=list, default=[i for i in range(10)])

    parser.add_argument("--save_dir",default=os.getcwd(), type=str)
    parser.add_argument("--save_key",default=save_key_default, type=str)
    parser.add_argument("--synthetic",default=True, type=bool)
    

    args = parser.parse_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        print('Made save dir {}'.format(args.save_dir) )
    
    assert os.path.exists(args.exp_data_path), 'Data path does not exist {}'.format(args.exp_data_path)

    main(args)
