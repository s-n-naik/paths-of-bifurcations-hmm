import torch
import os, glob
import numpy as np
import pyro
import numpy as np
import pandas as pd

import torch_geometric as geometric

from sklearn.metrics import adjusted_rand_score



def _score_hmm(gt_info, trained_guide):
    '''
    Compares trained model to the gt info
    Assumes format in guide of model 3 - probs_x, probs_y, probs_z, probs_s variables
    
    '''

    # GT model params
    tm_gt = gt_info['tm']
    mu_gt = gt_info['mu']
    cov_gt = gt_info['cov']
    std_gt = torch.sqrt(cov_gt)

    # Best model params
    tm_hmm =trained_guide.func.median()['probs_x'].cpu()
    # Make sure these are unscaled
    mu_hmm = trained_guide.func.median()['probs_y'].cpu().squeeze()*180
    # start = best_model.startprob_
    std_hmm = (trained_guide.func.median()['probs_z']).cpu()*180
    start_hmm = trained_guide.func.median()['probs_s'].cpu()
    cov_hmm = std_hmm**2

    best_model_info = {'tm':tm_hmm, 'mu':mu_hmm, 'cov':cov_hmm}
    similarity_score, Q_matrix, Se_matrix = _compute_hmm_similarity(gt_info, best_model_info)
    return similarity_score, Q_matrix, Se_matrix


def _compute_hmm_similarity(hmm1, hmm2):
    '''
    hmm1: dict keys = [tm, mu, cov]
    hmm2: dict keys = [tm, mu, cov]
    Make sure that hmm1 and hmm2  have mu and cov on the same scale (ie both degrees)
    '''
    tm_gt, tm_hmm = hmm1['tm'].cpu(),hmm2['tm'].cpu()
    mu_gt, mu_hmm = hmm1['mu'].cpu(),hmm2['mu'].cpu()
    cov_gt, cov_hmm =  hmm1['cov'].cpu(),  hmm2['cov'].cpu()
    
    
    # Compute stationary dist (pi) for both
    evals, evecs = torch.linalg.eig(tm_gt.T)
    evec1 = evecs[:,np.isclose(evals,1)]
    evec1 = evec1[:,0]
    stationary = evec1 / evec1.sum()
    pi_gt = stationary.real

    evals, evecs = torch.linalg.eig(tm_hmm.T)
    evec1 = evecs[:,np.isclose(evals,1)]
    evec1 = evec1[:,0]
    stationary = evec1 / evec1.sum()
    pi_hmm = stationary.real
    print("pi", pi_hmm, pi_gt)
    # correspondence amtrix (# rules hmm, # rules gt)    
    ES_matrix = torch.zeros(mu_hmm.shape[0], mu_gt.shape[0])
    # distance matrix (# rules hmm, # rules gt)
    Se_matrix = torch.zeros(mu_hmm.shape[0], mu_gt.shape[0])

    for i in range(ES_matrix.shape[0]):
        for j in range(ES_matrix.shape[1]):
            p = torch.distributions.MultivariateNormal(mu_hmm[i], torch.eye(3, device='cpu')*cov_hmm[i])
            q= torch.distributions.MultivariateNormal(mu_gt[j], torch.eye(3, device='cpu')*cov_gt[j])
            Se_matrix[i, j] =torch.distributions.kl_divergence(p, q).item()
            ES_matrix[i,j] = pi_hmm[i]*pi_gt[j]*Se_matrix[i,j]
    ES = torch.sum(ES_matrix)
    Q_matrix = ES_matrix/ES
    return geometric.nn.functional.gini(Q_matrix), Q_matrix, Se_matrix



def solve_gaussian_intersection(m1,m2,std1,std2):
    '''
    Get intersection between two gaussians
    '''
    a = 1/(2*std1**2) - 1/(2*std2**2)
    b = m2/(std2**2) - m1/(std1**2)
    c = m1**2 /(2*std1**2) - m2**2 / (2*std2**2) - np.log(std2/std1)
    return np.roots([a,b,c])



def get_oracle_threshold(data):
    '''
    Get intersection between two gaussians
    NOTE: only a single root is returned (if two, the root between the two means is returned)
    
    returns oracle_threshold: tensor of shape (3) with threshold for alpha1, alpha2, phi in degrees
    '''
    if data['gt_info']['mu'].shape[0] ==2:
        oracle_threshold = np.zeros(3)
        mu_gt = data['gt_info']['mu']
        cov_gt = data['gt_info']['cov']
        std_gt = torch.sqrt(cov_gt)

        print(std_gt, mu_gt)
        for i in range(3):
            roots = solve_gaussian_intersection(mu_gt[0][i], mu_gt[1][i],std_gt[0][i], std_gt[1][i])
            if len(roots)>1:
                # get root that lies in between rule means 
                idx_middle_root = torch.where((torch.min(mu_gt[0][i], mu_gt[1][i]) <torch.tensor(roots)) & (torch.tensor(roots) < torch.max(mu_gt[0][i], mu_gt[1][i])))
                root = roots[idx_middle_root]
            else:
                root = roots[0]
            print('Roots {} --> root {}'.format(roots, root))
            oracle_threshold[i] = root
    return oracle_threshold
