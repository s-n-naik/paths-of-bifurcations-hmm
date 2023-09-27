
import os, glob
import pandas as pd
import numpy as np
from tqdm import tqdm

from pprint import pprint

import torch
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


smoke_test = ('CI' in os.environ)
assert pyro.__version__.startswith('1.8.4')

pyro.enable_validation(True)



# Paths of bifurcations HMM
def model_3(sequences, lengths, args,batch_size=None, include_prior=None, verbose=False):
    '''
    Sequences has shape num_sequences x max_length x data dim (emission distribution dim)
    
    Lengths has shape num_sequences, lengths has a 0 entry if a sequence is missing
    
    args contains:
        R = # hidden states in HMM (int)
        batch_size = # sequences per batch in plate (int)
     
    
    Batch size: Use to override args.batch_size if not None
    '''
   
    options = dict(dtype=sequences.dtype, device=sequences.device)
    R = args.R
    if include_prior is None:
        include_prior = args.include_prior
    
    if batch_size is not None:
        batch_size = args.batch_size
    
    # print("HMM Inference with R={}, include_prior={}".format(R, include_prior))
    with ignore_jit_warnings():
        num_sequences, max_length, data_dim = map(int, sequences.shape)
        assert lengths.shape == (num_sequences,)
        assert lengths.max() <= max_length
    
    with handlers.mask(mask=include_prior):
        # Start probabilities for HMM
        probs_s = pyro.sample(
            "probs_s",
            dist.Dirichlet(torch.ones((1,R),**options)/R).to_event(1),
        )
        # Transition matrix for HMM
        probs_x = pyro.sample(
            "probs_x",
            dist.Dirichlet(torch.ones((R, R),**options)/R).to_event(1),
        )
        
        # Gaussian emission distributions (diagonal covariance matrix) - means
        probs_y = pyro.sample(
            "probs_y",
            dist.Normal(loc=torch.tensor(0 ,**options), scale=torch.tensor(0.1,**options)).expand([R, data_dim]).to_event(2),
        )
        # Gaussian emission distributions (diagonal covariance matrix)
        # Probs z goes into dist.Normal as the stdev of the distribution
        probs_z = pyro.sample(
        "probs_z", dist.InverseGamma(torch.tensor(1,**options),torch.tensor(0.1,**options)).expand([R, data_dim]).to_event(2),
    )
    
    # For observation distribution
    angles_plate = pyro.plate("angles", data_dim, dim=-1)

    with pyro.plate("sequences", num_sequences, batch_size, dim=-2) as batch:
        x=0
        # Iterate over batch of sequences with lengths in batch
        lengths_batch = lengths[batch]
        
        for t in pyro.markov(range(lengths_batch.max())):
            with handlers.mask(mask=(t < lengths_batch).unsqueeze(-1)):
                # Loop until length of sequence (masked)
                if t == 0:
                    # Draw current rule from start probabilities
                    x =  pyro.sample(
                    "x_{}".format(t),
                    dist.Categorical(probs_s[x]),
                    infer={"enumerate": "parallel"})
                    if verbose:
                        print("x_{}".format(t), x.shape)
                else:
                    # Draw current rule from previous rule from Transition matrix
                    x = pyro.sample(
                    "x_{}".format(t),
                    dist.Categorical(probs_x[x]),
                    infer={"enumerate": "parallel"},
                )
                    if verbose:
                        print("x_{}".format(t), x.shape)
                    
                # Draw observations from current rule
                with angles_plate:
                    
                    y=pyro.sample(
                        "y_{}".format(t),
                        dist.Normal(loc=probs_y[x.squeeze(-1)],scale=probs_z[x.squeeze(-1)]),
                        obs=sequences[batch, t],
                        infer={"enumerate": "parallel"},
                    )
                    if verbose:
                        print("y_{}".format(t), y.shape)




