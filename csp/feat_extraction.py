import numpy as np

def compute_Z(W, E, m):
    '''
    Will compute the Z
    Z = W @ E, 
    
    E is in the shape of N x M, N is number of electrodes, M is sample
    In application, E has nth trial, so there will be n numbers of Z
    
    Z, in each trial will have dimension of m x M, 
    where m is the first and last m rows of W, corresponds to smallest and largest eigenvalues
    '''
    Z = []
    
    W = np.delete(W, np.s_[m:-m:], 0)
    
    for i in range(E.shape[0]):
        Z.append(W @ E[i])
    
    return np.array(Z)


def feat_vector(Z):
    '''
    Will compute the feature vector of Z matrix
    
    INPUT:
    Z : projected EEG shape of T x N x S
    
    OUTPUT:
    feat : feature vector shape of T x m
    
    T = trial
    N = channel
    S = sample
    m = number of filter
    '''
    
    feat = []
    
    for i in range(Z.shape[0]):
        var = np.var(Z[i], ddof=1, axis=1)
        varsum = np.sum(var)
        
        feat.append(np.log10(var/varsum))
        
    return np.array(feat)


def true_label(feat, hand='left'):
    '''
    Will generate true label of left hand 0 or right hand 1 
    according to number of trial
    
    Parameters
    ----------
    feat: numpy array of shape T x m where T is no. of trials, m is the number of filter
    
    hand: either left or right hand corresponds to imagery task
    
    Return
    ------
    label: array of ones or zero, if hand is right or left respectively
    
    '''
    if hand == 'left':
        label = np.ones([len(feat), 1])*0
    else:
        label = np.ones([len(feat), 1])*1
        
    return label