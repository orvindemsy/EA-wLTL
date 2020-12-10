import numpy.linalg as la
import numpy as np

def csp_feat_ver5(XtrainRaw, XtestRaw, ytr, n_filter=3):
    '''
    Another adaptatino of MATAB csp feat function, this time function will receive EEG trials raw data
    
    Parameter:
    XtrainRaw : train trials, shape of trials x channels x samples
    XtestRaw : test trials, shape of trials x channels x samples
    ytr : the label class of training data, 1D shape of (trials, )
    n_filter: number of filter for spatial filter
    
    Return:
    feat_train: csp feature training, shapae of samples x 2*n_filters
    feat_test: csp feature training, shape of samples x 2*n_filters

    '''
    
    ids_left = np.argwhere(ytr == 0).ravel()
    ids_right = np.argwhere(ytr == 1).ravel()

    EEG_left = XtrainRaw[ids_left]
    EEG_right = XtrainRaw[ids_right]

    # Covariance of left and right
    cov_left = 0
    for signal in EEG_left:
        cov_left += np.cov(signal, rowvar=True, ddof=1)

    cov_left = cov_left/EEG_left.shape[0]

    cov_right = 0
    for signal in EEG_right:
        cov_right += np.cov(signal, rowvar=True, ddof=1)

    cov_right = cov_right/EEG_right.shape[0]

    mldiv = la.lstsq(cov_right, cov_left, rcond=None)[0]

    # Eigenvector and eigenvalues
    [eigval, eigvec] = la.eig(mldiv)

    # Sort, descending order, eigvec
    ids_dsc = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, ids_dsc]

    # W matrix
    W = np.delete(eigvec, np.s_[n_filter:-n_filter], axis=1)

    # Calculating feature train 
    feat_train = []
    for trial in XtrainRaw:
        X = W.T@trial
        feat_train.append(np.log10(np.diag(X@X.T)/np.trace(X@X.T)) ) 
    
    # Calculating feature test 
    feat_test = []
    for trial in XtestRaw:
        X = W.T@trial
        feat_test.append(np.log10(np.diag(X@X.T)/np.trace(X@X.T)) ) 
    
    return np.array(feat_train), np.array(feat_test)