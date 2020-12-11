import numpy as np
import sys
sys.path.append('D:\Google_Drive\JupyterNotebookProjects\bci-research\similarity-siamese\csp')

from csp.utils import subject_counter
from scipy.signal import firwin, freqs, lfilter


def fir_bandpass(numtaps, low, high, fs):
    fnyq = fs/2
    b = firwin(numtaps, np.array([low, high])/fnyq, pass_zero = 'bandpass')
    
    return b

# This function will apply bandpass filter to raw EEG data
def apply_bandpass(raw_EEG, b):
    '''
    INPUT:
    raw_EEG : EEG data in the shape of S x N
    b : coefficient of band-pass filter
    
    OUTPUT:
    EEG_filtered : filtered EEG data shape S x N
    
    N : number of channel
    S : number of sample
    '''
    raw_EEG = lfilter(b, 1, raw_EEG, axis=0)
    
    return raw_EEG


def fetch_left_right_EEG(data, ori_data, start=0.5, end=3.5, fs=250, ns=10):
    '''
    In this work only fetch the left and right data from BCICV2a 
    
    Parameters
    ----------
    
    Return 
    ------
    '''
    
    # Grab the position of EEG that corresponds to left '769' and right '770'
    for subj in data.keys():
        print('Processing for ', subj)
        data[subj]['left_pos'] = ori_data[subj]['epos'][ori_data[subj]['etyp'] == 769]
        data[subj]['right_pos'] = ori_data[subj]['epos'][ori_data[subj]['etyp'] == 770]
        
        # Temporary variable of left and right pos    
        temp_pos_left = data[subj]['left_pos']
        temp_pos_right = data[subj]['right_pos']
    
        temp_EEG_left = []
        temp_EEG_right = []
        
        # LEFT
        for j in range(len(temp_pos_left)):
            temp_EEG_left.append(data[subj]['EEG_filtered']\
                                 [temp_pos_left[j] + int(start*fs) : temp_pos_left[j] + int(end*fs)].T)
        
        data[subj]['EEG_left'] = np.array(temp_EEG_left)
        
        # RIGHT
        for j in range(len(temp_pos_right)):
            temp_EEG_right.append(data[subj]['EEG_filtered']\
                                 [temp_pos_right[j] + int(start*fs) : temp_pos_right[j] + int(end*fs)].T)
            
        data[subj]['EEG_right'] = np.array(temp_EEG_right)
        
        
    return data


def split_EEG_one_class(EEG_one_class, percent_train=0.8):
    '''
    split_EEG_one_class will receive EEG data of one class, with size of T x N x M, where
    T = number of trial
    N = number of electrodes
    M = sample number
    
    PARAMETER
    ---------
    EEG_data_one_class: the data of one class of EEG data
    
    percent_train: allocation percentage of training data, default is 0.8
    
    RETURN
    ------
    EEG_train: EEG data for training
    
    EEG_test: EEG data for test
    
    Both have type of np.arrray dimension of T x M x N
    '''

    # Number of all trials
    n = EEG_one_class.shape[0]
    
    n_tr = round(n*percent_train)
    n_te = n - n_tr

    
    EEG_train = EEG_one_class[:n_tr]
    EEG_test = EEG_one_class[n_tr:n_tr+n_te]
        
    return EEG_train, EEG_test


def process_s_data(data, eeg_key='EEG_filtered', start_t=0.5, end_t=3.5, fs=250):
    '''
    Parameter
    data:
    Dictionary of data of one subject
    
    key:
    This will be the key in the data, in which EEG data is being stored, shape of samples x n_electrodes
    The data inside this key will be splitted into no_trials x n_electrodes x samples
    

    Return
    all_trials:
    data containing all trials of that subject, shape of no_trials x n_electrodes x samples
    
    y:
    the true label of each trial
    
    all_pos:
    starting point of each trials
    '''
    # Event type and position of subject
    typ = data['etyp']
    pos = data['epos']

    # Grab position of each left (etype=769) and right (etype=770) class
    pos_left = pos[typ==769]
    pos_right = pos[typ==770]
    all_pos = np.hstack([pos_left, pos_right])

    # True label
    y_left = np.zeros(len(pos_left))
    y_right = np.ones(len(pos_right))
    all_y = np.hstack([y_left, y_right])

    # Sort them ascendingly based on event occurences
    ids = np.argsort(all_pos)
    all_pos = all_pos[ids]
    all_y = all_y[ids]

    fs=250

    # Now convert 's' data into data of trials
    s = data[eeg_key]

    all_trials = []
    for pos in all_pos:
        all_trials.append(s[(pos-1)+int(fs*start_t) : (pos-1)+int(fs*end_t)].T)

    all_trials = np.array(all_trials) 

    # Return these
    return all_trials, all_y, all_pos
