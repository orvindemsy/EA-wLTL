# Import necessary library
from scipy.linalg import sqrtm, inv 
import numpy as np

# Apply Euclidean Alignment
def apply_EA(data, key_list):
    '''
    Apply Euclidean aligment on array-like objects for 1 subject
    
    PARAMETER:
    data:
    dictionary data of one subject.
    dictionary containing either:
    - left and right EEG data of each subject 
    - combined all_trials
    
    key_list:
    keys inside dict in which data that is about to be aligned are stored
    
    OUTPUT:
    dictionary data with aligned version of key_list
    '''
    
    # So that this function can handles separated or combined left and right trials
    # If they are separated
    if len(key_list) > 1:
        # Concatenate left and right class
        print('Found %d key(s) in which EEG data is stored' %len(key_list))
        left_key = [key for key in key_list if 'left' in key]
        right_key = [key for key in key_list if 'right' in key]

        left_trial = data[left_key[0]]
        right_trial = data[right_key[0]]

        # Concate both left and right trial
        all_trials = np.concatenate([left_trial, right_trial], axis=0)
    # If they are not separated
    else:
        print('Found %d key(s) in which EEG data is stored' %len(key_list))
        all_trials = data[key_list[0]]
    
    # Calculate reference matrix
    RefEA = 0
    print('Computing reference matrix RefEA')

    # Iterate over all trials, compute reference EA
    for trial in all_trials:
        cov = np.cov(trial, rowvar=True)
        RefEA += cov

    # Average over all trials
    RefEA = RefEA/all_trials.shape[0]
    
    # Adding reference EA as a new key in data
    print('Add RefEA as a new key in data')
    data['RefEA'] = RefEA 
    
    # Compute R^(-0.5)
    R_inv = sqrtm(inv(RefEA))
    data['R_inv'] = R_inv
    
    # Again here, they way we stored the key is distinguised
    if len(key_list) > 1:
        # Perform alignment on each trial
        X_EA_left = []
        X_EA_right = []

        for left, right in zip(left_trial, right_trial):
            X_EA_left.append(R_inv@left)
            X_EA_right.append(R_inv@right)

        # Returning both aligned left and right
        return np.array(X_EA_left), np.array(X_EA_right)
    
    # If they are not separated
    else:           
        # Perform EA on each trial
        all_trials_EA = []
        
        for t in all_trials:
            all_trials_EA.append(R_inv@t)
        
        # Return all_trials_EA
        return np.array(all_trials_EA)
        