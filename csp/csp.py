import numpy as np
from scipy.linalg import sqrtm, inv

class CSP():
    def __init__(self, data=None):
        self.data = data
        
    def compute_cov(self, EEG_data):
        '''
        INPUT:
        EEG_data : EEG_data in shape T x N x S
        
        OUTPUT:
        avg_cov : covariance matrix of averaged over all trials
        '''
        cov=[]
        
        # EEG_data_left = self.data
        
        for i in range(EEG_data.shape[0]):
            cov.append(EEG_data[i]@EEG_data[i].T/np.trace(EEG_data[i]@EEG_data[i].T))   
        
        cov = np.mean(np.array(cov), 0)
        
        return cov
        
        
    def decompose_cov(self, avg_cov):
        '''
        This function will decompose average covariance matrix of one class of each subject into 
        eigenvalues denoted by lambda and eigenvector denoted by V
        Both will be in descending order
        
        Parameter:
        avgCov = the averaged covariance of one class
        
        Return:
        λ_dsc and V_dsc, i.e. eigenvalues and eigenvector in descending order
        
        '''
        λ, V = np.linalg.eig(avg_cov)
        λ_dsc = np.sort(λ)[::-1] # Sort eigenvalue descending order, default is ascending order sort
        idx_dsc = np.argsort(λ)[::-1] # Find index in descending order
        V_dsc = V[:, idx_dsc] # Sort eigenvectors descending order
        λ_dsc = np.diag(λ_dsc) # Diagonalize λ_dsc
        
        return λ_dsc, V_dsc 
    
    
    def white_matrix(self, λ_dsc, V_dsc):
        '''
        '''
        λ_dsc_sqr = sqrtm(inv(λ_dsc))
        P = (λ_dsc_sqr)@(V_dsc.T)
        
        return P
    
    def compute_S(self, avg_Cov, white):
        '''
        This function will compute S matrix, S = P * C * P.T
    
        INPUT:
        avg_Cov: averaged covariance of one class, dimension N x N, where N is number of electrodes
        white: the whitening transformation matrix
        
        OUTPUT:
        S
        '''
        S = white@avg_Cov@white.T
    
        return S
    
    def decompose_S(self, S_one_class, order='descending'):
        '''
        This function will decompose the S matrix of one class to get the eigen vector
        Both eigenvector will be the same but in opposite order
        
        i.e the highest eigenvector in S left will be equal to lowest eigenvector in S right matrix 
        '''
        # Decompose S
        λ, B = np.linalg.eig(S_one_class)
        
        # Sort eigenvalues either descending or ascending
        if order == 'ascending':
            idx = λ.argsort() # Use this index to sort eigenvector smallest -> largest
        elif order == 'descending':
            idx = λ.argsort()[::-1] # Use this index to sort eigenvector largest -> smallest
        else:
            print('Wrong order input')
        
        λ = λ[idx]
        B = B[:, idx]
        
        return B, λ 
    
    def spatial_filter(self, B, P):
        '''
        Will compute projection matrix using the following equation:
        W = B' @ P
        
        INPUT:
        B: the eigenvector either left or right class, choose one, size N x N, N is number of electrodes
        P: white matrix in size of N x N 
        
        OUTPUT:
        W spatial filter to filter EEG
        '''
        
        return (B.T@P)