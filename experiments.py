

"""
Created on Wed August 10 13:15:37 2022

@author: Ora Fandina
"""
import matplotlib.pyplot as plt




#Classic embedding methods, to compare with
from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.manifold import TSNE



import numpy as np






#metric spaces and qaulit ymeasures of embedding methods
import distortion_measures as dm
import metric_spaces as ms
import approx_algo as AA



""" Setting up various embeddings  """




""" Input: 
          space: numpy array containing vectors of the space
          k: int, number of dimensions to embed into
         
    Returns: 
         embedded vector space """

def JL_transf(space, k):
    transformer = GaussianRandomProjection(k)
    return transformer.fit_transform(space)


def PCA_transf(space, k):
    transformer = PCA(n_components=k, whiten = False,svd_solver='full')
    return transformer.fit_transform(space)




def TSNE_transf(space, k):
    transformer = TSNE(n_components=k, metric='precomputed', method='exact')
    return transformer.fit_transform(space)





""" Experiments loops   """

#MDS;SMACOF

"""   

Applies an embedidng on input_dists. Embeds into dimesnions in range_k array. Computes distortion of rank q.

Inputs: input_dist - distance matrix of an input metric space 
 
        range_k - n array of dimesnions to embed into 
        
        q - distortion rank 
        
        measure_type - a string from {'lq_dist', 'REM', 'sigma'}
        
        embedding_type - a string from {'PCA', 'TSNE', 'JL', 'Approx_Algo'}, an embedding to apply
        
""" 




def run_dim_range_experiment(input_dists, range_k, q, measure_type, embedding_type, T=10):
    answer=np.zeros(range_k.shape)
    
    
    measure_dict={
     'lq_dist': dm.lq_dist,
     'REM': dm.REM_q,
     'sigma': dm.sigma_q
    }
    
    measure=measure_dict[measure_type]
    
    embedding_dict={
            'PCA': PCA_transf,
            'TSNE': TSNE_transf,
            'JL': JL_transf,
            'Approx_Algo': AA.Approx_Algo
    }
    
    embedding=embedding_dict[embedding_type]
    print('Experiment: embedding with', embedding_type)
    
    if (embedding_type=='PCA' or embedding_type=='JL' or embedding_type=='TSNE'):
        input_space=ms.space_from_dists(input_dists)
        distortion=0
        for i in range(len(range_k)):
            for t in range(T):
              distortion+=measure(input_dists, ms.space_to_dists(embedding(input_space, range_k[i])),q)  
            answer[i]=distortion/T    
            
    else:
        distortion=0
        for i in range(len(range_k)):
            answer[i]=measure(input_dists, ms.space_to_dists(embedding(input_dists, range_k[i])),q)
    return(answer)    
         

# Plotting the results 

        
dists=ms.space_to_dist(ms.get_random_space(10,10))
print(run_dim_range_experiment(dists, np.array([3,6,8]), 3, 'lq_dist', 'PCA'))    
        

