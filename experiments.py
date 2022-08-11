

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
from numpy import linalg as LA
import scipy
import scipy.spatial


#convex opt package


#metric spaces and qaulit ymeasures of embedding methods
import distortion_measures as dm
import metric_spaces as ms
import approx_algo as AA


""" Setting up various embeddings  """

#Input: numpy array containing vectors of the space
def JL_transf(space, k):
    transformer = GaussianRandomProjection(k)
    result=transformer.fit_transform(space)
    return(result)


#Input: numpy array containing vectors of the space
def PCA_transf(space, k):
    transformer = PCA(n_components=k, whiten = False,svd_solver='full')
    result=transformer.fit_transform(space)
    return(result)

#Input: distance matrix (not necessarily Euclidean) 
#Output:embeded vectors in k dimesnions
def MDS_transf(original_dists,k):
    transformer=MDS(n_components=k,dissimilarity='precomputed')
    result=transformer.fit_transform(original_dists)
    return(result)


#Input: distance matrix (not necessarily Euclidean) 
#Output:embeded vectors in k dimesnions
def TSNE_transf(dists, k):
    transformer=TSNE(n_components=k, metric='precomputed', method='exact')
    result=transformer.fit_transform(dists)
    return(result)



""" Experiments loops   """

#MDS;SMACOF

"""   

Applies MDS embedidng om input_dists. embeds into dimesnions in range_k array. Computes distortion of rank q.

Inputs: input_dist - distance matrix of an input metric space
 
        range_k - n array of dimesnions to embed into 
        
        q - distortion rank 
        
        measure_type - a string from {'lq_dist', 'REM', 'sigma'}
        
        embedding_type - a string from {'MDS', 'PCA', 'TSNE', 'Approx_Algo'}, an embedding to apply
        
""" 

def run_dim_range_experiment(input_dists, range_k, q, measure_type, embedding_type):
    answer=np.zeros(range_k.shape)
    
    measure_dict={
     'lq_dist': dm.lq_dist,
     'REM': dm.REM_q,
     'sigma': dm.sigma_q
    }
    
    measure=measure_dict[measure_type]
    
    embedding_dict={
            'MDS': MDS_transf,
            'PCA': PCA_transf,
            'TSNE': TSNE_transf,
            'Approx_Algo': AA.Approx_Algo
    }
    
    embedding=embedding_dict[embedding_type]
    print('Experiment: embedding with', embedding_type)
    
    for i in range(len(range_k)):
        if (embedding_type=='Approx_Algo'):
            embedded=embedding(input_dists,range_k[i],q)
        else:
            embedded=embedding(input_dists,range_k[i])
        answer[i]=measure(input_dists,ms.space_to_dist(embedded),q)
    return(answer)    
         
        
dists=ms.space_to_dist(ms.get_random_space(10,10))
print(run_dim_range_experiment(dists, np.array([3,6,8]), 3, 'lq_dist', 'Approx_Algo'))    
        

