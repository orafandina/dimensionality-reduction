

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
        
#TSNE

def run_TSNE_range_k(d, range_k, q, t, epsilon):
    answer=np.zeros(range_k.shape)
    for i in range(t):
        #print("PCA Iteration t=",i)
        distorted_space=ms.get_random_epsilon_close_non_Eucl(d, epsilon)(d, epsilon)
        #print("euclidean metric space?", is_Euclidean_space(distorted_space))
        #X=get_random_space(d,d)
        #X=get_Normal_space(d,d)
        #X=get_Gamma_space(d,d)
        #distorted_space=space_to_dist(X)
        distorts=[]
        for i in range(range_k.size):
            #print("The dim k=",range_k[i])
            #print("The type of k is",type(k))
            embedded=TSNE_transf(distorted_space,range_k[i])
            #print("This step is done")
            lq_value=dm.lq_dist(distorted_space, ms.space_to_dist(embedded),q)
            distorts.append(lq_value)
        lq_array=np.array(distorts)
        answer=answer+lq_array
    av_distorts=np.true_divide(answer,t)
    print("The array of MDS lq-dists is:", av_distorts)
    return(av_distorts);


#Our APPROX Algorithm
def run_Approx_range_k_lqdist(d, range_k, q, T, epsilon):
    answer=np.zeros(range_k.shape)
    for i in range(T):
        distorted_space=ms.get_random_epsilon_close_non_Eucl(d, epsilon)(d, epsilon)
        distorts=[]
        for k in np.nditer(range_k):
            embedded=AA.Approx_Algo(distorted_space, k, q)
            lq_value=dm.lq_dist(distorted_space, ms.space_to_dist(embedded),q)
            distorts.append(lq_value)
        lq_array=np.array(distorts)
        answer=answer+lq_array
    av_distorts=np.true_divide(answer,T)
    print("The array of lq-dists is:", av_distorts, "for the dimensions", range_k)
    return(av_distorts)





































