

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




#metric spaces and qaulit ymeasures of embedding methods
import distortion_measures as dm
import metric_spaces as ms
import approx_algo as AA

import torch
import torchvision

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

"""  TO DO: 1. bring real data sets = MNIST, CIFAR and compare all these embeddings on these.
            2. compare our embedding with PyME embedding?
            3. Rewrite our approx algo using pytorch functionality, to speed it up. Currently it is not practical at all !""" 

""" Setting up various embeddings  """


"""Loading the MNIST data. Mnist_train and mnist_test are pytorch.Tensors after applying transform to Tensor """
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
#MNIST train has 60 000 images with labels
mnist_train = datasets.MNIST('data', train=True, download=True,transform=transform)
#has 10 000 images with labels
mnist_test = datasets.MNIST('../data', train=False, download=True, transform=transform)

train_dl = DataLoader(mnist_train, batch_size=1, shuffle=True)# Init the loader, wich will enumerate over the whole data set in batches, where each batch consists of tupels, photo+label
for i, (xb, yb) in enumerate(train_dl):             #each xb is a tensor containing bs rows, each row is a data point (tensor)
    #xb = xb.to(device)
    #yb = yb.to(device)
    xb = xb.view(xb.size(0), -1) #this turns the batch of data into a tensor, containing bs rows each of length 28*28 (the photo)
    xb=xb.numpy()
    
    
 # COLLECT SOME POINTS INTO AN ARRAY . After apply one of these algos onto this array...   




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
print(run_dim_range_experiment(dists, np.array([3,6,8]), 3, 'lq_dist', 'PCA'))    
        

