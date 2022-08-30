
"""
Created on Wed August 10 13:15:37 2022

@author: Ora Fandina
"""
import numpy as np
import matplotlib.pyplot as plt
#Classic embedding methods, to compare with
from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


#Local imports: metric spaces and qaulity measures of embedding methods
import distortion_measures as dm
import metric_spaces as ms
import approx_algo as AA


""" Setting up various embeddings  """

def JL_transf(space, k):
    """ Input: 
              space: vectors to embed 
              k: new dimesnion       
    
    Returns embedded vector space"""
    transformer = GaussianRandomProjection(k)
    return transformer.fit_transform(space)


def PCA_transf(space, k):
    transformer = PCA(n_components=k, whiten = False, svd_solver='full')
    return transformer.fit_transform(space)


def TSNE_transf(space, k): 
    transformer = TSNE(n_components=k, init='random', learning_rate='auto', method='exact')
    return transformer.fit_transform(space)





""" Experiments loops   """

def run_dim_range_experiment(input_dists, range_k, q, measure_type, embedding_type, T=10):
    
    """   

    Applies an embedidng on input_dists. Embeds into dimesnions in range_k array. Computes distortion of rank q.

    Inputs: 
        
           input_dist: distance matrix of an input metric space 
     
            range_k:  n array of dimesnions to embed into 
            
            q:  distortion rank 
            
            measure_type:  a string from {'lq_dist', 'REM', 'sigma', 'energy', 'stress'}
            
            embedding_type:  a string from {'PCA', 'TSNE', 'JL', 'Approx_Algo'}, an embedding to apply
            
            T: the number of repetitions, if the applied embedding has a randomness (JL, TSNE)
            
    """ 
    answer=np.zeros(range_k.shape)
    
    measure_dict={
     'lq_dist': dm.lq_dist,
     'REM': dm.REM_q,
     'sigma': dm.sigma_q,
     'energy': dm.energy, 
     'stress': dm.stress 
    }
    
    measure=measure_dict[measure_type]
   
    embedding_dict={
            'PCA': PCA_transf,
            'TSNE': TSNE_transf,
            'JL': JL_transf,
            'Approx_Algo': AA.Approx_Algo
    } 
    embedding=embedding_dict[embedding_type]
    print('Experiment: embedding with', embedding_type, ', measuring with', measure_type)

    
    if (embedding_type=='PCA' or embedding_type=='JL' or embedding_type=='TSNE'):
        input_space=ms.space_from_dists(input_dists)
        for i in range(len(range_k)):
            distortion=0
            times=0
            if(embedding_type=='PCA'):
                times=1
            else: 
                times=T
            for t in range(times):
                distortion+=measure(input_dists, ms.space_to_dist(embedding(input_space, range_k[i])),q)
            answer[i]=distortion/times
            
    else:
        distortion=0
        for i in range(len(range_k)):
            embedded=embedding(input_dists, range_k[i], q, measure_type)
            answer[i]=measure(input_dists, ms.space_to_dist(embedded),q)
    return(answer)    
         

# Plotting the results 

def results_plot(range_k, distorts_embedding_list, measure_type,order):
    plt.figure()
    for i in range(len(distorts_embedding_list)):
        distorts_to_plot=np.around(distorts_embedding_list[i][0],2)
        label_str=distorts_embedding_list[i][1]
        plt.plot(range_k, distorts_to_plot, label=label_str)
    
    plt.legend(loc="upper right")
    plt.xlabel("new dimension")
    plt.ylabel(measure_type+': q='+str(order))
    plt.title("Comparing embedding methods: synthetic data")
    plt.show()
    return;






""" A basic experiment, synthetic data.  

Sample a random space containing 100 point of dimension 100. Embed it into 10, 15 and 20 dimesnions
with PCA, JL and Approx_Algo algorithms. Compare energy_2 error. Note that in this setup Approx_Algo and JL
should return essentially the same reuslt, beating the other PR methods as we formally prove in the paper.

"""

q=2
measure_type='energy'

range_k=np.array([10,15,20])
space=ms.get_random_space(100,100)
dists=ms.space_to_dist(space)      
JL_distorts=run_dim_range_experiment(dists, range_k, q, measure_type, 'JL')
PCA_distorts=run_dim_range_experiment(dists, range_k, q, measure_type, 'PCA')
Approx_distorts=run_dim_range_experiment(dists, range_k, q, measure_type, 'Approx_Algo') 

distorts=[JL_distorts, PCA_distorts, Approx_distorts]

embeddings=['JL','PCA','Approx_Algo']
dist_emb_list=list(zip(distorts, embeddings))
results_plot(range_k, dist_emb_list, measure_type,q)
  



""" Synthetic data set, comparing our Approx Algo with PCA (commonly applied on non-Euclidean data).
    Sample an epsilon-close non_Euclidean space containing 100 points. Apply PCA and Approx_Algo, embedding into
    10, 15 and 20 dimensions. Compare l2-distortions.
 """ 
q=2
measure_type='stress'
dists=ms.get_random_epsilon_close_non_Eucl(n=50, epsilon=0.8)

range_k=np.array([3,5,7])
Approx_distorts=run_dim_range_experiment(dists, range_k, q, measure_type, 'Approx_Algo')

PCA_distorts=run_dim_range_experiment(dists, range_k, q, measure_type, 'PCA')
distorts=[Approx_distorts,PCA_distorts]
embeddings=['Approx_Algo','PCA']
dist_emb_list=list(zip(distorts, embeddings))
results_plot(range_k, dist_emb_list, measure_type,q)



      
"""  Embedding real data sets. Just play with different data sets available in torchvision, for example.
  """
  

#LOADING MNIST to numpy 
from torchvision import datasets

k,q=4,2
train_set = datasets.MNIST('./data', train=True, download=True)

train_set_array=train_set.data.numpy()
space_all=np.reshape(train_set_array, (60000, 784))
space=space_all[:1000,:785] #extracting the first 1000 points for the data set
dists=ms.space_to_dist(space)
new_space=AA.Approx_Algo(dists, k, q)



 

