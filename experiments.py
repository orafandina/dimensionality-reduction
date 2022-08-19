

"""
Created on Wed August 10 13:15:37 2022

@author: Ora Fandina
"""
import matplotlib.pyplot as plt




#Classic embedding methods, to compare with
from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import PCA
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
    """ Input: 
              space: vectors to embed 
              k: new dimesnion       
    
    Returns embedded vector space"""
    transformer = GaussianRandomProjection(k)
    return transformer.fit_transform(space)


def PCA_transf(space, k):
    """ Input: 
              space: vectors to embed 
              k: new dimesnion        
              
    Returns embedded vector space"""
    transformer = PCA(n_components=k, whiten = False, svd_solver='full')
    return transformer.fit_transform(space)




def TSNE_transf(space, k):
    """ Input: 
              space: vectors to embed 
              k: new dimesnion      
              
     Returns embedded vector space"""  
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
            
            measure_type:  a string from {'lq_dist', 'REM', 'sigma'}
            
            embedding_type:  a string from {'PCA', 'TSNE', 'JL', 'Approx_Algo'}, an embedding to apply
            
            T: the number of repetitions, if the applied embedding has a randomness (JL, TSNE)
            
    """ 

    
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
            answer[i]=measure(input_dists, ms.space_to_dist(embedding(input_dists, range_k[i],q)),q)
    return(answer)    
         

# Plotting the results 

def results_plot(range_k, distorts_embedding_list, measure_type):
    plt.figure()
    for i in range(len(distorts_embedding_list)):
        distorts_to_plot=np.around(distorts_embedding_list[i][0],2)
        label_str=distorts_embedding_list[i][1]
        plt.plot(range_k, distorts_to_plot, label=label_str)
    
    plt.legend(loc="upper right")
    plt.xlabel("new dimension")
    plt.ylabel(measure_type)
    plt.title("Comparing embedding methods: synthetic data")
    plt.show()
    return;






""" A basic experiment, synthetic data.  

Create a random space containing 100 point of dimension 100. Embed it into 10, 15 and 20 dimesnions
with PCA, JL and TSNE algorithms. Compare l2-distortions.

"""
range_k=np.array([10,15,20])
space=ms.get_random_space(100,100)
dists=ms.space_to_dist(space)      
JL_distorts=run_dim_range_experiment(dists, range_k, 2, 'lq_dist', 'JL')
PCA_distorts=run_dim_range_experiment(dists, range_k, 2, 'lq_dist', 'PCA')
TSNE_distorts=run_dim_range_experiment(dists, range_k, 2, 'lq_dist', 'TSNE') 

distorts=[JL_distorts, PCA_distorts, TSNE_distorts]

embeddings=['JL','PCA','TSNE']
dist_emb_list=list(zip(distorts, embeddings))
results_plot(range_k, dist_emb_list, 'lq_dits')
  

     
"""  Embedding real data sets.  """
