# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 17:45:38 2019

@author: fandi
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 10:03:37 2019

@author: fandina
"""


import matplotlib.pyplot as plt
import numpy as np
from random import gauss
from sklearn.random_projection import GaussianRandomProjection
import scipy

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap
import sys
import math
from numpy.linalg import matrix_rank
from mpl_toolkits.mplot3d import Axes3D   

# Random spaces
def get_Normal_space(size, dim):
    print("The experiment uses Normal space")
    metric_space=np.random.normal(0,1,(size, dim))
    return(metric_space);
    
def get_Gamma_space(size, dim):
    print("The experiment uses Gamma space")
    metric_space=np.random.standard_gamma(3,(size, dim))
    return(metric_space);    
    
def get_Exponent_space(size, dim):
    metric_space=np.random.standard_exponential((size, dim))
    return(metric_space);
  
    
    
def get_random_space(size, dim):
    print("The experiment uses general random space")
    space=np.zeros((size, dim))
    for i in range(size):
        sdv=np.random.randint(1,15)
        for j in range(dim):
            space[i,j]=np.random.normal(0,sdv)
    return(space);        
            
#=============================================================================================    
#General stuff
def space_to_dist(space):
    dist=scipy.spatial.distance.pdist(space,metric='euclidean')
    matrix_dist=scipy.spatial.distance.squareform(dist)
    return matrix_dist;
  
def how_many_contracted_pairs(input_dist, embedded_dist):
    [rows, cols]=input_dist.shape
    pairs=scipy.special.binom(rows, 2)
    contracted=0;
    expanded=0;
    for i in range(rows):
        for j in range(i+1, cols):
            if(expans(input_dist[i,j], embedded_dist[i,j])<=1):
                contracted=contracted+1;
            else:
                expanded=expanded+1
    print("Num of expanded pairs", (expanded/pairs)*100, "Num of contracted pairs", (contracted/pairs)*100)             
    return([expanded/pairs, contracted/pairs]);
         

  #=====================================================================================  

# Computing sigma-dist values 

def expans(old, new):
    #print("The problem is here?")
    return(new/old);    

def contr(old, new):
    if(new==0):
        return(float('inf'));
    return(old/new);    

def distortion(old, new):
    if(contr(old,new)==float('inf')):
       return(float('inf'))
    distort=max(expans(old, new), contr(old, new))
    return(distort);
    
#l_q distortion measure
def lq_dist(input_dist, embedded_dist, q):
    [rows, cols]=input_dist.shape
    answer=0
    pairs=scipy.special.binom(rows, 2)
    for i in range (rows):
        for j in range(i+1,cols):
            curr=distortion(input_dist[i,j], embedded_dist[i,j])
            if(curr== float('inf')):
               print("The pair with index", [i,j],"has been contracted to 0")
               return(float('inf'));
            answer=answer+((curr)**q)
    #print("l_q-dist for q=",q," is:",((answer/pairs))**(1/float(q)))        
    return(((answer/pairs))**(1/float(q)));
  
    
    
#REM distortion measure 
def rem(old, new):
    dist=distortion(old, new)
    if(dist==float('inf')):
      return(float('inf'));
    rem_dist=abs(dist-1)
    return(rem_dist);   
    
    
    
def REM_q(input_dist, embedded_dist, q):
    [rows, cols]=input_dist.shape
    answer=0
    pairs=scipy.special.binom(rows, 2)
    for i in range (rows):
        for j in range(i+1,cols):
            curr=rem(input_dist[i,j], embedded_dist[i,j])
            if(curr== float('inf')):
               #print("The pair with index", [i,j],"has been contracted to 0")
               return(float('inf'));
            answer=answer+((curr)**q)
    #print("REM is:", ((answer/pairs))**(1/float(q)))               
    return(((answer/pairs))**(1/float(q)));  

   
def l1_expans(input_dist, embedded_dist):
    [rows, cols]=input_dist.shape
    answer=0
    #all_expans_list=[]
    pairs=scipy.special.binom(rows, 2)
    for i in range(rows):
        for j in range(i+1, cols):
            curr=expans(input_dist[i,j], embedded_dist[i,j])
            answer=answer+curr
            #all_expans_list.append(curr)
    #print("l1-expans is:", answer/pairs) 
    #the_list=np.array(all_expans_list)
    #print("PCA:The expansions are:", the_list)
    #print("The l_1 expans is:", answer/pairs)
    return(answer/pairs);      

def sigma_q(input_dist, embedded_dist, q):
    [rows, cols]=input_dist.shape
    av_expans=l1_expans(input_dist, embedded_dist)
    if(av_expans==0.0):
        #print("ERROR")
        sys.exit("Oh, the error is there");
    answer=0.0
    pairs=scipy.special.binom(rows, 2)
    for i in range(rows):
        for j in range(i+1,cols):
            exp=expans(input_dist[i,j], embedded_dist[i,j])
            curr=abs((exp/av_expans)-1)
            answer=answer+((curr)**q)
    #print("sigma_q is:",((answer/pairs))**(1/float(q)))
    return(((answer/pairs))**(1/float(q)));        
   

#scaling factor for more efficient JL

def lq_expans(input_dist, embedded_dist,q):
    [rows, cols]=input_dist.shape
    answer=0
    #all_expans_list=[]
    pairs=scipy.special.binom(rows, 2)
    for i in range(rows):
        for j in range(i+1, cols):
            curr=expans(input_dist[i,j], embedded_dist[i,j])
            answer=answer+(curr)**q
    return((answer/pairs)**(1/float(q)));  


def lq_contr(input_dist, embedded_dist,q):
    [rows, cols]=input_dist.shape
    answer=0
    #all_expans_list=[]
    pairs=scipy.special.binom(rows, 2)
    for i in range(rows):
        for j in range(i+1, cols):
            curr=contr(input_dist[i,j], embedded_dist[i,j])
            if(curr==float('inf')):
                sys.exit("Contraction ERROR");
            answer=answer+(curr)**q
    return((answer/pairs)**(1/float(q))); 

#multiplicative factor: to normalize JL you have to multiply by this
def scaling_factor(input_dist, embedded_dist,q):
    contracts=lq_contr(input_dist, embedded_dist,q)
    expans=lq_expans(input_dist, embedded_dist,q)
    alpha= math.sqrt(contracts/expans)
    return(alpha);            

#=============================================================================================
# Transforms

def JL_transf(space, k):
    transformer = GaussianRandomProjection(k)
    result=transformer.fit_transform(space)
    return(result);
    
    
def Isomap_transf(space, k):
    transformer = Isomap(n_components=k,)
    result=transformer.fit_transform(space)
    return(result);

def PCA_transf(space, k):
    transformer = PCA(n_components=k, whiten = False,svd_solver='full')
    result=transformer.fit_transform(space)
    return(result);
    
    
 
#==============================================================================
  
    
#COMMENTS: The experiment for JL and sigma-distortion.
# The algorithm is a s follows:
#  1. Pick a random space (our bad random space) of a given size d times d
#  2. Repeeat the following t times:
#     3. embed X with JL (non-normilized by the optimal factor) into the range of
#        new dimesnions range_k.
#     4. for each embedding compute the sigm - distortion for a given value of q
#     5. output the average (over the t repetitons) of all the sigma-dists    
    
    

def run_JL_range_k_sigma(d, range_k, q, t):
    #X=get_Normal_space(d,d)
    X=get_random_space(d,d)
    answer=np.zeros(range_k.shape)
    for i in range(t):
        distorts=[]
        for k in np.nditer(range_k):
            embedded=JL_transf(X,k)
            lq_value=sigma_q(space_to_dist(X), space_to_dist(embedded),q)
            distorts.append(lq_value)
        lq_array=np.array(distorts)
        answer=answer+lq_array
    av_distorts=np.true_divide(answer,t)
    #print("The array of lq-dists is:", av_distorts)
    return(av_distorts);


#COMMENTS: the same as before, just for the normalized JL
def run_normalized_JL_range_k_sigma(d, range_k, q, t):
    #X=get_Normal_space(d,d)
    X=get_random_space(d,d)
    answer=np.zeros(range_k.shape)
    for i in range(t):
        #print("Repetition t=",i)
        distorts=[]
        for k in np.nditer(range_k):
            #print("Round k=",k)
            embedded=JL_transf(X,k)
            alpha=scaling_factor(space_to_dist(X), space_to_dist(embedded),q)
            #print("The scaling factor is:", alpha)
            embedded*alpha
            lq_value=sigma_q(space_to_dist(X), space_to_dist(embedded),q)
            distorts.append(lq_value)
        lq_array=np.array(distorts)
        answer=answer+lq_array
    av_distorts=np.true_divide(answer,t)
    #print("The array of normalized lq-dists is:", av_distorts)
    return(av_distorts);



#COMMENTS: repeats the JL experiment for T times (over the picing a random space)
def run_JL_exp_sigma(d,range_k, q,t,T):
    #print("The experiment uses a regular version of JL.")
    answ=np.zeros(range_k.shape)
    for i in range(T):
        distorts=run_JL_range_k_sigma(d, range_k, q, t)
        answ=answ+distorts
    av_distorts=np.true_divide(answ, T)
    #print("Sigma distorts of JL:",np.around(av_distorts,2))
    return(av_distorts);

#COMMENTS: the same as before, for the normalized JL
def run_JL_exp_normalized_sigma(d,range_k, q,t,T):
    #print("The experiment uses normalized version of JL.")
    answ=np.zeros(range_k.shape)
    for i in range(T):
        distorts=run_normalized_JL_range_k_sigma(d, range_k, q, t)
        answ=answ+distorts
    av_distorts=np.true_divide(answ, T)
    #print("Normalized JL sigma  distorts:", np.around(av_distorts,2))
    return(av_distorts);

     
    
#===========================================================================
 
#COMMENTS: PCA experiment for sigma-dist, repeated T times (for randomly picking the space)

def run_PCA_range_k_sigma(d, range_k, q, t):
    answer=np.zeros(range_k.shape)
    for i in range(t):
        #print("PCA Iteration t=",i)
        #X=get_Normal_space(d,d)
        X=get_random_space(d,d)
        distorts=[]
        for i in range(range_k.size):
            #print("The dim k=",range_k[i])
            #print("The type of k is",type(k))
            embedded=PCA_transf(X,range_k[i])
            #print("This step is done")
            sigma_value=sigma_q(space_to_dist(X), space_to_dist(embedded),q)
            distorts.append(sigma_value)
        sigma_array=np.array(distorts)
        answer=answer+sigma_array
    av_distorts=np.true_divide(answer,t)
    #print("The array of PCA sigma dists is:", av_distorts)
    return(av_distorts);   
#===========================================================================================   
    
 #COMMENT:ISOMAP the same as above

def run_Isomap_range_k_sigma(d, range_k, q, t):
    answer=np.zeros(range_k.shape)
    for i in range(t):
        #print("PCA Iteration t=",i)
        #X=get_Normal_space(d,d)
        X=get_random_space(d,d)
        distorts=[]
        for i in range(range_k.size):
            #print("The dim k=",range_k[i])
            #print("The type of k is",type(k))
            embedded=Isomap_transf(X,range_k[i])
            #print("This step is done")
            sigma_value=sigma_q(space_to_dist(X), space_to_dist(embedded),q)
            distorts.append(sigma_value)
        sigma_array=np.array(distorts)
        answer=answer+sigma_array
    av_distorts=np.true_divide(answer,t)
    #print("The array of ISOMAP lq-dists is:", av_distorts)
    return(av_distorts); 



#=============================================================================================    

#COMMENTS: Plotting the results all together 

def experiment_plot(d, range_k, q,t,T):
    print("The running experiment is for d=", d, "and for q=",q)
    JL_plot=run_JL_exp_normalized_sigma(d,range_k, q,t,T)
    #JL_plot=run_JL_exp(d,range_k, q,t,T)
    to_plot_JL=np.around(JL_plot,2)
    
    Isomap_plot=run_Isomap_range_k_sigma(d, range_k, q, t)
    to_plot_Isomap=np.around(Isomap_plot, 2)
    
    PCA_plot=run_PCA_range_k_sigma(d, range_k, q, t)
    to_plot_PCA=np.around(PCA_plot,2)
    
    plt.figure()
    plt.plot(range_k, to_plot_PCA, color='m', label='PCA')
    plt.plot(range_k, to_plot_Isomap, 'g', label='Isomap')
    plt.plot(range_k, to_plot_JL, color='c', label='JL')
    plt.legend(loc="upper right")
    plt.xlabel("dimension k", size=13)
    plt.ylabel("sigma distortion", size=13)
    #plt.title("Chosing the number of dimensions k")
    #plt.yticks([1, 10,50])
    #plt.xticks([2,4,6,8,10,12,14,16,18,20])
    plt.show()
    return;    


#COMMENTS: the parameters we choose to run on: d=800, q=10. 
experiment_plot(800, np.array([3,5,8,10,12,14,16,18,20,30]),5, 8,2)

#maybe it is interestinf to see what happens for the natural case of q=2
#experiment_plot(300, np.array([2,4,6,8,10,12,14,16]),2, 4,4)



#also, the same experiment for lq dists, as a recommendation for choosing the right value of k
#=======================================================================
#def run_phase_transition_experiment(d, range_k, q,t,T):
#    print("The running phase transition experiment is for d=", d, "and for q=",q)
#    #JL_plot=run_JL_exp_normalized(d,range_k, q,t,T)
#    for_third_q=run_JL_exp(d, range_k, 12,t,T)
#    to_plot_third_q=np.around(for_third_q,2)
#    
#    JL_plot=run_JL_exp(d,range_k, q,t,T)
#    to_plot_JL=np.around(JL_plot,2) 
#    
#    for_second_q=run_JL_exp(d, range_k, 8,t,T)
#    to_plot_second_q=np.around(for_second_q,2)
#    
#    colors=plt.cm.Blues(np.linspace(0.3, 1.0, 3))
#
#    plt.figure()
#    plt.plot(range_k, to_plot_third_q, color=colors[0], label='q=12')
#    plt.plot(range_k, to_plot_JL, color=colors[1], label='q=10')
#    plt.plot(range_k, to_plot_second_q, color=colors[2], label='q=8')
#    
#    plt.legend(loc="upper right")
#    plt.xlabel("the new dimension k")
#    plt.ylabel("lq distortion")
#    plt.title("Phase transition phenomenon of the JL")
#    #plt.yticks([1, 10,50])
#    #plt.xticks([2,3,4,5,6,8,10,12,14])
#    plt.show()
#    return;        


#run_phase_transition_experiment(300,np.array([4,6,8,10,12,14,16]),10,8,8)





 
    
    
    
    
   
    
    
    
    

    


    
        
        


    