# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 11:36:55 2019

@author: fandi
"""
# Experiment 3: showing that the figure of the VL for Isomap is really bad. 
#That basically, sigma-dist of the JL is much much beter, it is basically just a const.


# The setting: k and q fixed. And, the dimension d goes from 20 to 100. Then run Isomap
#to see that for fixed q and k, the sigma-dist is incresing witht he original dimesnion
#But,  for the JL, the sigma-dist is bounded by sqrt(q/k) - regardless of d!
#It should be constant (or, if we run on averge w.r to random input space), it is close to some const in Jl
#but increases in Isomap. 




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

# Random spaces
def generate_Normal_space(size, dim):
    metric_space=np.random.normal(0,1,(size, dim))
    return(metric_space);
    
def generate_Gamma_space(size, dim):
    metric_space=np.random.standard_gamma(3,(size, dim))
    return(metric_space);    
    
def generate_Exponent_space(size, dim):
    metric_space=np.random.standard_exponential((size, dim))
    return(metric_space);
  
    

def generate_mixed_space(size, dim):
    space=np.zeros((size, dim))
    for i in range(size):
        for j in range(dim):
            sdv=np.random.randint(10,15)
            space[i,j]=np.random.normal(0,sdv)
    #print(space)
    return(space);
    
def get_random_space(size, dim):
    space=np.zeros((size, dim))
    for i in range(size):
        sdv=np.random.randint(1,15)
        for j in range(dim):
            space[i,j]=np.random.normal(0,sdv)
    return(space);        
            
    
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
         

    

# Computing sigma-dist values 

def expans(old, new):
    #print("The problem is here?")
    return(new/old);    
   
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
        print("ERROR")
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


#COMMENTS: Code for expirements
    
#Isomap:  input k (we will run on k=20), q (we will run on q=2), 
#range_d (we will run on [700, 1000]) - range of original dimensions, 
#t is the number of times the input space of size d and dim d was chosen.
#The alg, generates X of size d *d, embeds into k dims and computes sigma_q.
#Repeats this t times and computes the average. 
#Do this for each d. Plots the graph: x -s d, and sigma_q is y.



def run_Isomap_dim_range(k, q, range_d, t):
    print("ISOMAP experiment: the dimesnion k is", k, "the q is:", q)
    sigma_dists=[]
    for d in np.nditer(range_d):
        curr_sigma=0.0
        for i in range(t):
            X=get_random_space(d,d)
            original_dists=space_to_dist(X) 
            embedded_X=Isomap_transf(X,k)
            embedded_dists=space_to_dist(embedded_X)
            curr_sigma=curr_sigma+(sigma_q(original_dists, embedded_dists,q))
        sigma_dists.append(curr_sigma/t)
        #print("Current distsorions after embedidng the space of dimension",d,"is", sigma_dists)
    answer=np.array(sigma_dists)
    for_plot=np.around(answer,2)
    #print("ISOMAP:The sigma distortions array is", answer)
    #plt.figure()
    #plt.plot(range_d, for_plot)
    #plt.xlabel("the original dimension d")
    #plt.ylabel("sigma_distortion")
    #plt.title("ISOMAP embedding")
    #plt.show()
    return(for_plot);
    

# COMMENTS: the same as before, for the PCA
def run_PCA_dim_range(k, q, range_d, t):
    print("PCA experiment: the dimesnion k is", k, "the q is:", q,"Repititions t =", t)
    sigma_dists=[]
    for d in np.nditer(range_d):
        curr_sigma=0.0
        for i in range(t):
            X=get_random_space(d,d)
            #X=generate_Normal_space(d,d)
            #print("The rank of the generated space is:", matrix_rank(X))
            #X=np.identity(d)
            #Y=X/math.sqrt(2)
            original_dists=space_to_dist(X)
            #print("the original dists are:", original_dists)
            embedded_X=PCA_transf(X,k)
            #print("The size of the embeded space:", embedded_X.shape)
            embedded_dists=space_to_dist(embedded_X)
            #print("PCA:")
            #how_many_contracted_pairs(original_dists, embedded_dists)
            prelim=sigma_q(original_dists, embedded_dists,q)
            if(prelim==float('inf')):
                print("PCA: l_1 dist=0")
                return(False);
            curr_sigma=curr_sigma+(prelim)
        sigma_dists.append(curr_sigma/t)
        #print("Current distsorions after embedidng the space of dimension",d,"is", sigma_dists)
    answer=np.array(sigma_dists)
    for_plot=np.around(answer,2)
    #print("PCA:The sigma distortions array is", answer)
    #plt.figure()
    #plt.plot(range_d, for_plot)
    #plt.xlabel("the original dimension d")
    #plt.ylabel("sigma_distortion")
    #plt.title("PCA embedding")
    #plt.show()
    return(for_plot);            

# COMMENTS: The same for the JL transform
def run_JL_dim_range(k, q, range_d, t):
    print("JL experiment: the dimesnion k is", k, "the q is:", q, "Repititions t =", t)
    sigma_dists=[]
    for d in np.nditer(range_d):
        curr_sigma=0.0
        for i in range(t):
            X=get_random_space(d,d)
            #X=np.identity(d)
            #Y=X/math.sqrt(2)
            original_dists=space_to_dist(X) 
            embedded_X=JL_transf(X,k)
            embedded_dists=space_to_dist(embedded_X)
            curr_sigma=curr_sigma+(sigma_q(original_dists, embedded_dists,q))
        sigma_dists.append(curr_sigma/t)
        #print("Current distsorions after embedidng the space of dimension",d,"is", sigma_dists)
    answer=np.array(sigma_dists)
    #print("The sigma distortions array for JL is", answer)
    for_plot=np.around(answer,2)
    #print("rounded results:", for_plot)
    #plt.figure()
    #plt.plot(range_d, for_plot)
    #plt.xlabel("the original dimension d")
    #plt.ylabel("sigma_distortion")
    #plt.title("JL embedding")
    #plt.show()
    return(for_plot);


#COMMENTS: plotting the experiment 
def experiment_plot(k, q, range_d, t):
    JL_plot=run_JL_dim_range(k, q, range_d, t)
    Isomap_plot=run_Isomap_dim_range(k, q, range_d, t)
    PCA_plot=run_PCA_dim_range(k, q, range_d, t)
    plt.figure()
    plt.plot(range_d, PCA_plot, color='m', label='PCA')
    plt.plot(range_d, Isomap_plot, 'g', label='Isomap')
    plt.plot(range_d, JL_plot, color='c', label='JL')
    plt.legend(loc="upper right")
    plt.xlabel("input space dimension d", size=13)
    plt.ylabel("sigma distortion", size=13)
    plt.show()
    return;
    
    
    
#GOOD PARAMETERS    
range_d=np.array([700,750,800,900,1000])
experiment_plot(20,1,range_d,10)





# IS sigma-dits robust to outliers? If we apply it on normal random, and then on our random space. 
#Does the value change?

#X_norm=generate_Normal_space(400,400)
#X_random=get_weired_space(400,400)
#
#x_norm_emb=PCA_transf(X_norm, 5)
#x_random_emb=PCA_transf(X_random,5)
#
#normal_result=sigma_q(space_to_dist(X_norm), space_to_dist(x_norm_emb),2)
#random_result=sigma_q(space_to_dist(X_random), space_to_dist(x_random_emb),2)
#print("The normal sigma is:", normal_result, "And the random sigma is:", random_result)


    






























