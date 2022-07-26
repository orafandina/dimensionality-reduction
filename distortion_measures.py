import sys 
import numpy as np
from numpy import linalg as LA
import math
import warnings

# Computing distortion measures 

#Expansion of a pair of points
def expans(old, new):
  if (old==0): 
     sys.exit("Expansion error: division by 0")
  return(new/old);

#Contraction of a pair of points
def contr(old, new):
    if(new==0):
       sys.exit("Contraction error: division by 0");
    return(old/new);

#Distortion of a pair
def distortion(old, new):
    expansion=expans(old, new)
    contraction=contr(old,new)
    return(max(expansion, contraction))


#Returns a vector of expans and a vector of contracts. The vectors are compressed to have only (n choose 2) distances
def distorts_vectors(input_dists, embedded_dists):
    mask=np.tri(input_dists.shape[0],input_dists.shape[0],-1,bool)
    old_dists=input_dists[mask]
    new_dists=embedded_dists[mask]
    if 0 in new_dists:
      sys.exit('Distortion: There is a pair that has been contracted to 0')  
    contracts=old_dists/new_dists
    expans=new_dists/old_dists
    return(contracts, expans)

#Worst case distortion. input_dists and embedded_dists are matrices of distances (dtype numpy array) 
def wc_distortion(input_dists, embedded_dists):
    contracts_v, expans_v=distorts_vectors(input_dists,embedded_dists)
    contract=np.amax(contracts_v)
    expans=np.amax(expans_v)
    return(contract*expans)
  
   
#l_q distortion measure
def lq_dist(input_dists, embedded_dists, q):
    contracts_v, expans_v=distorts_vectors(input_dists,embedded_dists)
    distorts=np.maximum(contracts_v, expans_v)
    pairs=len(distorts)
    return(LA.norm(distorts, ord=q)/(pairs**(1/q)))

 
#Other Average Distortion Measures: used in Approx Algorithm     

#Not for use, just for definition of the REM measure
def rem(old, new):
    dist=distortion(old, new)
    rem_dist=abs(dist-1)
    return(rem_dist);



def REM_q(input_dists, embedded_dists, q):
    contracts_v, expans_v=distorts_vectors(input_dists,embedded_dists)
    distorts=np.maximum(contracts_v, expans_v)
    pairs=len(distorts)
    rem_v=distorts-np.ones((pairs,))
    return(LA.norm(rem_v, ord=q)/(pairs**(1/q)))
    
    

#Defined with r=1
def sigma_q(input_dists, embedded_dists, q):
    contracts_v, expans_v=distorts_vectors(input_dists, embedded_dists)
    pairs=len(contracts_v)
    av_expans=LA.norm(expans_v, 1)/pairs
    answ_v=(expans_v/av_expans)-np.ones((pairs,))
    return(LA.norm(answ_v, ord=q)/(pairs**(1/q)))
    

def energy(input_dists, embedded_dists, q):
    contracts_v, expans_v=distorts_vectors(input_dists, embedded_dists)
    pairs=len(contracts_v)
    return(LA.norm(expans_v-np.ones((pairs,)), ord=q)/(pairs**(1/q)))
  
  
def stress(input_dists, embedded_dists, q):
    additive_err=LA.norm(input_dists-embedded_dists, ord=q)
    return(additive_err/LA.norm(input_dists,ord=q))





    
    
 






