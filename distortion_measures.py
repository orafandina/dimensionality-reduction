import sys 
import numpy as np
from numpy import linalg as LA
import math

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
def distorts_vectors(input_dist, embedded_dist):
    mask=np.tri(input_dist.shape[0],input_dist.shape[0],-1,bool)
    old_dists=input_dist[mask]
    new_dists=embedded_dist[mask]
    if 0 in new_dists:
      sys.exit('Distortion error: There is a pair that has been contracted to 0')
    contracts=old_dists/new_dists
    expans=new_dists/old_dists
    return(contracts, expans)

#Worst case distortion. input_dists and embedded_dists are matrices of distances (dtype numpy array) 
def wc_distortion(input_dist, embedded_dist):
    contracts_v, expans_v=distorts_vectors(input_dist,embedded_dist)
    contract=np.amax(contracts_v)
    expans=np.amax(expans_v)
    return(contract*expans)
  
   
#l_q distortion measure
def lq_dist(input_dist, embedded_dist, q):
    contracts_v, expans_v=distorts_vectors(input_dist,embedded_dist)
    distorts=np.maximum(contracts_v, expans_v)
    pairs=len(distorts)
    return(LA.norm(distorts, ord=q)/(pairs**(1/q)))

 
#Other Average Distortion Measures: used in Approx Algorithm     

#Not dfor use, just for definition of the REM measure
def rem(old, new):
    dist=distortion(old, new)
    rem_dist=abs(dist-1)
    return(rem_dist);



def REM_q(input_dist, embedded_dist, q):
    contracts_v, expans_v=distorts_vectors(input_dist,embedded_dist)
    distorts=np.maximum(contracts_v, expans_v)
    pairs=len(distorts)
    rem_v=distorts-np.ones((pairs,))
    return(LA.norm(rem_v, ord=q)/(pairs**(1/q)))
    
    


def sigma_q(input_dist, embedded_dist, q):
    contracts_v, expans_v=distorts_vectors(input_dist, embedded_dist)
    pairs=len(contracts_v)
    av_expans=LA.norm(expans_v, 1)/pairs
    answ_v=(expans_v/av_expans)-np.ones((pairs,))
    return(LA.norm(answ_v, ord=q)/(pairs**(1/q)))
    


#multiplicative factor: to normalize JL to multiply by this, for faster computations in CVXP
def scaling_factor(input_dist, embedded_dist,q):
    contracts_v, expans_v=distorts_vectors(input_dist, embedded_dist)
    pairs=len(contracts_v)
    return(math.sqrt(LA.norm(contracts_v, ord=q)/(pairs**(1/q))/LA.norm(expans_v, ord=q)/(pairs**(1/q))))
    
    
 






