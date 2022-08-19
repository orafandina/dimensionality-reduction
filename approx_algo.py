import numpy as np
import cvxpy as cp
from sklearn.random_projection import GaussianRandomProjection



def JL_transf(space, k):
    transformer = GaussianRandomProjection(k)
    result=transformer.fit_transform(space)
    return(result)


""" 
Input: 
    
    input_dists: 2D array containing pairwise distances of metric space to embed
    
    new_dim: int, a number of dimensions to embed into
    
    q: int, the rank of measure objective
    
Returns vectors in the new dimensions, preserving pairwise distances on average.  

"""

def Approx_Algo(input_dists, new_dim, q):
    [rows, cols]=input_dists.shape
    #Step1: convex optimization

    #Normalize the input metric by the largest dsiatnce (this should not change the optimal embedding,
    #but this helps to speed up the computations and make them more precise).
    max_dist=np.amax(input_dists)
    div_input_dists=np.divide(input_dists, max_dist)

    G=cp.Variable((rows-1,rows-1), PSD=True)

    #Z i sthe matrix of the new dists, squared
    Z=cp.Variable((rows,rows),symmetric=True)

    #E is the matrix of expansions, squared
    E=cp.Variable((rows, rows), symmetric=True)

    #C is the matrix of contractions, squared
    C=cp.Variable((rows, cols), symmetric=True)
    C=cp.inv_pos(E)

    #M is the matrix of distortions, squared
    M=cp.Variable((rows, cols),symmetric=True)
    M=cp.maximum(E, C)


    one=cp.Parameter()
    one.value=1

    #the constraints describe the convex boundary set
    constraints=[]
    for j in range(1,rows):
        constraints=constraints+[Z[0,j]==G[j-1,j-1]]

    for i in range(1, rows):
        for j in range(i+1,rows):
            constraints=constraints+[Z[i,j]==G[i-1,i-1]+G[j-1,j-1]-2*G[i-1,j-1]]

    for i in range(rows):
        for j in range(i+1,rows):
            constraints=constraints+[Z[i,j]>=0]

    for i in range(rows):
            constraints=constraints+[E[i,i]==one]


    for i in range (rows):
        for j in range (i+1, rows):
            constraints=constraints+[Z[i,j]==E[i,j]*(div_input_dists[i,j]**2)]


    #The optimization objective function is l_q-distortion.
    if((q/2)==1):
        prob=cp.Problem(cp.Minimize(cp.norm1(M)),constraints)
    else: 
        prob=cp.Problem(cp.Minimize(cp.Pnorm(M, p=q/2)),constraints)
    prob.solve()

    #After the optimization step, the matrix G contains the optimal pairwise Euclidean distances
    #approximating original pairwise distances. 

    #Recovering the resulting vectors of the embedding from the distances, by computing eigenvalue decomposition of G.
    eig_vals, eig_vectors=np.linalg.eigh(G.value)
    sqrt_eigs=np.sqrt(eig_vals)
    D_matrix=np.diag(sqrt_eigs)
    
    #The rows of U should be the orthonormal basis of the eig_vectors.
    U_matrix=np.transpose(eig_vectors)
    the_vectors=np.matmul(D_matrix, U_matrix)

    #The original vectors are the cols of the above matrix.ss
    recov_vecs=np.transpose(the_vectors)

    #The assumption is that the first vector is mapped to 0 vector. So we bring it back.
    vectors=np.vstack([recov_vecs, np.zeros(np.shape(recov_vecs)[1])])
   

    #Note:  We could use the Cholesky decomposition of python,
    #but there are floating point issues, so we implemented our own decomposition.

    #Step 2: embed the high dimimensional vectors into vectors of dimension new_dim, with the JL projection.
    #Output is the set of k-dimensional vectors.
 
  
    low_dim_space=JL_transf(vectors, new_dim)

    #Bring the normalization factor back.
    real_low_dim_space=low_dim_space*max_dist
    return(real_low_dim_space);
