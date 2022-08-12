

""" Metric spaces: creating, testing for being valid, Euclidean, non-Euclidean, etc.  """


import numpy as np
import scipy
import scipy.spatial





"Verifies the triangle inequality. "
def is_metric_space(dists_matrix):
    [rows,cols]=dists_matrix.shape
    for i in range(rows):
        for j in range(i):
            for k in range(rows):
                if(dists_matrix[i,j] > dists_matrix[i,k] + dists_matrix[k,j]):
                    return(False)
    return(True)
   



"Checking whether an input space is Euclidean."

# Constructs the gram matrix from the distance matrix, for checking PSD
def Gram_matrix_from_dists(dists):
    sq_dists=np.square(dists)
    [rows,cols]=sq_dists.shape
    sq_norms=sq_dists[0] #first row contains squared norms, i.e., distances from x_0=0 vector by assumption    
    norms_matrix=np.tile(sq_norms, (rows, 1))
    sum_norms=np.zeros((rows,cols))
    for i in range(1,cols):
        sum_norms[i]=np.copy(np.full(cols,sq_norms[i]))
    G_matrix=(1/2)*(norms_matrix+sum_norms-sq_dists)
    return(G_matrix)    
    


#Checks if the input matrix is positive semi definite
def is_pos_def(X):
    return (np.all(np.linalg.eigvalsh(X) >= 0));


#input: distance matrix
def is_Euclidean_space(dists):
    return(is_pos_def(Gram_matrix_from_dists(dists))) 
  
    
   
# Recovers the original vectors from the distance matrix (if such exist). 
def isom_Eucl_embedding(dists):
    G_matrix=Gram_matrix_from_dists(dists)
    if(~(is_pos_def(G_matrix))):
        return(None)
    L=np.linalg.cholesky(G_matrix)
    original_vectors=np.vstack([np.zeros(np.shape(L)[1]),L])
    return(original_vectors);    
  


"Computing l_p distance matrices, from a given vector space/"

def space_to_dist(space):
    dist=scipy.spatial.distance.pdist(space,metric='euclidean')
    matrix_dist=scipy.spatial.distance.squareform(dist)
    #answer=np.around(matrix_dist,8)
    #print("The distances are", matrix_dist)
    return (matrix_dist);


def infty_space_to_dist(space):
    dist=scipy.spatial.distance.pdist(space,metric='chebyshev')
    matrix_dist=scipy.spatial.distance.squareform(dist)
    return matrix_dist;

def space_to_lp_dists(space,par):
    dist=scipy.spatial.distance.pdist(space, 'minkowski', p=par)
    matrix_dist=scipy.spatial.distance.squareform(dist)
    answer=np.around(matrix_dist,8)
    return(answer);



"Genearting random spaces."


#Retruns a randomly generated vector space, of size and dimesnion dim, as numpy 2D array
def get_random_space(size, dim):
    space=np.random.randn(size, dim)
    for i in range(size):
        sdv=np.random.randint(1,30)
        space[i]=sdv*space[i]
    return(space);








""" Generates a non-Euclidean metric space that is "epsilon-close" to a given Euclidean space. 

Input: distance matrix of a given Euclidean metric space, numpy 2D array. 
Output: distance matrix of a non-Euclidean space, numpy 2D array
        The resulting metric space is an epsilon close to the input space, i.e., can be embedded into it with 
        distortion 1+epsilon.
        The algorithm is randomized and it always outputs a valid metric space.
        
        There is some small probability that the output space is still a Euclidean space 
        (if the input space was Euclidean). 
        To reduce this probability we run the algorithm for several iterations (# T)

NOTE: for some runs, the result of is_metric_space(output) can result in False, due to rounding issues. """

"TODO: make it more efficient"

def get_epsilon_close_metric(dists_matrix, epsilon, T):
    copy_dists_matrix=np.copy(dists_matrix)
    [rows, cols]=dists_matrix.shape

    #The distorted metric space
    generated_metric_dists=np.zeros((rows, cols))
    for i in range(rows):
        for j in range(i+1, rows):
            lower_range=[]
            upper_range=[]
            for k in range(rows):
                if (k!=i and k!=j):
                    min_z=min(copy_dists_matrix[i,k], copy_dists_matrix[j,k])
                    max_z=max(copy_dists_matrix[i,k], copy_dists_matrix[j,k])
                    lower_range.append(max_z-min_z)
                    upper_range.append(max_z+min_z)
                    continue
            lower_array=np.array(lower_range)
            upper_array=np.array(upper_range)
            min_new=np.amax(lower_range)
            max_new=np.amin(upper_range)
            r=copy_dists_matrix[i,j]
            Finish=False
            possible_new_dists=[]
            for t in range(T):
                noise=np.random.normal(0, epsilon)
                if (noise>=0):
                    factor=1+noise
                else:
                    factor=1/(1-noise)
                r_new=factor*r
                if (r_new>=min_new and r_new<=max_new):
                    Finish=True
                    possible_new_dists.append(r_new)
            if(Finish==True):
                new_dist=possible_new_dists[0]
            else:
                new_dist=min_new
            generated_metric_dists[i,j]=new_dist
            generated_metric_dists[j,i]=new_dist
            copy_dists_matrix[i,j]=new_dist
            copy_dists_matrix[j,i]=new_dist
    return(generated_metric_dists)


 





#COMMENTS: from our random space, loop the above code until you get a non-Euclidean result
def get_random_epsilon_close_non_Eucl(n, epsilon):
    original=get_random_space(n,n)
    original_Eucl_dists=space_to_dist(original)
    distorted_dists=get_epsilon_close_metric(original_Eucl_dists, epsilon, 5)
    while(is_Euclidean_space(distorted_dists**2)==True):
        original=get_random_space(n,n)
        original_Eucl_dists=space_to_dist(original)
        distorted_dists=get_epsilon_close_metric(original_Eucl_dists, epsilon, 2)
    return(distorted_dists)











