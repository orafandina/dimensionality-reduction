
""" 

Created on July 2021, by Ora Fandina.

Metric spaces: creating, testing for being valid, Euclidean, non-Euclidean, etc.  """


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
   
       
# Constructs the gram matrix from the squared distance matrix, for checking PSD, the input is already squared 
def Gram_matrix_from_dists(sq_dists):
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
    return(is_pos_def(Gram_matrix_from_dists(dists**2))) 
  

def space_from_dists(input_dists, squared=False):
    """ 
    Returns the metric space, such that input_dists is its pairwise Euclidean distance matrix.
    Input: 
        
        input_dists: pairwise distances, assumed to be Euclidean distances
                 
        squared: bool, indicates if the distances are squared  
        
    Rises warning if the input matrix is not PSD     
               
     """       
    if(squared==True):
        dists=input_dists
    else:
        dists=np.square(input_dists)
    Gram=Gram_matrix_from_dists(dists)
    if(~is_pos_def(Gram)):
        print('Warrning: metric_spaces: space_from_dists The distance matrix is non-Euclidean, an approximation will be returned.')
    return(space_from_Gram(Gram, is_pos_def(Gram)))
 

def space_from_Gram(Gram_matrix, is_PSD=True):
    eig_vals, eig_vectors=np.linalg.eigh(Gram_matrix)
    if(is_PSD==False):
        sqrt_eigs=np.sqrt(np.abs(eig_vals))
    else:
        sqrt_eigs=np.sqrt(eig_vals)
    D_matrix=np.diag(sqrt_eigs)
    #The rows of U should be the orthonormal basis of the eig_vectors.
    U_matrix=np.transpose(eig_vectors)
    the_vectors=np.matmul(D_matrix, U_matrix)
    #The original vectors are the cols of the above matrix.ss
    return (np.transpose(the_vectors));


def space_to_dist(space):
    "Comoutes l_p distance matrices, from a given vector space"
    
    dist=scipy.spatial.distance.pdist(space,metric='euclidean')
    matrix_dist=scipy.spatial.distance.squareform(dist)
    #answer=np.around(matrix_dist,8)
    #print("The distances are", matrix_dist)
    return(matrix_dist);


def infty_space_to_dist(space):
    dist=scipy.spatial.distance.pdist(space,metric='chebyshev')
    matrix_dist=scipy.spatial.distance.squareform(dist)
    return matrix_dist;

def space_to_lp_dists(space,par):
    dist=scipy.spatial.distance.pdist(space, 'minkowski', p=par)
    matrix_dist=scipy.spatial.distance.squareform(dist)
    answer=np.around(matrix_dist,8)
    return(answer);


"Genearting synthetic random spaces."

def get_random_space(size, dim):
    """Retruns a randomly generated vector space, of size and dimesnion dim, as numpy 2D array
  """
    
    space=np.random.randn(size, dim)
    for i in range(size):
        sdv=np.random.randint(1,30)
        space[i]=sdv*space[i]
    return(space);


def get_epsilon_close_metric(dists_matrix, epsilon, T):
    """ Generates a non-Euclidean metric space that is "epsilon-close" to a given Euclidean space. 

    Input: distance matrix of a given Euclidean metric space, numpy 2D array. 
    Output: distance matrix of a non-Euclidean space, numpy 2D array
            The resulting metric space is epsilon-close to the input space, i.e., can be embedded into it with 
            distortion 1+epsilon.
            The algorithm is randomized and it always outputs a valid metric space.
            
            There is some small probability that the output space is still a Euclidean space 
            (if the input space was Euclidean). 
            To reduce this probability we run the algorithm for several iterations (# T)

    NOTE: for some runs, the result of is_metric_space(output) can result in False, due to rounding issues. """
    
    copy_dists=np.copy(dists_matrix)
    [rows, cols]=dists_matrix.shape

    #The distorted metric space
    generated_metric=np.zeros((rows, cols))
    
    for i in range(rows):
        for j in range(i+1, cols):
            upper_range=copy_dists[i]+copy_dists[j]
            lower_range=np.abs(copy_dists[i]-copy_dists[j])
            #considel only triangles i,j,k with all different vertices
            excptIndx=[i,j]
            mask=np.ones(cols, dtype=bool)
            mask[excptIndx]=False
            largest_value=np.amin(upper_range[mask])
            smallest_value=np.amax(lower_range[mask])
            r=copy_dists[i][j]
            #need to randomly pick a number from [smallest, largest] which is 1+-eps close to r
            t=1
            noise=np.random.normal(0, epsilon)
            if (noise>=0):
                factor=1+noise
            else:
                factor=1/(1-noise)
            r_new=factor*r
            while((r_new< smallest_value or r_new >largest_value) and t<=T):
                 t+=1
                 noise=np.random.normal(0, epsilon)
                 if (noise>=0):
                     factor=1+noise
                 else:
                     factor=1/(1-noise)
                 r_new=factor*r
            if(r_new>=smallest_value and r_new <=largest_value):
                new_dist=r_new
            else:
                r_new=smallest_value
            generated_metric[i,j]=new_dist
            generated_metric[j,i]=new_dist
            copy_dists[i,j]=new_dist
            copy_dists[j,i]=new_dist
    return(generated_metric)


 
#loop the above code until you get a non-Euclidean space us a result.
#Returns distance matrix of the generated space. Bounded to execute at most 50 tries, enough with high probability.
def get_random_epsilon_close_non_Eucl(n, epsilon):
    Tries=1
    original=get_random_space(n,n)
    original_Eucl_dists=space_to_dist(original)
    distorted_dists=get_epsilon_close_metric(original_Eucl_dists, epsilon, 5)
    while(is_Euclidean_space(distorted_dists) and Tries<=50):
        Tries+=1
        print('trying out a new while loop')
        original=get_random_space(n,n)
        original_Eucl_dists=space_to_dist(original)
        distorted_dists=get_epsilon_close_metric(original_Eucl_dists, epsilon, 2)
    if(Tries==50):
        print('Warrning: get_random_epsilon_close_non_Eucl -- might have generated a Euclidean space, verify.')
    return(distorted_dists)











