import numpy as np
import cvxpy as cp
from sklearn.random_projection import GaussianRandomProjection

import metric_spaces as ms
import distortion_measures as dm

def JL_transf(space, k):
    transformer = GaussianRandomProjection(k)
    result=transformer.fit_transform(space)
    return(result)


def Approx_Algo(input_dists, new_dim, q, objective):
    """ Returns vectors in the new dimensions, preserving pairwise distances on average. 

            Input: 
        
                input_dists: 2D array containing pairwise distances of metric space to embed
        
                new_dim: int, a number of dimensions to embed into
        
                q: int, the rank of measure objective
                
                objective: str, objective distortion measure to appproximate. 
                           'lq_dist', optimizing lq_distortion
                           'stress', optimizing the stress distortion
                           'energy', optimizing the energy distortion
                           'sigma', optimizing sigma distortion
                      
    """
 
    [rows, cols]=input_dists.shape
    #Step1: convex optimization

    #Normalize the input metric by the largest dsiatnce, to speed up  
    #(this does not change the optimal embedding)
    max_dist=np.amax(input_dists)
    div_input_dists=np.divide(input_dists, max_dist)
    
     
    if(objective=='lq_dist'):
        recov_vecs=solve_optim_lqdist(div_input_dists, q)
    else:
        if(objective=='energy'):
            recov_vecs=solve_optim_energy(div_input_dists,q)
        else:
            if(objective=='stress'):
                recov_vecs=solve_optim_stress(div_input_dists,q)
            else:
                if(objective=='sigma'):
                    recov_vecs=solve_optim_sigma(div_input_dists,q)
                    
    #Step 2
   
    low_dim_space=JL_transf(recov_vecs, new_dim)

    #Bring the normalization factor back.
    real_low_dim_space=low_dim_space*max_dist
    return(real_low_dim_space);


def solve_optim_lqdist(input_dists,q):
    [rows, cols]=input_dists.shape
    G=cp.Variable((rows,cols), PSD=True)
    
    #Z is the matrix of the new dists, squared
    Z=cp.Variable((rows,cols),symmetric=True)
    #E is the matrix of expansions, squared
    E=cp.Variable((rows, cols), symmetric=True)

    #C is the matrix of contractions, squared
    C=cp.Variable((rows, cols), symmetric=True)
    C=cp.inv_pos(E)

    #M is the matrix of distortions, squared
    M=cp.Variable((rows, cols),symmetric=True)
    M=cp.maximum(E, C)

    
    #z_ij >=0 and z_0j== <v_j,v_j> and a tecnichal constr. 
    constraints=[Z>=0, Z[0]==cp.diag(G), cp.diag(E)==1]
    #z_ij== (expans_ij)^2*(old_ij)^2
    constraints+=[Z==cp.multiply(E, (input_dists)**2)]
   
    #z_ij=G_ii+G_jj-2G_ij, vectorized
    G_expression=cp.Variable((rows, cols))
    
    for i in range(rows):
       constraints+=[G_expression[i]==cp.diag(G)]
    
    constraints=constraints+[G_expression.T + G_expression - 2*G==Z]
       
    #optimization objective function, l_q-distortion
    if((q/2)==1):
        prob=cp.Problem(cp.Minimize(cp.norm1(M)),constraints)
    else: 
        prob=cp.Problem(cp.Minimize(cp.Pnorm(M, p=q/2)),constraints)
    print('Solving the cvxpy problem ...')
    prob.solve(solver='SCS')
    print('The problem has been solved, continuing to the RP step')
    recov_vecs=ms.space_from_Gram(G.value, ms.is_pos_def(G.value))
    return(recov_vecs)

def solve_optim_energy(input_dists,q):
    [rows, cols]=input_dists.shape
    
    G=cp.Variable((rows,cols), PSD=True)
    #Z is the matrix of the new dists, squared
    Z=cp.Variable((rows,cols),symmetric=True)
    #E is the matrix of expansions, squared
    E=cp.Variable((rows, cols), symmetric=True)
      
    constraints=[Z>=0, Z[0]==cp.diag(G), cp.diag(E)==1]
    #z_ij== (expans_ij)^2*(old_ij)^2
    constraints+=[Z==cp.multiply(E, (input_dists)**2)]
   
    #z_ij=G_ii+G_jj-2G_ij, vectorized
    G_expression=cp.Variable((rows, cols))
    
    for i in range(rows):
       constraints+=[G_expression[i]==cp.diag(G)]
    
    constraints=constraints+[G_expression.T + G_expression - 2*G==Z]
    
    #matrix of additive distorts, (sqrt(z_ij)/d_ij-1)== sqrt(Expans)-1
    
    obj_expression=cp.abs(E-1)
    
    if(q==1):
        prob=cp.Problem(cp.Minimize(cp.norm1(obj_expression)),constraints)
    else:
        prob=cp.Problem(cp.Minimize(cp.Pnorm(obj_expression, p=q)),constraints)
    print('Solving the cvxpy problem ...')
    prob.solve(solver='SCS')
    print('The problem has been solved, continuing to the RP step')
    recov_vecs=ms.space_from_Gram(G.value, ms.is_pos_def(G.value))
    return(recov_vecs)
          
    
def solve_optim_stress(input_dists, q):
    [rows, cols]=input_dists.shape
    
    G=cp.Variable((rows,cols), PSD=True)
    #Z is the matrix of the new dists, squared
    Z=cp.Variable((rows,cols),symmetric=True)
    #E is the matrix of expansions, squared
    E=cp.Variable((rows, cols), symmetric=True)
      
    constraints=[Z>=0, Z[0]==cp.diag(G), cp.diag(E)==1]
    #z_ij== (expans_ij)^2*(old_ij)^2
    constraints+=[Z==cp.multiply(E, (input_dists)**2)]
   
    #z_ij=G_ii+G_jj-2G_ij, vectorized
    G_expression=cp.Variable((rows, cols))
    
    for i in range(rows):
       constraints+=[G_expression[i]==cp.diag(G)]
    
    constraints=constraints+[G_expression.T + G_expression - 2*G==Z]
    
    #matrix of additive distorts, (sqrt(z_ij)/d_ij-1)== sqrt(Expans)-1
    obj_expression=Z-input_dists**2
    if(q==1):
        prob=cp.Problem(cp.Minimize(cp.norm1(obj_expression)),constraints)
    else:
        prob=cp.Problem(cp.Minimize(cp.Pnorm(obj_expression, p=q)),constraints)
    print('Solving the cvxpy problem ...')
    prob.solve(solver='SCS')
    print('The problem has been solved, continuing to the RP step')
    recov_vecs=ms.space_from_Gram(G.value, ms.is_pos_def(G.value))
    return(recov_vecs)
    
       
def solve_optim_sigma(input_dists, q):
    [rows, cols]=input_dists.shape
    num_pairs=rows*(rows-1)/2
    
    G=cp.Variable((rows,cols), PSD=True)
    #Z is the matrix of the new dists, squared
    Z=cp.Variable((rows,cols),symmetric=True)
    #E is the matrix of expansions, squared
    E=cp.Variable((rows, cols), symmetric=True)

    constraints=[Z>=0, Z[0]==cp.diag(G), cp.diag(E)==1]
    #z_ij== (expans_ij)^2*(old_ij)^2
    constraints+=[Z==cp.multiply(E, (input_dists)**2)]
   
    #z_ij=G_ii+G_jj-2G_ij, vectorized
    G_expression=cp.Variable((rows, cols))
    
    for i in range(rows):
       constraints+=[G_expression[i]==cp.diag(G)]
    
    constraints=constraints+[G_expression.T + G_expression - 2*G==Z]
    
    #normalization constraint for sigma 
    constraints=constraints+[cp.sum(E)==num_pairs] 
    
    obj_expression=cp.abs(E-1)
    if(q==1):
        prob=cp.Problem(cp.Minimize(cp.norm1(obj_expression)),constraints)
    else:
        prob=cp.Problem(cp.Minimize(cp.Pnorm(obj_expression, p=q)),constraints)
    print('Solving the cvxpy problem ...')
    prob.solve(solver='SCS')
    print('The problem has been solved, continuing to the RP step')
    recov_vecs=ms.space_from_Gram(G.value, ms.is_pos_def(G.value))
    return(recov_vecs)

    
    
    
    
    
    
    