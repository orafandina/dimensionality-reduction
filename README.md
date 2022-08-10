# Dimensionality reduction with provable guarantees


The code is the implementation of the approximation algorithm with provable guarantees, as appears
in our paper [Dimensionality Reduction: theoretical perspective on practical measures](https://proceedings.neurips.cc/paper/2019/file/94f4ede62112b790c91d5e64fdb09cb8-Paper.pdf), NeurIPS 2019. 
<br>
<br>

## Approximation Algorithm 

The algorithm is implemented in `approx_algo.py`.
The input to the algorithm is a finite metric space $X$, given by the matrix of the pairwise distances; an integer $k \geq 3$, denoting the target dimension and parameter $q \geq 1$, denoting the desired moment in the loss function. 

The algorithm computes an embedding $F: X \to \ell_2^k$ into a $k$- dimensional Euclidean space with 
<br>
<br>
> l_q-distortion(F) =(1+O(q/k))*OPT

<br>

where $OPT$ is the $l_q$-distortion of the **optimal** embedding of $X$ into a $k$ - dimensional Euclidean space.

<br>  

The algorithm works in two steps: 

1. FComputes an optimal embedding of $X$ into a high dimensional Euclidean space. The optimality is in the sense of preserving the $l_q$ - distortion. In this step the dimension of the resulting vectors is not restricted, and will be of dimesnion $n$ which is the number of the points in the input metric space $X$. To find such an embedding, we write the appropriate convex optimization program and solve it with the solver implemnted in the cvxpy python package.   

   
2. Next, we aplly the JL projection method to reuce the dimesnion of the output set from the first setp. We embed the vectors into $k$ - dimensions. 
   The projection method used is the Gaussian matrix based, which is crucial and necessary to the guarantees to be true as proved in our paper.

We give here the implementation for optimizing the lq_distortion, while optimizing for the other measures is done similarly.

## Dependencies 

``` pip install cvxpy ```


## Metric Spaces 
`metric_spaces.py` module is for working with metric spaces: randomly generating metric spaces, checking for being a valid metric space, for being a Euclidean and more.

## Distortion Measures
`distortion_measures.py` contains implementation of various distortion measures of an embedding, e.g., $l_q$-distortion, $REM$ and more, as discussed in our paper. 

## Tests
This is the main file, contains experiments for reproducability of the results in the paper. 

## Jupyter notebook for a quck interaction with the code 

The notebook version of the code, with step by step instructions is in `Embedding.ipynb`.



# Citation 
If you use our implementation in your research, please cite our paper:

```
@inproceedings{BFN19,
  title = {Dimensionality reduction: theoretical perspective on practical measures},
  author = {Yair Bartal and Nova Fandina and Ofer Neiman},
  year = {2019},
  booktitle = {NeurIPS 2019},
  }
```




# Code Contributor 
Ora Fandina
