# Gabor-Filtered-Fourier-Neural-Operator
Gabor-Filtered Fourier Neural Operator for Solving Partial Differential Equations (GFNO)

This is the code of the paper "Gabor-Filtered Fourier Neural Operator for Solving Partial Differential Equations".

## Abstract
The Fourier Neural Operator (FNO) solves a family of Partial Differential Equations (PDEs) by learning the nonlinear mapping from the parameter space to the solution space. In this study, we enhance FNO by integrating a learnable Gabor-Filtered module, leading to the Gabor-Filtered Fourier Neural Operator (GFNO).  The frequency transformation matrix is pivotal to the expressiveness of FNO. In the proposed Gabor-Filtered module,  the parametric Gabor filters provide regularization to the elements of the frequency transformation matrix, and it assigns the higher weights to the key frequencies in the frequency domain. This enables the frequency transformation matrix to reduce redundancies and emphasize the key frequency bands. Our evaluation, covering five different PDEs and a Climate modeling challenge, shows that GFNO outperforms the original FNO.  Compared with FNO, GFNO has average error reductions of 8.3%, 26.3%, and 28.7% on solving 1-d Burgers’ equation, 2-d Darcy Flow equation, and 3-d (2-d + time) Navier-Stokes equation, respectively. As a version of GFNO with fewer number of feature channels, GFNO-small uses merely 19.5%, 39.1%, and 64.1% of the number of parameters of FNO.  Despite having fewer parameters, GFNO-small reduces errors by 24.9%, 21.3%, and 16.3% compared to FNO in solving the three mentioned equations, respectively.


## Architecture 
![Image text](https://raw.github.com/yourName/repositpry/master/yourprojectName/img-folder/test.jpg)




## Environment
python 3.8, torch 1.10, CUDA 11.3

## Dataset
(1) The dataset of 1d Burgers’ equation, 2-d Darcy Flow equation, and 3-d (2-d + time) Navier-Stokes equation:

Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., and Anandkumar A., “Fourier Neural Operator for Parametric Partial Differential Equations”, ICLR, 2021. doi:10.48550/arXiv.2010.08895.


(2) The dataset of Mechanical MNIST:

E. Lejeune, Mechanical mnist: A benchmark dataset for mechanical metamodels, Extreme Mechanics Letters 36 (2020) 100659.

(3) The dataset of Shallow Water equation, Climate modeling:

G. Kissas, J. H. Seidman, L. F. Guilhoto, V. M. Preciado, G. J. Pappas, P. Perdikaris, Learning operators with coupled attention, Journal of
Machine Learning Research 23 (2022) 1–63.


## Acknowledgement
Our model is based on the original work of Fourier Neural Operators:

https://github.com/zongyi-li/fourier_neural_operator





