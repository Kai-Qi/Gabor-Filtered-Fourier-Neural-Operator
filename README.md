# Gabor-Filtered-Fourier-Neural-Operator
Gabor-Filtered Fourier Neural Operator for Solving Partial Differential Equations (GFNO and AGFNO)

This is the code of paper "Gabor-Filtered Fourier Neural Operator for Solving Partial Differential Equations"

The Fourier Neural Operator (FNO) solves a family of Partial Differential
Equations (PDEs) by learning the nonlinear mapping from the parameter space
to the solution space. In this study, we enhance FNO by integrating learnable \textit{Gabor-Filtered module}, leading to the Gabor-Filtered Fourier Neural Operator (GFNO). 
 The frequency transformation matrix is pivotal to the expressiveness of FNO. 
 In the proposed \textit{Gabor-Filtered module},  
the parametric Gabor filters provide regularization to the elements of the
frequency transformation matrix, and it assigns the higher weights to the key frequencies in the frequency domain.
This enables the frequency transformation matrix to reduce redundancies and emphasize the key frequency bands.
Our evaluation, covering five different PDEs and a Climate modeling challenge, shows that GFNO outperforms the original FNO. 
Compared with FNO, GFNO have average error reductions in $8.3\%, 26.3\%$, and $28.7\%$ on solving 1-d Burgers’ equation, 2-d Darcy Flow equation, and 3-d (2-d + time) Navier-Stokes equation, respectively.
As a version of GFNO with fewer number of feature channels, GFNO-small uses merely $19.5\%$, $39.1\%$, and $64.1\%$ of the number of parameters of FNO. 
Despite having fewer parameters, GFNO-small reduces errors by $24.9\%$, $21.3\%$, and $16.3\%$ compared to FNO on solving the three mentioned equations, respectively.
