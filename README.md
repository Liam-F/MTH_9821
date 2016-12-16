# MTH_9821
Baruch MFE MTH 9821 Numerical Methods for Finance

Copyright 2016 @ Yuchen Qi, All rights reserved

Thanks to professor Dan Stefanica and teaching assistant Kris Joanidis who designed all the course material and assignments.

## Getting started
- Python 2.7 interpreter, if you wish to use Python 3.0, you only have to change only the print commands which are everywhere in this repo.
- Numpy package with version 1.11.0 or above is a must
- Scipy package is used to generate normal cdf

## What is in the repo
- Basically all the algos needed for MTH 9821
- Linear Algebra
- Black-Scholes pricer
- Trees pricer
- Monte Carlo pricer
- Finite Difference pricer

### Linear algebra
Numpy.linalg has much efficient implementation for all the algos here, this is only for class exercise

#### NMF_linear_solve
- Forward and backward substitution
- LU decomposition and Cholesky decomposition
All with banded variants

#### NMF_iter_solve
Iterative methods to solve linear system
- Gauss-Sidel
- Jacobi
- Successive over relaxation(SOR)
All with banded variants

#### Option and Barrier_Option
Implementation of a basic option class and a barrier option class (derived from option class)

#### NMF_Black_Scholes
Price a plain vanilla European option with Black-Scholes formula and calculate Greeks

#### Binomial Tree and Trinomial Tree
NMF_Binomial_Tree, NMF_Fast_Binomial_Tree, NMF_Trinomial_Tree, NMF_Fast_Trinomial_Tree
- Contains tree pricing method and Greeks calculation
- Binomial tree has an implied volatility calculating method using secant method
- Use the Fast variant if able. Plain vanilla American and European optoins are supported

#### Monte Carlo methods
NMF_RND is a pseudo random number generator for uniform random variables or normal random variables

Use numpy.random module instead if able
- Read Control_Variates.py to get an idea how this works, it is not well orgnized because of the assignment
- NMF_Monte_Carlo contains a simple pricer for plain vanilla options
- It is not likely to be completed

#### Finite Difference methods
NMF_Heat_PDE is the solver of heat pde, it requires NMF_linear_solve and NMF_iter_solve to work
- Includes Forward Euler(fast but not accurate), Backward Euler(balance speed and accuracy), and Crank-Nicolson(slow but accurate) methods to solve the heat pde numerically

NMF_Finite_Difference is the option price solver, it requires NMF_Heat_PDE to work

Has variants for barrier options and options paying discrete dividends
