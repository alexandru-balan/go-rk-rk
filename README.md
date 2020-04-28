# go-ml-linear-systems

This is a package that focuses on solving linear systems of type **Xβ = y** where X is a
low-rank matrix and β and y are two vectors. The β vector is the vector of unknowns
that we want to solve for. The theoretical background implemented in this package can be found 
in the paper: [Iterative methods for solving factorized linear systems
](https://arxiv.org/abs/1701.07453v4)
 
 ## Some basic info to get ready to GO
 
 The idea is to solve the aforementioned system by approximating the optimal result.
 In order to do that we are not going to work with the full system that was mentioned before,
 but with a decomposition of X and two subsystems:
 - **Ux = y**
 - **Vβ = x**
 
 were **X = UV** and U is of size m * k and V is of size k * n full-rank matrices. Thus substituting the second subsystem in the first we get the 
 full system.
 
 There are three possible settings (X a full-rank matrix):
 
 1. X is underdetermined, m < n, then there are indefinitely many solutions and 
 we are interested in the least Euclidean norm solution of the full-system:
 **β<sub>LN</sub> = X<sup>\*</sup> (X X<sup>\*</sup>)<sup>-1</sup>y**.
 
 2. In the overdetermined setting, m > n, the full-system can have one or no solution.
 If there is a solution then, the system is called consistent, and we have:
 **β<sub>unique</sub> = (X<sup>\*</sup> X)<sup>-1</sup>X<sup>\*</sup>y**
 
 3. If the system is inconsistent then we seek to minimize the sum of squared residuals
 **r = Xβ<sub>LS</sub> - y**, where **β<sub>LS</sub> = (X<sup>\*</sup> X)<sup>-1</sup>X<sup>\*</sup>y**
 
 If the X matrix is rank-deficient then there are indefinitely many cases regardless of 
 m and n. Then
 
 **β<sub>LN</sub> = X<sup>\*</sup> (X X<sup>\*</sup>)<sup>+</sup>y**, if m < n
 
 **β<sub>LN</sub> = (X<sup>\*</sup> X)<sup>+</sup>X<sup>\*</sup>y**, if m > n
 
 The algorithms implemented by this package work by interlacing the solving of U and V as described
 in the academical paper.
 