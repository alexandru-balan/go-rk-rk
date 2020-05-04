package utils

import (
	"gonum.org/v1/gonum/mat"
	"log"
)

func SolveLeastSquares(X *mat.Dense, y *mat.VecDense) mat.VecDense {
	// Create an SVD representation
	svd := new(mat.SVD)
	success := svd.Factorize(X, mat.SVDThin)

	if !success {
		log.Panic("Can't factorize the X into SVD")
	}

	Right := new(mat.Dense)
	Left := new(mat.Dense)
	Values := svd.Values(nil)
	svd.VTo(Right)
	svd.UTo(Left)

	// Computing the least-squares solution
	c := new(mat.VecDense)
	c.MulVec(Left.T(), y)

	d := new(mat.VecDense)
	d.DivElemVec(c.SliceVec(0, len(Values)), mat.NewVecDense(len(Values), Values))

	b := new(mat.VecDense)
	b.MulVec(Right, d)

	return *b
}
