package algorithms

import (
	"gonum.org/v1/gonum/mat"
	"sync"
)

// RkRek returns the min-norm least-squares solution of a system A*b=y
//
// Parameters:
// U and V are two matrices such that U*V=A.
// y is a vector such that A*b=y.
// iterations is the maximum number of iterations the algorithm is allowed to perform.
// tolerance is the desired error.
// keepErrors is an optional boolean specifying whether to keep the errors calculated at each step.
//
// Returns the b vector which is the solution to A*b=y and an array of errors.
//
// Notes:
// Pass a negative number as the iterations number to default to 100_000.
// Only the first value in keepErrors is evaluated.
// If keepErrors[0] is false then the returned errors array will be empty.
func RkRek(U, V *mat.Dense, y *mat.VecDense, iterations int, tolerance float64, keepErrors ...bool) (mat.VecDense, []float64) {
	// STEP 0.
	// Initialization of variables
	if iterations < 0 {
		iterations = 100_000
	}

	rowsU, colsU := U.Dims()
	rowsV, colsV := V.Dims()

	Utr := mat.NewDense(colsU, rowsU, nil)
	Utr.Copy(U.T())

	x := mat.NewVecDense(colsU, nil)
	z := mat.NewVecDense(rowsU, nil)
	z.CopyVec(y)
	b := mat.NewVecDense(colsV, nil)

	var errors []float64

	// STEP 1.
	// Compute the squared frobenius norm of U, V and U transposed
	frobeniusU := FrobeniusSquared(U)
	frobeniusV := FrobeniusSquared(V)
	frobeniusUtr := FrobeniusSquared(Utr)

	// STEP 2.
	// Compute the probability of choosing a row from U, V and Utr(probability for each column of U)
	probsU := make([]float64, rowsU)
	probsV := make([]float64, rowsV)
	probsUtr := make([]float64, colsU)

	waitGroup := new(sync.WaitGroup)
	waitGroup.Add(3)
	go GetRowsProbability(probsU, frobeniusU, U, rowsU, waitGroup)
	go GetRowsProbability(probsV, frobeniusV, V, rowsV, waitGroup)
	go GetRowsProbability(probsUtr, frobeniusUtr, Utr, colsU, waitGroup)
	waitGroup.Wait()

	// STEP 3.
	// Main algorithm routine. Choosing random rows and updating z, x and b vectors
	for i := 0; i < iterations; i++ {
		randU := GetRandomRow(probsU)
		randV := GetRandomRow(probsV)
		randUtr := GetRandomRow(probsUtr)

		chosenU := U.RowView(randU)
		chosenV := V.RowView(randV)
		chosenUtr := Utr.RowView(randUtr)

		euclideanU := EuclideanNormSquared(chosenU)
		euclideanV := EuclideanNormSquared(chosenV)
		euclideanUtr := EuclideanNormSquared(chosenUtr)

		z.AddScaledVec(
			z,
			-mat.Dot(chosenUtr, z)/euclideanUtr,
			chosenUtr)

		x.AddScaledVec(
			x,
			(y.AtVec(randU)-z.AtVec(randU)-mat.Dot(chosenU, x))/euclideanU,
			chosenU)

		b.AddScaledVec(
			b,
			(x.At(randV, 0)-mat.Dot(chosenV, b))/euclideanV,
			chosenV)

		if keepErrors[0] {
			errVec := new(mat.VecDense)
			errVec.MulVec(V, b)
			errVec2 := new(mat.VecDense)
			errVec2.MulVec(U, errVec)
			errVec2.SubVec(errVec2, y)
			errors = append(errors, EuclideanNormSquared(errVec2))
			if errors[i] <= tolerance {
				break
			}
		}
	}

	return *b, errors
}
