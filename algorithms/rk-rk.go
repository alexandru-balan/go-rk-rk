package algorithms

import (
	"gonum.org/v1/gonum/mat"
	"sync"
)

// RkRk returns the minimum norm solution to the system A*b=y, where A=U*V
// Knowing the whole matrix A is unnecessary and so it is never computed by the algorithm.
//
// Parameters:
// U, V are mat.Dense matrices that are a defactorization of A.
// y is mat.VecDense vector that represents the expected output for the system.
// iterations is an int representing how many times MAX is the algorithm allowed to run.
// tolerance is a float64 that represents the maximal error allowed.
// keepErrors is an optional boolean that specifies whether you want the function to retain the error at each iteration.
//
// Returns the vector b that solves A*b=y and a []float64 array containing the errors at each iteration.
//
// Notes:
// Pass a negative number as the iteration to use the default value of 100_000
// Even though you can pass as many boolean values for keepErrors only the first will be taken into account
func RkRk(U, V *mat.Dense, y *mat.VecDense, iterations int, tolerance float64, keepErrors ...bool) (mat.VecDense, []float64) {

	// STEP 0.
	// Initialization of variables
	if iterations < 0 {
		iterations = 100_000
	}

	rowsU, colsU := U.Dims()
	rowsV, colsV := V.Dims()

	x := mat.NewVecDense(colsU, nil)
	b := mat.NewVecDense(colsV, nil)

	var errors []float64

	// STEP 1.
	// Computing the frobenius norm uf U and V
	frobeniusU := FrobeniusSquared(U)
	frobeniusV := FrobeniusSquared(V)

	// STEP 2.
	// Computing the probability of each row of U and V
	probsU := make([]float64, rowsU)
	probsV := make([]float64, rowsV)

	waitGroup := sync.WaitGroup{}
	waitGroup.Add(2)
	go GetRowsProbability(probsU, frobeniusU, U, rowsU, &waitGroup)
	go GetRowsProbability(probsV, frobeniusV, V, rowsV, &waitGroup)
	waitGroup.Wait()

	// STEP 3.
	// Repeating the same process until we go insane
	for i := 0; i < iterations; i++ {
		randU := GetRandomRow(probsU)
		randV := GetRandomRow(probsV)

		chosenU := U.RowView(randU)
		chosenV := V.RowView(randV)

		euclideanU := EuclideanNormSquared(chosenU)
		euclideanV := EuclideanNormSquared(chosenV)

		x.AddScaledVec(
			x,
			(y.At(randU, 0)-mat.Dot(chosenU, x))/euclideanU,
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
