package algorithms

import (
	rand2 "golang.org/x/exp/rand"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/sampleuv"
	"math"
	"sync"
	"time"
)

var errors []float64

func getRandomRow(rowsProb []float64) int {
	source := rand2.NewSource(uint64(time.Now().UnixNano()))

	index, _ := sampleuv.NewWeighted(rowsProb, source).Take()

	return index
}

// euclideanNorm will return the euclidean norm of a vector
func euclideanNorm(vector mat.Vector) float64 {
	return mat.Norm(vector, 2)
}

func frobeniusSquared(matrix *mat.Dense) float64 {
	return math.Pow(mat.Norm(matrix, 2), 2.0)
}

func getRowsProbability(probVector []float64, frobenius float64, matrix *mat.Dense, rownum int, group *sync.WaitGroup) {
	for i := 0; i < rownum; i++ {
		probVector[i] = math.Pow(euclideanNorm(matrix.RowView(i)), 2) / frobenius
	}

	group.Done()
}

// !!! Only the first option for keepErrors will be used
func RkRk(U, V *mat.Dense, B, y *mat.VecDense, iterations int, keepErrors ...bool) (mat.VecDense, []float64) {

	// STEP 0.
	// Initialization of variables
	rows_U, cols_U := U.Dims()
	rows_V, cols_V := V.Dims()

	x := mat.NewVecDense(cols_U, nil)
	b := mat.NewVecDense(cols_V, nil)

	errors = make([]float64, iterations)

	// STEP 1.
	// Computing the frobenius norm uf U and V
	frobenius_U := frobeniusSquared(U)
	frobenius_V := frobeniusSquared(V)

	// STEP 2.
	// Computing the probability of each row of U and V
	probs_U := make([]float64, rows_U)
	probs_V := make([]float64, rows_V)

	waitGroup := sync.WaitGroup{}
	waitGroup.Add(2)
	go getRowsProbability(probs_U, frobenius_U, U, rows_U, &waitGroup)
	go getRowsProbability(probs_V, frobenius_V, V, rows_V, &waitGroup)
	waitGroup.Wait()

	// STEP 3.
	// Repeating the same process until we go insane
	for i := 0; i < iterations; i++ {
		rand_U := getRandomRow(probs_U)
		rand_V := getRandomRow(probs_V)

		chosen_U := U.RowView(rand_U)
		chosen_V := V.RowView(rand_V)

		euclidean_U := math.Pow(euclideanNorm(chosen_U), 2)
		euclidean_V := math.Pow(euclideanNorm(chosen_V), 2)

		x.AddScaledVec(
			x,
			(y.At(rand_U, 0)-mat.Dot(chosen_U, x))/euclidean_U,
			chosen_U)
		b.AddScaledVec(
			b,
			(x.At(rand_V, 0)-mat.Dot(chosen_V, b))/euclidean_V,
			chosen_V)

		if keepErrors[0] {
			err_vec := new(mat.VecDense)
			err_vec.SubVec(b, B)
			errors[i] = math.Pow(euclideanNorm(err_vec), 2)
		}
	}

	return *b, errors
}
