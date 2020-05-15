package algorithms

import (
	rand2 "golang.org/x/exp/rand"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/sampleuv"
	"math"
	"sync"
	"time"
)

// EuclideanNormSquared returns the squared euclidean norm of a mat.Vector.
//
// @param vector : mat.Vector -- The vector for which you need the squared euclidean norm
// Returns a float64 value
func EuclideanNormSquared(vector mat.Vector) float64 {
	return math.Pow(mat.Norm(vector, 2), 2)
}

// FrobeniusSquared returns the squared frobenius norm of a mat.Dense matrix
func FrobeniusSquared(matrix *mat.Dense) float64 {
	return math.Pow(mat.Norm(matrix, 2), 2.0)
}

// GetRandomRow performs weighted sampling with the weights you provide in the rowsProb array
//
// The scope of this function is to return a random row index for the RkRk and RekRek algorithms.
// Each row probability is computed by GetRowsProbability method
func GetRandomRow(rowsProb []float64) int {
	source := rand2.NewSource(uint64(time.Now().UnixNano()))

	index, _ := sampleuv.NewWeighted(rowsProb, source).Take()

	return index
}

// GetRowsProbability computes the probabilities of choosing a random row from a matrix
//
// The probability is computed as the squared euclidean norm of the row divided by the
// squared frobenius norm of the matrix
//
// Since this method is intended to be used with the RkRk and RekRek algorithms which require computing
// the probabilities of many rows, this method requires a WaitGroup and has multithreaded behaviour.
func GetRowsProbability(probVector []float64, frobenius float64, matrix *mat.Dense, rownum int, group *sync.WaitGroup) {
	for i := 0; i < rownum; i++ {
		probVector[i] = EuclideanNormSquared(matrix.RowView(i)) / frobenius
	}

	group.Done()
}
