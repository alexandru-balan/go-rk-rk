package main

import (
	"fmt"
	"github.com/alexandru-balan/go-rk-rk/algorithms"
	"github.com/alexandru-balan/go-rk-rk/generators/gaussian"
	"gonum.org/v1/gonum/mat"
	"math"
	"sync"
	"time"
)

const (
	MEAN      = 0.0
	DEVIATION = 1.0
	m         = 250
	n         = 100
	k         = 75
)

func main() {

	startTime := time.Now().UnixNano()

	U := mat.NewDense(m, k, nil)
	V := mat.NewDense(k, n, nil)
	X := mat.NewDense(m, n, nil)
	B := mat.NewVecDense(n, nil)
	b := mat.NewVecDense(n, nil)
	y := mat.NewVecDense(m, nil)

	// Getting random test values for our matrices
	waitGroup := new(sync.WaitGroup)
	waitGroup.Add(3)
	go gaussian.Generate(U, MEAN, DEVIATION, waitGroup)
	go gaussian.Generate(V, MEAN, DEVIATION, waitGroup)
	go gaussian.GenerateVector(B, MEAN, DEVIATION, waitGroup)
	waitGroup.Wait()

	X.Mul(U, V)
	y.MulVec(X, B)

	//*B = utils.SolveLeastSquares(X, y)

	tolerance := math.Pow(10, -32)
	var errors []float64
	*b, errors = algorithms.RekRek(U, V, y, 1_000_000, tolerance, true)
	// utils.Plot(errors, "./build/scatter.png")

	fmt.Println(errors[0])
	fmt.Println(errors[len(errors)-1])

	endTime := time.Now().UnixNano()

	fmt.Printf("%f seconds\n", (float64(endTime-startTime))/1_000_000_000)
}
