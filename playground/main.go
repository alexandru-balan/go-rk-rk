package main

import (
	"fmt"
	"github.com/go-rk-rk/algorithms"
	"github.com/go-rk-rk/generators/gaussian"
	"github.com/go-rk-rk/utils"
	"gonum.org/v1/gonum/mat"
	"sync"
	"time"
)

const (
	MEAN      = 0.0
	DEVIATION = 1.0
	m         = 5
	n         = 3
	k         = 2
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

	*B = utils.SolveLeastSquares(X, y)

	var errors []float64
	*b, errors = algorithms.RkRk(U, V, B, y, 10000, true)
	utils.Plot(errors, "./build/scatter.png")

	fmt.Println(errors[0])
	fmt.Println(errors[len(errors)-1])

	endTime := time.Now().UnixNano()

	fmt.Printf("%f seconds\n", (float64(endTime-startTime))/1_000_000_000)
}
