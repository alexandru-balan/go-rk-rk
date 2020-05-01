package main

import (
	"fmt"
	"github.com/go-ml-linear-systems/src/algorithms"
	"github.com/go-ml-linear-systems/src/generators/normal"
	"gonum.org/v1/gonum/mat"
	"math"
	"sync"
	"time"
)

const (
	MEAN      = 0
	DEVIATION = 1
	m         = 200
	n         = 150
	k         = 100
)

func main() {

	U := mat.NewDense(m, k, nil)
	V := mat.NewDense(k, n, nil)
	X := mat.NewDense(m, n, nil)
	B := mat.NewDense(n, 1, nil)
	y := mat.NewDense(m, 1, nil)

	var waitGroup sync.WaitGroup
	waitGroup.Add(3)

	startTime := time.Now().UnixNano()
	go normal.Generate(U, MEAN, DEVIATION, &waitGroup)
	go normal.Generate(V, MEAN, DEVIATION, &waitGroup)
	go normal.Generate(B, MEAN, DEVIATION, &waitGroup)
	waitGroup.Wait()
	X.Mul(U, V)
	y.Mul(X, B)

	algorithms.RkRk(U, V, y, B, int(7*math.Pow(10, 4)), true)

	//fmt.Printf("Matrix U = \n%.4v\n\n", mat.Formatted(U, mat.Prefix(""), mat.Squeeze()))

	endTime := time.Now().UnixNano()

	fmt.Printf("%.3f seconds\n", (float64(endTime-startTime))/1_000_000_000)
}
