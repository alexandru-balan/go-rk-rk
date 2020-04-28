package main

import (
	"fmt"
	"go-ml-linear-systems/src/algorithms"
	"go-ml-linear-systems/src/generators/normal"
	"gonum.org/v1/gonum/mat"
	"math"
	"sync"
	"time"
)

const (
	MEAN      = 7
	DEVIATION = 2
)

func main() {

	U := mat.NewDense(200, 100, nil)
	V := mat.NewDense(100, 150, nil)
	B := mat.NewDense(150, 1, nil)
	X := mat.NewDense(200, 150, nil)
	y := mat.NewDense(200, 1, nil)

	var waitGroup sync.WaitGroup
	waitGroup.Add(3)

	startTime := time.Now().UnixNano()
	go normal.Generate(U, MEAN, DEVIATION, &waitGroup)
	go normal.Generate(V, MEAN, DEVIATION, &waitGroup)
	go normal.Generate(B, MEAN, DEVIATION, &waitGroup)
	waitGroup.Wait()
	X.Mul(U, V)
	y.Mul(X, B)
	endTime := time.Now().UnixNano()

	fmt.Printf("%.3f seconds\n", (float64(endTime-startTime))/1_000_000)

	algorithms.RkRk(U, V, y, B, int(7*math.Pow(10, 4)), true)

	//fmt.Printf("Matrix U = \n%.4v\n\n", mat.Formatted(U, mat.Prefix(""), mat.Squeeze()))
}
