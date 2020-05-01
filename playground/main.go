package main

import (
	"fmt"
	"github.com/go-ml-linear-systems/generators/normal"
	"gonum.org/v1/gonum/mat"
	"sync"
	"time"
)

const (
	MEAN      = 0
	DEVIATION = 1
	m         = 20
	n         = 15
	k         = 10
)

func main() {

	U := mat.NewDense(m, k, nil)
	V := mat.NewDense(k, n, nil)
	X := mat.NewDense(m, n, nil)
	B := mat.NewDense(n, 1, nil)
	//b := mat.NewDense(n, 1, nil)
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

	//var err []float64

	_ = B.Solve(X, y)

	/**b, err = algorithms.RkRk(U, V, y, B, 70_000, true)
	Plot(err, "./build/scatter.png")

	fmt.Println(err[0])
	fmt.Println(err[69_999])*/

	//*b, _ = algorithms.RekRek(U, V, y, B, 100_000, true)
	//Plot(err, "./build/scatter2.png")

	//fmt.Printf("Matrix U = \n%v\n\n", mat.Formatted(U.RowView(0).T(), mat.Prefix(""), mat.Squeeze()))

	//fmt.Println(U.RowView(0).AtVec(0))

	endTime := time.Now().UnixNano()

	fmt.Printf("%f seconds\n", (float64(endTime-startTime))/1_000_000_000)
}
