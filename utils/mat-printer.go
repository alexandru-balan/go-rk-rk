package utils

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
)

func PrintMatrix(matrix *mat.Dense, name string) {
	fmt.Printf("%s=\n%v\n\n", name, mat.Formatted(matrix, mat.Prefix(""), mat.Squeeze()))
}

func PrintVector(vector *mat.VecDense, name string) {
	fmt.Printf("%s=\n%v\n\n", name, mat.Formatted(vector, mat.Prefix(""), mat.Squeeze()))
}
