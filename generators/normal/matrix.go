package normal

import (
	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
	"sync"
	"time"
)

func getRow(numberOfRepetitions, numberOfSamples int, mean, sigma float64, c chan []float64) {
	seed := rand.NewSource(uint64(time.Now().UnixNano()))

	for r := 0; r < numberOfRepetitions; r++ {
		rowData := make([]float64, numberOfSamples)
		for i := 0; i < numberOfSamples; i++ {
			rowData[i] = distuv.Normal{Mu: mean, Sigma: sigma, Src: seed}.Rand()
		}
		c <- rowData
	}

	close(c)
}

func Generate(matrix *mat.Dense, mean, sigma float64, group *sync.WaitGroup) {
	rows, cols := matrix.Dims()

	c := make(chan []float64, rows)

	go getRow(rows, cols, mean, sigma, c)

	j := 0
	for row := range c {
		matrix.SetRow(j, row)
		j++
	}

	group.Done()
}
