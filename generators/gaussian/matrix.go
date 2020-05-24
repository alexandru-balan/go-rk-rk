package gaussian

import (
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
	"sync"
)

func getRow(numberOfRepetitions, numberOfSamples int, mean, sigma float64, c chan []float64) {
	//seed := rand.NewSource(uint64(time.Now().UnixNano()))

	for r := 0; r < numberOfRepetitions; r++ {
		rowData := make([]float64, numberOfSamples)
		for i := 0; i < numberOfSamples; i++ {
			rowData[i] = distuv.Normal{Mu: mean, Sigma: sigma, Src: nil}.Rand()
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

func GenerateVector(vector *mat.VecDense, mean, sigma float64, group *sync.WaitGroup) {
	length := vector.Len()
	//seed := rand.NewSource(uint64(time.Now().UnixNano()))
	distribution := distuv.Normal{Mu: mean, Sigma: sigma, Src: nil}

	for i := 0; i < length; i++ {
		vector.SetVec(i, distribution.Rand())
	}

	group.Done()
}
