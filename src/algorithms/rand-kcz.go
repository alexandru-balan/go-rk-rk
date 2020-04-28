package algorithms

import (
	rand2 "golang.org/x/exp/rand"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
	"math"
	"sync"
	"time"
)

const GOROUTINES = 10

/**
TODO: While the row is not chosen, generate a random number from the normal distribution and compare it with the probability of the first row; if the probability is not met then go to next row and generate another number

*/
func getRandomRow(rowsProb []float64, maxRows int, c chan int) {
	seed := rand2.NewSource(uint64(time.Now().UnixNano()))
	chosen := int(distuv.Uniform{Min: 0, Max: float64(maxRows), Src: seed}.Rand())
	var random float64
	for {
		random = distuv.Uniform{Min: 0, Max: 1, Src: seed}.Rand()
		if rowsProb[chosen] > random {
			c <- chosen
		} else {
			chosen = int(distuv.Uniform{Min: 0, Max: float64(maxRows), Src: seed}.Rand())
		}
	}
}

func sumSquares(slice []float64, c chan float64) {
	sum := 0.0
	for _, val := range slice {
		sum += math.Pow(val, 2.0)
	}

	c <- sum
}

// euclideanNorm will return the squared euclidean norm of a vector of float64 elements
func euclideanNorm(vector []float64) float64 {
	c := make(chan float64, GOROUTINES)

	for i := 0; i < GOROUTINES; i++ {
		go sumSquares(vector[i*len(vector)/GOROUTINES:(i+1)*(len(vector)/GOROUTINES)], c)
	}

	sum := 0.0
	for i := 0; i < GOROUTINES; i++ {
		sum += <-c
	}
	close(c)

	return sum
}

func frobeniusSquared(matrix *mat.Dense) float64 {
	return math.Pow(mat.Norm(matrix, 2), 2.0)
}

func computeRowProbability(frobenius float64, row []float64, c chan float64) {
	euclidean := euclideanNorm(row)

	c <- euclidean / frobenius
}

func getRowsProbability(probVector []float64, frobenius float64, matrix *mat.Dense, rownum int, group *sync.WaitGroup) {
	c := make(chan float64, rownum)

	for i := 0; i < rownum; i++ {
		go computeRowProbability(frobenius, matrix.RawRowView(i), c)
	}

	for i := 0; i < rownum; i++ {
		probVector[i] = <-c
	}
	close(c)
	group.Done()
}

// !!! Only the first option for keepErrors will be used
func RkRk(U, V, y, B *mat.Dense, iterations int, keepErrors ...bool) {
	urows, ucols := U.Dims()
	vrows, vcols := V.Dims()

	errors := make([]float64, iterations)

	x := mat.NewDense(ucols, 1, nil)
	b := mat.NewDense(vcols, 1, nil)

	// Int communication channel for getting random rows of U and V
	c2 := make(chan int)

	// Compute the frobenius norm of the U and V matrices
	uFrobenius := frobeniusSquared(U)
	vFrobenius := frobeniusSquared(V)

	// Computing the probability of each row of U and V
	uRowsProb := make([]float64, urows)
	vRowsProb := make([]float64, vrows)

	wg := sync.WaitGroup{}
	wg.Add(2)
	go getRowsProbability(uRowsProb, uFrobenius, U, urows, &wg)
	go getRowsProbability(vRowsProb, vFrobenius, V, vrows, &wg)
	wg.Wait()

	for i := 0; i < iterations; i++ {

		go getRandomRow(uRowsProb, urows, c2)
		go getRandomRow(vRowsProb, vrows, c2)
		uRandomRow := <-c2
		vRandomRow := <-c2

		// Update the x vector
		chosenRow := U.RawRowView(uRandomRow)

		// aux is the representation of the chosen row as a matrix structure with one row
		aux := mat.NewDense(1, ucols, chosenRow)

		var aux2 mat.Dense
		aux2.Mul(aux, x)
		aux3 := (y.RawRowView(uRandomRow)[0] - aux2.At(0, 0)) / euclideanNorm(chosenRow)

		// aux4 is the adjugate transpose matrix; since this is a real-world facing package there are no complex numbers
		// So the transpose is used
		aux4 := mat.NewDense(ucols, 1, nil)
		aux4.Copy(aux.T())

		aux4.Scale(aux3, aux4)

		// Updating x
		x.Add(x, aux4)

		// Update the b vector using x (we use the same variables as above)
		chosenRow = V.RawRowView(vRandomRow)
		aux = mat.NewDense(1, vcols, chosenRow)
		aux2.Reset()
		aux2.Mul(aux, b)
		aux3 = (x.RawRowView(vRandomRow)[0] - aux2.At(0, 0)) / euclideanNorm(chosenRow)
		aux4 = mat.NewDense(vcols, 1, nil)
		aux4.Copy(aux.T())
		aux4.Scale(aux3, aux4)

		// Updating b
		b.Add(b, aux4)
		if keepErrors[0] {
			aux5 := mat.NewDense(vcols, 1, nil)
			aux5.Sub(b, B)
			column := make([]float64, vcols)
			for j := 0; j < vcols; j++ {
				column[j] = aux5.At(j, 0)
			}
			errors[i] = euclideanNorm(column)
		}
	}

	points := make()
}
