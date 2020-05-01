package algorithms

import (
	"fmt"
	rand2 "golang.org/x/exp/rand"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
	"math"
	"sort"
	"sync"
	"time"
)

const GOROUTINES = 10

var errors []float64

func getRandomRow(rowsProb []float64, maxRows int, c chan int) {
	source := rand2.NewSource(uint64(time.Now().UnixNano()))

	cdf := make([]float64, len(rowsProb))
	floats.CumSum(cdf, rowsProb)

	val := distuv.Uniform{Min: 0.0, Max: 1.0, Src: source}.Rand()

	sort.Float64s(rowsProb)
	index := sort.Search(len(cdf), func(i int) bool {
		return cdf[i] > val
	})

	c <- index
}

func sumSquares(slice []float64) float64 {
	sum := 0.0
	for _, val := range slice {
		sum += math.Pow(val, 2.0)
	}

	return sum
}

// euclideanNorm will return the squared euclidean norm of a vector of float64 elements
func euclideanNorm(vector []float64) float64 {
	return sumSquares(vector)
}

func frobeniusSquared(matrix *mat.Dense) float64 {
	return math.Pow(mat.Norm(matrix, 2), 2.0)
}

func getRowsProbability(probVector []float64, frobenius float64, matrix *mat.Dense, rownum int, group *sync.WaitGroup) {
	for i := 0; i < rownum; i++ {
		probVector[i] = euclideanNorm(matrix.RawRowView(i)) / frobenius
	}

	group.Done()
}

// !!! Only the first option for keepErrors will be used
func RkRk(U, V, y, B *mat.Dense, iterations int, keepErrors ...bool) (mat.Dense, []float64) {
	urows, ucols := U.Dims()
	vrows, vcols := V.Dims()

	errors = make([]float64, iterations)

	x := mat.NewDense(ucols, 1, nil)
	b := mat.NewDense(vcols, 1, nil)

	// Int communication channels for getting random rows of U and V
	c2 := make(chan int)
	c3 := make(chan int)

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
		go getRandomRow(vRowsProb, vrows, c3)
		uRandomRow := <-c2
		vRandomRow := <-c3

		// Update the x vector
		uChosenRow := U.RawRowView(uRandomRow)
		uEuclidean := euclideanNorm(uChosenRow)

		uRow := mat.NewDense(1, ucols, uChosenRow)
		uMultiplied := mat.NewDense(1, 1, nil)
		uMultiplied.Mul(uRow, x)
		uScaler := (y.RawRowView(uRandomRow)[0] - uMultiplied.At(0, 0)) / uEuclidean
		uScaled := mat.NewDense(ucols, 1, nil)
		uScaled.Scale(uScaler, uRow.T())
		x.Add(x, uScaled)

		// Update the b vector using x (we use the same variables as above)
		vChosenRow := V.RawRowView(vRandomRow)
		vEuclidean := euclideanNorm(vChosenRow)

		vRow := mat.NewDense(1, vcols, vChosenRow)
		vMultiplied := mat.NewDense(1, 1, nil)
		vMultiplied.Mul(vRow, b)
		vScaler := (x.RawRowView(vRandomRow)[0] - vMultiplied.At(0, 0)) / vEuclidean
		vScaled := mat.NewDense(vcols, 1, nil)
		vScaled.Scale(vScaler, vRow.T())
		b.Add(b, vScaled)

		if keepErrors[0] {
			aux := mat.NewDense(vcols, 1, nil)
			aux.Sub(b, B)
			column := make([]float64, vcols)
			for j := 0; j < vcols; j++ {
				column[j] = aux.At(j, 0)
			}
			errors[i] = euclideanNorm(column)
		}
	}

	return *b, errors
}

func RekRek(U, V, y, B *mat.Dense, iterations int, keepErrors ...bool) (mat.Dense, []float64) {
	urows, ucols := U.Dims()
	vrows, vcols := V.Dims()

	UCol := mat.NewDense(ucols, urows, nil)
	UCol.Copy(U.T())

	errors = make([]float64, iterations)

	x := mat.NewDense(ucols, 1, nil)
	b := mat.NewDense(vcols, 1, nil)
	z := mat.NewDense(urows, 1, nil)
	z.Copy(y)

	// Int communication channels for getting random rows of U and V
	c2 := make(chan int)
	c3 := make(chan int)
	c4 := make(chan int) // Getting random columns of U

	// Compute the frobenius norm of the U and V matrices
	uFrobenius := frobeniusSquared(U)
	vFrobenius := frobeniusSquared(V)

	// Computing the probability of each row of U and V
	uRowsProb := make([]float64, urows)
	vRowsProb := make([]float64, vrows)
	uColsProb := make([]float64, ucols)

	wg := sync.WaitGroup{}
	wg.Add(3)
	go getRowsProbability(uRowsProb, uFrobenius, U, urows, &wg)
	go getRowsProbability(vRowsProb, vFrobenius, V, vrows, &wg)
	go getRowsProbability(uColsProb, uFrobenius, UCol, ucols, &wg)
	wg.Wait()

	for i := 0; i < iterations; i++ {
		go getRandomRow(uRowsProb, urows, c2)
		go getRandomRow(vRowsProb, vrows, c3)
		go getRandomRow(uColsProb, ucols, c4)

		uRandomRow := <-c2
		vRandomRow := <-c3
		uRandomCol := <-c4

		// Updating z
		chosenCol := UCol.RawRowView(uRandomCol)
		aux := mat.NewDense(urows, 1, chosenCol)
		var aux2 mat.Dense
		aux2.Mul(aux.T(), z)
		aux3 := (aux2.At(0, 0)) / euclideanNorm(chosenCol)
		aux4 := mat.NewDense(urows, 1, nil)

		aux4.Scale(aux3, aux)

		z.Sub(z, aux4)

		// Updatting x using z
		chosenRow := U.RawRowView(uRandomRow)
		aux = mat.NewDense(1, ucols, chosenRow)
		aux2.Mul(aux, x)
		fmt.Printf("%f + %f - %f\n", y.RawRowView(uRandomRow)[0], aux2.At(0, 0), z.RawRowView(uRandomRow)[0])
		aux3 = (y.RawRowView(uRandomRow)[0] + aux2.At(0, 0) - z.RawRowView(uRandomRow)[0]) / euclideanNorm(chosenRow)
		//fmt.Println(aux3)
		aux4 = mat.NewDense(ucols, 1, nil)
		aux4.Copy(aux.T())
		aux4.Scale(aux3, aux4)

		// Updating x
		x.Add(x, aux4)

		// Updating b using x
		chosenRow = V.RawRowView(vRandomRow)
		aux = mat.NewDense(1, vcols, chosenRow)
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
			//fmt.Printf("%f - %f = %f\n", B.At(0,0), b.At(0,0), B.At(0,0) - b.At(0,0))
			column := make([]float64, vcols)
			for j := 0; j < vcols; j++ {
				column[j] = aux5.At(j, 0)
			}
			errors[i] = euclideanNorm(column)
		}
	}

	return *b, errors
}
