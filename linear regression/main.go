package main

import (
	"fmt"
	"math"

)

func calculateRMSE(predictions []float64, actual []float64) float64 {
	sumSquaredErrors := 0.0
	for i := range predictions {
		error := predictions[i] - actual[i]
		sumSquaredErrors += math.Pow(error, 2)
	}
	meanSquaredError := sumSquaredErrors / float64(len(predictions))
	rmse := math.Sqrt(meanSquaredError)
	return rmse
}


// Implement Gradient Decent
func GradientDecent(m_current []float64, b_current float64, x_vals [][]float64, y_vals []float64, lr float64) ([]	float64, float64) {

	

	// TODO: length of x must be equal to length of y

	n := len(x_vals)
	n_float := float64(n)

	m_gradients := make([]float64, len(m_current))
	b_gradient := 0.0
	

	for i := 0; i < n; i++ {
		x := x_vals[i]
		y := y_vals[i]

		prediction_y := 0.0

		for j := range x {
			prediction_y += m_current[j] * x[j]
		}
		prediction_y += b_current

		for j := range x {
			m_gradients[j] += -(2 / n_float) * x[j] * (y - prediction_y)
		}

		b_gradient += -(2 / n_float) * (y - prediction_y)
	}

	new_m := make([]float64, len(m_current))

	for j := range m_current {
		new_m[j] = m_current[j] - m_gradients[j]*lr
	}

	b := b_current - b_gradient*lr

	return new_m, b
}

type Model struct {
	m []float64
	b float64
}



// Implement Fitting
func (model *Model) fit(epochs int, x [][]float64, y []float64) {
	m := make([]float64, len(x[0])) 
	b := 0.0

	for epoch := 0; epoch < epochs; epoch++ {
		m, b = GradientDecent(m, b, x, y, 0.01)
	}

	model.m = m
	model.b = b
}

func (model *Model) predict(x [][]float64) []float64 {
	predictions := make([]float64, len(x))

	for i, currentX := range x {
		result := model.b
		for j := range currentX {
			result += model.m[j] * currentX[j]
		}
		predictions[i] = result
	}

	return predictions
}


func main() {
	model := Model{}

	yearsExperience := []float64{
		1.2000000000000002,
		1.4000000000000001,
		1.6,
		2.1,
		2.3000000000000003,
		3.0,
		3.1,
		3.3000000000000003,
		3.3000000000000003,
		3.8000000000000003,
		4.0,
		4.1,
		4.1,
		4.199999999999999,
		4.6,
		5.0,
		5.199999999999999,
		5.3999999999999995,
		6.0,
		6.1,
		6.8999999999999995,
		7.199999999999999,
		8.0,
		8.299999999999999,
		8.799999999999999,
		9.1,
		9.6,
		9.7,
		10.4,
	}
	
	salary := []float64{
		39344.0,
		46206.0,
		37732.0,
		43526.0,
		39892.0,
		56643.0,
		60151.0,
		54446.0,
		64446.0,
		57190.0,
		63219.0,
		55795.0,
		56958.0,
		57082.0,
		61112.0,
		67939.0,
		66030.0,
		83089.0,
		81364.0,
		93941.0,
		91739.0,
		98274.0,
		101303.0,
		113813.0,
		109432.0,
		105583.0,
		116970.0,
		112636.0,
		122392.0,
	}

	fmt.Println(len(yearsExperience))
	fmt.Println(len(salary))

	x := make([][]float64, len(yearsExperience))
	for i := range x {
		x[i] = []float64{1, yearsExperience[i]} // Adding the intercept term
	}
	
	model.fit(100, x, salary)

	toPredict := [][]float64{
		{1, 2.5}, // Predicting salary for 2.5 years of experience
		{1, 7.0}, // Predicting salary for 7.0 years of experience
	}

	predictions := model.predict(toPredict)

	fmt.Println(predictions)

	rmse := calculateRMSE(predictions, salary)
	fmt.Printf("Root Mean Squared Error (RMSE): %.2f\n", rmse)
	
}