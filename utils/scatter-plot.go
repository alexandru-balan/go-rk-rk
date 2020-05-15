package utils

import (
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
	"image/color"
	"log"
	"math"
)

func Plot(values []float64, path string) {
	p, err := plot.New()
	if err != nil {
		log.Panic(err)
	}

	points := make(plotter.XYs, len(values))
	for i := range points {
		points[i].X = float64(i)
		points[i].Y = values[i]
	}

	p.Y.Min = math.Pow(10, -10)
	p.Title.Text = "REK-RK"
	p.X.Label.Text = "iterations"
	p.Y.Label.Text = "error"
	p.Add(plotter.NewGrid())

	scatter, err := plotter.NewScatter(points)
	if err != nil {
		log.Panic(err)
	}

	scatter.GlyphStyle.Color = color.RGBA{R: 255, B: 128, A: 255}
	scatter.GlyphStyle.Radius = vg.Points(2)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}

	p.Add(scatter)

	err = p.Save(400, 400, path)
	if err != nil {
		log.Panic(err)
	}
}
