package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"regexp"
	"strconv"

	dt "github.com/nfisher/dtgo/decisiontree"
)

func main() {
	var inputFile string
	flag.StringVar(&inputFile, "input", "", "input csv file for training")
	flag.Parse()

	if inputFile == "" {
		fmt.Println("An input filename must be provided.\n")
		flag.Usage()
		return
	}

	r, err := os.Open(inputFile)
	if err != nil {
		fmt.Println("unable to open input file:", err)
		return
	}

	var header []string
	var trainingData dt.Rows
	var testData dt.Rows

	csvr := csv.NewReader(r)
	rows, err := csvr.ReadAll()
	if err != nil {
		fmt.Println("error reading csv:", err)
	}

	for i := range rows {
		row := rows[i]
		if i == 0 {
			header = row
			continue
		}

		var d []interface{}
		for k := range row {
			v := row[k]

			if numberMatch.MatchString(v) {
				f, err := strconv.ParseFloat(v,64)
				if err != nil {
					log.Println("unable to parse float:", err)
					return
				}
				d = append(d, f)
				continue
			}
			d = append(d, v)
		}

		if rand.Float64() < 0.65 {
			trainingData = append(trainingData, d)
			continue
		}

		testData = append(testData, d)
	}

	tree := dt.Build(trainingData, header)
	dt.Print(tree, "")

	for _, d := range testData {
		fmt.Printf("Actual: %s, Predicted: %s, From: %v\n", d[len(d)-1], dt.PredictionMap(dt.Classify(d, tree).Predictions), d)
	}
}

var (
	numberMatch = regexp.MustCompile(`^\d+(?:\.\d+)?$`)
)
