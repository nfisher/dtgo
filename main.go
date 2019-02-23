package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"os"

	"github.com/nfisher/dtgo/data"
	"github.com/nfisher/dtgo/decisiontree"
)

func Exec(inputFile string) {

	rows, err := readCSV(inputFile)
	if err != nil {
		fmt.Println("Unable to read CSV:", err.Error())
		return
	}

	d := data.New(rows[0], 0.20)

	err = data.Coerce(rows, d)
	if err != nil {
		fmt.Println("Unable to coerce CSV:", err.Error())
		return
	}


	decisionTree(d)
}

func decisionTree(d *data.TrainingAppender) {
	training, testData := d.Data()
	tree := decisiontree.Train(training, d.Header())

	decisiontree.Print(tree, "")

	fmt.Println("========================================================================")
	fmt.Println("training size:", len(training), "test size:", len(testData))

	for _, td := range testData {
		actual := td[len(td)-1]
		prediction := decisiontree.PredictionMap(decisiontree.Classify(td, tree).Predictions)
		keys := make([]string, 0, len(prediction))
		if len(prediction) > 1 {
			fmt.Printf("Actual: %s, Predicted: %s, From: %v\n", actual, prediction, td)
			continue
		}
		for k := range prediction {
			keys = append(keys, k)
		}
		predicted := keys[0]
		if actual != predicted {
			fmt.Printf("Actual: %s, Predicted: %s, From: %v\n", actual, keys[0], td)
			continue
		}

		fmt.Println("Predicted:", actual)
	}
}

func readCSV(inputFile string) ([][]string, error) {
	r, err := os.Open(inputFile)
	if err != nil {
		return nil, err
	}
	defer r.Close()

	rows, err := csv.NewReader(r).ReadAll()
	if err != nil {
		return nil, err
	}

	return rows, nil
}

func main() {
	var inputFile string
	flag.StringVar(&inputFile, "input", "", "input csv file for training")
	flag.Parse()

	if inputFile == "" {
		fmt.Println("An input filename must be provided.")
		flag.Usage()
		return
	}

	Exec(inputFile)
}
