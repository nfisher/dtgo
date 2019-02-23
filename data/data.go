package data

import (
	"math/rand"
	"regexp"
	"strconv"
	"time"

	dt "github.com/nfisher/dtgo/decisiontree"
)

type Appender interface {
	Append([]interface{})
}

type Trainer interface {
	Header() []string
	Data() (dt.Rows, dt.Rows)
}

type TrainingAppender struct {
	header       []string
	data dt.Rows
	split        float64
}

func (a *TrainingAppender) Header() []string {
	return a.header
}

func (a *TrainingAppender) Data() (dt.Rows, dt.Rows) {
	var trainingData dt.Rows
	var testData dt.Rows

	for i := range a.data {
		row := a.data[i]
		if rand.Float64() < a.split {
			testData = append(testData, row)
			continue
		}

		trainingData = append(trainingData, row)
	}

	return trainingData, testData
}

func (a *TrainingAppender) Append(row []interface{}) {
	a.data = append(a.data, row)
}

func New(header []string, split float64) *TrainingAppender {
	rand.Seed(time.Now().UnixNano())
	return &TrainingAppender{split: split, header: header}
}

func Coerce(rows [][]string, a Appender) error {
	for i := range rows {
		if i == 0 {
			continue
		}

		row := rows[i]

		var d []interface{}
		for k := range row {
			v := row[k]

			if numberMatch.MatchString(v) {
				f, err := strconv.ParseFloat(v, 64)
				if err != nil {
					return err
				}
				d = append(d, f)
				continue
			}
			d = append(d, v)
		}
		a.Append(d)
	}

	return nil
}

var (
	numberMatch = regexp.MustCompile(`^\d+(?:\.\d+)?$`)
)
