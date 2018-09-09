package main

import (
	"fmt"
	"math"
)

func main() {
	trainingData := Rows{
		{"Green", 3, "Apple"},
		{"Yellow", 3, "Apple"},
		{"Red", 1, "Grape"},
		{"Red", 1, "Grape"},
		{"Yellow", 3, "Lemon"},
	}

	/*
		q := Question{Column: 0, Value: "Green"}
		fmt.Println(q.Match(trainingData[0]))
		fmt.Println(Gini(Rows{{"Apple"}, {"Apple"}}))
		fmt.Println(Gini(Rows{{"Apple"}, {"Orange"}, {"Grape"}, {"Grapefruit"}, {"Blueberry"}}))

		uncertainty := Gini(trainingData)
		trueRows, falseRows := Partition(trainingData, Question{0, "Green"})
		fmt.Println(InfoGain(trueRows, falseRows, uncertainty))

		trueRows, falseRows = Partition(trainingData, Question{0, "Red"})
		fmt.Println(InfoGain(trueRows, falseRows, uncertainty))

		fmt.Println(FindBestSplit(trainingData))
		fmt.Println(PrintLeaf(Classify(trainingData[0], tree).Predictions))
	*/

	tree := BuildTree(trainingData)
	PrintTree(tree, "")

	dd := Rows{
		{"Green", 3, "Apple"},
		{"Yellow", 4, "Apple"},
		{"Red", 2, "Grape"},
		{"Red", 1, "Grape"},
		{"Yellow", 3, "Lemon"},
	}

	for _, d := range dd {
		fmt.Printf("Actual: %s, Predicted: %s\n", d[len(d)-1], PrintLeaf(Classify(d, tree).Predictions))
	}

}

var header = []string{"color", "diameter", "label"}

func ClassCounts(rr Rows) LabelCount {
	counts := make(LabelCount)

	for _, r := range rr {
		label := r[len(r)-1].(string)
		c, ok := counts[label]
		if !ok {
			counts[label] = 0
		}
		counts[label] = c + 1
	}

	return counts
}

func UniqueVals(rows Rows, col int) []interface{} {
	uniq := make([]interface{}, 0)

	if len(rows) > 0 {
		v := rows[0][col]

		switch v.(type) {
		case int:
			ii := UniqueInts(rows, col)
			for i := range ii {
				uniq = append(uniq, i)
			}

		case string:
			ss := UniqueStrings(rows, col)
			for s := range ss {
				uniq = append(uniq, s)
			}

		default:
			fmt.Println("What the fuck are you doing here?")
			return nil
		}
	}

	return uniq
}

// UniqueStrings returns the unique values for the specified col.
func UniqueStrings(rr Rows, col int) StringSet {
	set := make(StringSet)

	for _, r := range rr {
		s := r[col].(string)
		set[s] = true
	}

	return set
}

func UniqueInts(rr Rows, col int) IntSet {
	set := make(IntSet)
	for _, r := range rr {
		s := r[col].(int)
		set[s] = true
	}

	return set
}

// LableCount is a collection for aggregating label counts.
type LabelCount map[string]int

// Rows is a collection for providing training data.
type Rows [][]interface{}

// StringSet is a container that acts as a set primitive for strings.
type StringSet map[string]bool

// IntSet is a container that acts as a set primitive for ints.
type IntSet map[int]bool

type Question struct {
	Column int
	Value  interface{}
}

func (q Question) Match(example []interface{}) bool {
	v := example[q.Column]
	switch t := v.(type) {
	case int:
		i := v.(int)
		qv := q.Value.(int)
		return i >= qv

	case string:
		s := v.(string)
		qv := q.Value.(string)
		return s == qv

	default:
		fmt.Printf("Unknown type %T\n", t)
		return false
	}
}

func (q Question) String() string {
	condition := "=="

	switch q.Value.(type) {
	case int:
		condition = ">="
	}

	return fmt.Sprintf("Is %s %s %v?", header[q.Column], condition, q.Value)
}

func Partition(rr Rows, q Question) (Rows, Rows) {
	var trueRows Rows
	var falseRows Rows

	for _, r := range rr {
		if q.Match(r) {
			trueRows = append(trueRows, r)
		} else {
			falseRows = append(falseRows, r)
		}
	}

	return trueRows, falseRows
}

func Gini(rr Rows) float64 {
	impurity := 1.0
	counts := ClassCounts(rr)
	var rowLen float64 = float64(len(rr))

	for _, c := range counts {
		prob := float64(c) / rowLen
		impurity -= math.Pow(prob, 2.0)
	}

	return impurity
}

func InfoGain(left, right Rows, uncertainty float64) float64 {
	leftLen := float64(len(left))
	rightLen := float64(len(right))

	p := leftLen / (leftLen + rightLen)
	return uncertainty - p*Gini(left) - (1-p)*Gini(right)
}

func FindBestSplit(rr Rows) (float64, *Question) {
	var bestGain float64
	var bestQuestion *Question
	uncertainty := Gini(rr)
	nFeatures := len(rr[0]) - 1

	for col := 0; col < nFeatures; col++ {
		vv := UniqueVals(rr, col)

		for _, v := range vv {
			q := Question{col, v}
			trueRows, falseRows := Partition(rr, q)
			if len(trueRows) == 0 || len(falseRows) == 0 {
				continue
			}

			gain := InfoGain(trueRows, falseRows, uncertainty)
			if gain >= bestGain {
				bestGain = gain
				bestQuestion = &q
			}
		}
	}

	return bestGain, bestQuestion
}

type DecisionNode struct {
	Question    *Question
	TrueBranch  *DecisionNode
	FalseBranch *DecisionNode
	IsLeaf      bool
	Predictions LabelCount
}

func BuildTree(rr Rows) *DecisionNode {
	gain, question := FindBestSplit(rr)

	if gain == 0 || question == nil {
		return &DecisionNode{
			IsLeaf:      true,
			Predictions: ClassCounts(rr),
		}
	}

	trueRows, falseRows := Partition(rr, *question)

	trueBranch := BuildTree(trueRows)
	falseBranch := BuildTree(falseRows)
	return &DecisionNode{
		Question:    question,
		TrueBranch:  trueBranch,
		FalseBranch: falseBranch,
	}
}

func PrintTree(node *DecisionNode, spacing string) {
	if node.IsLeaf {
		fmt.Printf("%sPrediction %v\n", spacing, node.Predictions)
		return
	}

	fmt.Printf("%v%s\n", spacing, node.Question)
	fmt.Printf("%v--> True:\n", spacing)
	PrintTree(node.TrueBranch, spacing+"  ")
	fmt.Printf("%v--> False:\n", spacing)
	PrintTree(node.FalseBranch, spacing+"  ")
}

func Classify(row []interface{}, node *DecisionNode) *DecisionNode {
	if node.IsLeaf {
		return node
	}

	if !node.Question.Match(row) {
		return Classify(row, node.FalseBranch)
	}

	return Classify(row, node.TrueBranch)
}

func PrintLeaf(cc LabelCount) map[string]string {
	var total float64 = 0.0
	for _, c := range cc {
		total += float64(c)
	}

	probs := make(map[string]string)

	for k, c := range cc {
		probs[k] = fmt.Sprintf("%v%%", float64(c)/total*100.0)
	}

	return probs
}
