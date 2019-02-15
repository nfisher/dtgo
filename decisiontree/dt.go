package decisiontree

import (
	"fmt"
	"math"
)

// LableCount is a collection for aggregating label counts.
type LabelCount map[string]int

// Rows is a collection for providing training data.
type Rows [][]interface{}

// StringSet is a container that acts as a set primitive for strings.
type StringSet map[string]bool

// IntSet is a container that acts as a set primitive for ints.
type IntSet map[int]bool

type FloatSet map[float64]bool

type Question struct {
	Column int
	Value  interface{}
	header []string
}

func (q Question) Match(example []interface{}) bool {
	v := example[q.Column]
	switch t := v.(type) {
	case float64:
		f := v.(float64)
		qv := q.Value.(float64)
		return f >= qv

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

	return fmt.Sprintf("Is %s %s %v?", q.header[q.Column], condition, q.Value)
}

type DecisionNode struct {
	Question    *Question
	TrueBranch  *DecisionNode
	FalseBranch *DecisionNode
	IsLeaf      bool
	Predictions LabelCount
}

func Build(rr Rows, header []string) *DecisionNode {
	gain, question := findBestSplit(rr, header)

	if gain == 0 || question == nil {
		return &DecisionNode{
			IsLeaf:      true,
			Predictions: classCounts(rr),
		}
	}

	trueRows, falseRows := partition(rr, *question)

	trueBranch := Build(trueRows, header)
	falseBranch := Build(falseRows, header)
	return &DecisionNode{
		Question:    question,
		TrueBranch:  trueBranch,
		FalseBranch: falseBranch,
	}
}

func Print(node *DecisionNode, spacing string) {
	if node.IsLeaf {
		fmt.Printf("%sPrediction %v\n", spacing, node.Predictions)
		return
	}

	fmt.Printf("%v%s\n", spacing, node.Question)
	fmt.Printf("%v --> True:\n", spacing)
	Print(node.TrueBranch, spacing+"  ")
	fmt.Printf("%v --> False:\n", spacing)
	Print(node.FalseBranch, spacing+"  ")
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

func PredictionMap(cc LabelCount) map[string]string {
	var total = 0.0
	for _, c := range cc {
		total += float64(c)
	}

	probs := make(map[string]string)

	for k, c := range cc {
		probs[k] = fmt.Sprintf("%v%%", float64(c)/total*100.0)
	}

	return probs
}

func classCounts(rr Rows) LabelCount {
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

func uniqueVals(rows Rows, col int) []interface{} {
	uniq := make([]interface{}, 0)

	if len(rows) > 0 {
		v := rows[0][col]

		switch v.(type) {
		case float64:
			ff := uniqueFloats(rows, col)
			for f := range ff {
				uniq = append(uniq, f)
			}

		case int:
			ii := uniqueInts(rows, col)
			for i := range ii {
				uniq = append(uniq, i)
			}

		case string:
			ss := uniqueStrings(rows, col)
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

// uniqueStrings returns the unique values for the specified col.
func uniqueStrings(rr Rows, col int) StringSet {
	set := make(StringSet)

	for _, r := range rr {
		s := r[col].(string)
		set[s] = true
	}

	return set
}

func uniqueInts(rr Rows, col int) IntSet {
	set := make(IntSet)
	for _, r := range rr {
		s := r[col].(int)
		set[s] = true
	}

	return set
}

func uniqueFloats(rr Rows, col int) FloatSet {
	set := make(FloatSet)
	for _, r := range rr {
		s := r[col].(float64)
		set[s] = true
	}

	return set
}

func partition(rr Rows, q Question) (Rows, Rows) {
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

func gini(rr Rows) float64 {
	impurity := 1.0
	counts := classCounts(rr)
	var rowLen = float64(len(rr))

	for _, c := range counts {
		prob := float64(c) / rowLen
		impurity -= math.Pow(prob, 2.0)
	}

	return impurity
}

func infoGain(left, right Rows, uncertainty float64) float64 {
	leftLen := float64(len(left))
	rightLen := float64(len(right))

	p := leftLen / (leftLen + rightLen)
	return uncertainty - p*gini(left) - (1-p)*gini(right)
}

func findBestSplit(rr Rows, header []string) (float64, *Question) {
	var bestGain float64
	var bestQuestion *Question
	uncertainty := gini(rr)
	nFeatures := len(rr[0]) - 1

	for col := 0; col < nFeatures; col++ {
		vv := uniqueVals(rr, col)

		for _, v := range vv {
			q := Question{Column: col, Value: v, header: header}
			trueRows, falseRows := partition(rr, q)
			if len(trueRows) == 0 || len(falseRows) == 0 {
				continue
			}

			gain := infoGain(trueRows, falseRows, uncertainty)
			if gain >= bestGain {
				bestGain = gain
				bestQuestion = &q
			}
		}
	}

	return bestGain, bestQuestion
}
