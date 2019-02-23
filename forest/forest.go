package forest

import "github.com/nfisher/dtgo/decisiontree"

type Forest struct{}

func Train(rows decisiontree.Rows, header []string) *Forest {
	return &Forest{}
}
