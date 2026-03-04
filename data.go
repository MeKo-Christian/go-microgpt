package microgpt

import (
	"bufio"
	"fmt"
	"math/rand"
	"net/http"
	"os"
	"sort"
	"strings"
)

const defaultNamesURL = "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt"

func LoadDocs(inputPath string) ([]string, error) {
	if _, err := os.Stat(inputPath); os.IsNotExist(err) {
		if err := downloadFile(defaultNamesURL, inputPath); err != nil {
			return nil, err
		}
	}

	f, err := os.Open(inputPath)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	docs := make([]string, 0, 4096)
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line != "" {
			docs = append(docs, line)
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return docs, nil
}

func downloadFile(url, dstPath string) error {
	resp, err := http.Get(url) // #nosec G107: fixed trusted URL
	if err != nil {
		return fmt.Errorf("download %s: %w", url, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download %s failed with status %s", url, resp.Status)
	}

	out, err := os.Create(dstPath)
	if err != nil {
		return fmt.Errorf("create %s: %w", dstPath, err)
	}
	defer out.Close()

	if _, err := out.ReadFrom(resp.Body); err != nil {
		return fmt.Errorf("write %s: %w", dstPath, err)
	}
	return nil
}

type Tokenizer struct {
	Chars []rune
	BOS   int
	stoi  map[rune]int
}

func NewTokenizer(docs []string) *Tokenizer {
	uniq := make(map[rune]struct{}, 128)
	for _, doc := range docs {
		for _, ch := range doc {
			uniq[ch] = struct{}{}
		}
	}

	chars := make([]rune, 0, len(uniq))
	for ch := range uniq {
		chars = append(chars, ch)
	}
	sort.Slice(chars, func(i, j int) bool { return chars[i] < chars[j] })

	stoi := make(map[rune]int, len(chars))
	for i, ch := range chars {
		stoi[ch] = i
	}

	return &Tokenizer{
		Chars: chars,
		BOS:   len(chars),
		stoi:  stoi,
	}
}

func (t *Tokenizer) VocabSize() int {
	return len(t.Chars) + 1
}

func (t *Tokenizer) EncodeWithBOS(doc string) []int {
	tokens := make([]int, 0, len(doc)+2)
	tokens = append(tokens, t.BOS)
	for _, ch := range doc {
		tokens = append(tokens, t.stoi[ch])
	}
	tokens = append(tokens, t.BOS)
	return tokens
}

func ShuffleStrings(rng *rand.Rand, xs []string) {
	rng.Shuffle(len(xs), func(i, j int) {
		xs[i], xs[j] = xs[j], xs[i]
	})
}

func SampleWeighted(rng *rand.Rand, weights []float64) int {
	total := 0.0
	for _, w := range weights {
		if w > 0 {
			total += w
		}
	}
	if total <= 0 {
		return rng.Intn(len(weights))
	}

	r := rng.Float64() * total
	acc := 0.0
	for i, w := range weights {
		if w <= 0 {
			continue
		}
		acc += w
		if r <= acc {
			return i
		}
	}
	return len(weights) - 1
}
