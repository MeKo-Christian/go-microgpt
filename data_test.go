package microgpt

import (
	"math/rand"
	"os"
	"path/filepath"
	"testing"
)

func TestTokenizerBasics(t *testing.T) {
	tok := NewTokenizer([]string{"ba", "ac"})

	if tok.BOS != 3 {
		t.Fatalf("BOS = %d, want 3", tok.BOS)
	}
	if got, want := tok.VocabSize(), 4; got != want {
		t.Fatalf("VocabSize = %d, want %d", got, want)
	}

	got := tok.EncodeWithBOS("cab")
	want := []int{3, 2, 0, 1, 3}
	if len(got) != len(want) {
		t.Fatalf("encoded len = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("encoded[%d] = %d, want %d", i, got[i], want[i])
		}
	}
}

func TestLoadDocsFromFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "input.txt")
	content := "\n  anna  \n\nbob\n   \ncarla\n"
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write test input: %v", err)
	}

	docs, err := LoadDocs(path)
	if err != nil {
		t.Fatalf("LoadDocs error: %v", err)
	}

	want := []string{"anna", "bob", "carla"}
	if len(docs) != len(want) {
		t.Fatalf("docs len = %d, want %d", len(docs), len(want))
	}
	for i := range want {
		if docs[i] != want[i] {
			t.Fatalf("docs[%d] = %q, want %q", i, docs[i], want[i])
		}
	}
}

func TestSampleWeightedEdgeCases(t *testing.T) {
	rng := rand.New(rand.NewSource(1))

	for range 100 {
		got := SampleWeighted(rng, []float64{0, 0, 5, 0})
		if got != 2 {
			t.Fatalf("SampleWeighted selected %d, want 2", got)
		}
	}

	for range 100 {
		got := SampleWeighted(rng, []float64{0, 0, 0})
		if got < 0 || got >= 3 {
			t.Fatalf("fallback index out of range: %d", got)
		}
	}
}

func TestDownloadFileInvalidURL(t *testing.T) {
	dst := filepath.Join(t.TempDir(), "download.txt")
	if err := downloadFile("://invalid-url", dst); err == nil {
		t.Fatal("expected error for invalid URL, got nil")
	}
}

func TestShuffleStringsPreservesElements(t *testing.T) {
	xs := []string{"a", "b", "c", "d", "e"}
	in := append([]string(nil), xs...)

	rng := rand.New(rand.NewSource(7))
	ShuffleStrings(rng, xs)

	seen := make(map[string]int, len(xs))
	for _, x := range xs {
		seen[x]++
	}
	for _, x := range in {
		if seen[x] != 1 {
			t.Fatalf("element %q count = %d, want 1", x, seen[x])
		}
	}
}
