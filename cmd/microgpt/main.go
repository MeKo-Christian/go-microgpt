package main

import (
	"fmt"
	"math/rand"
	"os"

	"github.com/spf13/cobra"
	microgpt "go-microgpt"
)

func main() {
	var (
		inputPath  string
		seed       int64
		modelCfg   = microgpt.ModelConfig{NLayer: 1, NEmbd: 16, BlockSize: 16, NHead: 4}
		trainOpts  = microgpt.DefaultTrainOptions()
		sampleOpts = microgpt.DefaultSampleOptions()
	)

	cmd := &cobra.Command{
		Use:          "microgpt",
		Short:        "Train and sample a tiny character-level GPT",
		SilenceUsage: true,
		RunE: func(cmd *cobra.Command, args []string) error {
			if sampleOpts.Temperature <= 0 {
				return fmt.Errorf("temperature must be > 0, got %f", sampleOpts.Temperature)
			}
			if trainOpts.NumSteps <= 0 {
				return fmt.Errorf("steps must be > 0, got %d", trainOpts.NumSteps)
			}
			if modelCfg.NLayer <= 0 || modelCfg.NEmbd <= 0 || modelCfg.BlockSize <= 0 || modelCfg.NHead <= 0 {
				return fmt.Errorf("all model dimensions must be > 0")
			}
			if modelCfg.NEmbd%modelCfg.NHead != 0 {
				return fmt.Errorf("n-embd (%d) must be divisible by n-head (%d)", modelCfg.NEmbd, modelCfg.NHead)
			}

			rng := rand.New(rand.NewSource(seed))

			docs, err := microgpt.LoadDocs(inputPath)
			if err != nil {
				return fmt.Errorf("load docs: %w", err)
			}
			if len(docs) == 0 {
				return fmt.Errorf("no documents found in %s", inputPath)
			}
			microgpt.ShuffleStrings(rng, docs)
			fmt.Printf("num docs: %d\n", len(docs))

			tokenizer := microgpt.NewTokenizer(docs)
			fmt.Printf("vocab size: %d\n", tokenizer.VocabSize())

			model := microgpt.NewModel(modelCfg, tokenizer.VocabSize(), rng)
			fmt.Printf("num params: %d\n", len(model.Params))

			err = microgpt.Train(model, tokenizer, docs, trainOpts, func(metrics microgpt.StepMetrics) {
				fmt.Printf("step %4d / %4d | loss %.4f\r", metrics.Step, metrics.Total, metrics.Loss)
			})
			if err != nil {
				return fmt.Errorf("train model: %w", err)
			}

			fmt.Println("\n--- inference (new, hallucinated names) ---")
			samples, err := microgpt.GenerateSamples(model, tokenizer, sampleOpts, rng)
			if err != nil {
				return fmt.Errorf("generate samples: %w", err)
			}

			for i, sample := range samples {
				fmt.Printf("sample %2d: %s\n", i+1, sample)
			}
			return nil
		},
	}

	cmd.Flags().StringVar(&inputPath, "input", "input.txt", "path to training text file (one document per line)")
	cmd.Flags().IntVar(&trainOpts.NumSteps, "steps", trainOpts.NumSteps, "number of training steps")
	cmd.Flags().Float64Var(&sampleOpts.Temperature, "temperature", sampleOpts.Temperature, "sampling temperature in (0, 1]")
	cmd.Flags().IntVar(&sampleOpts.NumSamples, "samples", sampleOpts.NumSamples, "number of generated samples after training")
	cmd.Flags().Int64Var(&seed, "seed", 42, "RNG seed")

	cmd.Flags().IntVar(&modelCfg.NLayer, "n-layer", modelCfg.NLayer, "number of transformer layers")
	cmd.Flags().IntVar(&modelCfg.NEmbd, "n-embd", modelCfg.NEmbd, "embedding dimension")
	cmd.Flags().IntVar(&modelCfg.BlockSize, "block-size", modelCfg.BlockSize, "maximum context length")
	cmd.Flags().IntVar(&modelCfg.NHead, "n-head", modelCfg.NHead, "number of attention heads")

	if err := cmd.Execute(); err != nil {
		os.Exit(1)
	}
}
