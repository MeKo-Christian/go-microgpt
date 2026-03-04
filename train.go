package microgpt

import (
	"errors"
	"fmt"
	"math/rand"
)

type TrainOptions struct {
	NumSteps     int
	LearningRate float64
	Beta1        float64
	Beta2        float64
	EpsAdam      float64
}

func DefaultTrainOptions() TrainOptions {
	return TrainOptions{
		NumSteps:     1000,
		LearningRate: 0.01,
		Beta1:        0.85,
		Beta2:        0.99,
		EpsAdam:      1e-8,
	}
}

type StepMetrics struct {
	Step  int
	Total int
	Loss  float64
}

type ProgressFn func(metrics StepMetrics)

func Train(model *Model, tokenizer *Tokenizer, docs []string, opts TrainOptions, progress ProgressFn) error {
	if model == nil {
		return errors.New("model must not be nil")
	}
	if tokenizer == nil {
		return errors.New("tokenizer must not be nil")
	}
	if len(docs) == 0 {
		return errors.New("docs must not be empty")
	}
	if opts.NumSteps <= 0 {
		return fmt.Errorf("num steps must be > 0, got %d", opts.NumSteps)
	}

	cfg := model.Config()
	adam := NewAdam(len(model.Params), opts.LearningRate, opts.Beta1, opts.Beta2, opts.EpsAdam)
	for step := 0; step < opts.NumSteps; step++ {
		doc := docs[step%len(docs)]
		tokens := tokenizer.EncodeWithBOS(doc)
		n := cfg.BlockSize
		if len(tokens)-1 < n {
			n = len(tokens) - 1
		}

		cache := NewKVCache(cfg.NLayer)
		losses := make(Vec, 0, n)
		for posID := 0; posID < n; posID++ {
			tokenID := tokens[posID]
			targetID := tokens[posID+1]
			logits := model.ForwardToken(tokenID, posID, cache)
			probs := Softmax(logits)
			losses = append(losses, probs[targetID].Log().Neg())
		}

		loss := NewValue(0)
		for _, lt := range losses {
			loss = loss.Add(lt)
		}
		loss = loss.MulScalar(1.0 / float64(n))

		loss.Backward()
		adam.Step(model.Params, step, opts.NumSteps)
		if progress != nil {
			progress(StepMetrics{
				Step:  step + 1,
				Total: opts.NumSteps,
				Loss:  loss.Data,
			})
		}
	}

	return nil
}

type SampleOptions struct {
	NumSamples  int
	Temperature float64
}

func DefaultSampleOptions() SampleOptions {
	return SampleOptions{
		NumSamples:  20,
		Temperature: 0.5,
	}
}

func GenerateSamples(model *Model, tokenizer *Tokenizer, opts SampleOptions, rng *rand.Rand) ([]string, error) {
	if model == nil {
		return nil, errors.New("model must not be nil")
	}
	if tokenizer == nil {
		return nil, errors.New("tokenizer must not be nil")
	}
	if rng == nil {
		return nil, errors.New("rng must not be nil")
	}
	if opts.NumSamples < 0 {
		return nil, fmt.Errorf("num samples must be >= 0, got %d", opts.NumSamples)
	}
	if opts.Temperature <= 0 {
		return nil, fmt.Errorf("temperature must be > 0, got %f", opts.Temperature)
	}

	cfg := model.Config()
	invTemp := 1.0 / opts.Temperature
	samples := make([]string, 0, opts.NumSamples)
	for range opts.NumSamples {
		cache := NewKVCache(cfg.NLayer)
		tokenID := tokenizer.BOS
		sample := make([]rune, 0, cfg.BlockSize)
		for posID := 0; posID < cfg.BlockSize; posID++ {
			logits := model.ForwardToken(tokenID, posID, cache)
			scaled := make(Vec, len(logits))
			for i, l := range logits {
				scaled[i] = l.MulScalar(invTemp)
			}
			probs := Softmax(scaled)
			weights := make([]float64, len(probs))
			for i, p := range probs {
				weights[i] = p.Data
			}

			tokenID = SampleWeighted(rng, weights)
			if tokenID == tokenizer.BOS {
				break
			}
			sample = append(sample, tokenizer.Chars[tokenID])
		}
		samples = append(samples, string(sample))
	}

	return samples, nil
}
