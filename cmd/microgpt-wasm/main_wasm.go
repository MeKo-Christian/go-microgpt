//go:build js && wasm

package main

import (
	"errors"
	"fmt"
	"math/rand"
	"strings"
	"sync"
	"sync/atomic"
	"syscall/js"
	"time"

	microgpt "go-microgpt"
)

type appState struct {
	mu sync.Mutex

	seed      int64
	rng       *rand.Rand
	docs      []string
	tokenizer *microgpt.Tokenizer
	model     *microgpt.Model

	modelCfg microgpt.ModelConfig

	training      atomic.Bool
	stopRequested atomic.Bool
}

var state = &appState{
	seed: 42,
	modelCfg: microgpt.ModelConfig{
		NLayer:    1,
		NEmbd:     16,
		BlockSize: 16,
		NHead:     4,
	},
}

type trainOptions struct {
	NumSteps       int
	LearningRate   float64
	Beta1          float64
	Beta2          float64
	EpsAdam        float64
	LiveSampleTemp float64
	SampleEvery    int
	FinalSamples   int
}

func defaultTrainOptions() trainOptions {
	base := microgpt.DefaultTrainOptions()
	return trainOptions{
		NumSteps:       base.NumSteps,
		LearningRate:   base.LearningRate,
		Beta1:          base.Beta1,
		Beta2:          base.Beta2,
		EpsAdam:        base.EpsAdam,
		LiveSampleTemp: 0.5,
		SampleEvery:    50,
		FinalSamples:   10,
	}
}

func main() {
	kernel := map[string]any{
		"version":     "0.1.0-wasm",
		"loadDataset": js.FuncOf(loadDataset),
		"initModel":   js.FuncOf(initModel),
		"train":       js.FuncOf(trainAsync),
		"generate":    js.FuncOf(generateAsync),
		"stop":        js.FuncOf(stopTraining),
	}

	js.Global().Set("MicroGPTKernel", js.ValueOf(kernel))
	println("MicroGPT wasm kernel loaded")
	select {}
}

func loadDataset(_ js.Value, args []js.Value) any {
	if state.training.Load() {
		return errResult("cannot load dataset while training")
	}

	if len(args) < 1 {
		return errResult("loadDataset requires dataset text")
	}

	text := args[0].String()
	docs := parseDocs(text)
	if len(docs) == 0 {
		return errResult("dataset is empty")
	}

	seed := int64(42)
	if len(args) > 1 && isNumber(args[1]) {
		seed = int64(args[1].Int())
	}

	rng := rand.New(rand.NewSource(seed))
	microgpt.ShuffleStrings(rng, docs)
	tokenizer := microgpt.NewTokenizer(docs)

	state.mu.Lock()
	state.seed = seed
	state.rng = rng
	state.docs = docs
	state.tokenizer = tokenizer
	state.model = nil
	state.mu.Unlock()

	chars := make([]any, len(tokenizer.Chars))
	for i, r := range tokenizer.Chars {
		chars[i] = string(r)
	}

	return okResult(map[string]any{
		"numDocs":    len(docs),
		"vocabSize":  tokenizer.VocabSize(),
		"bos":        tokenizer.BOS,
		"chars":      chars,
		"sampleDocs": docs[:min(5, len(docs))],
	})
}

func initModel(_ js.Value, args []js.Value) any {
	if state.training.Load() {
		return errResult("cannot initialize model while training")
	}

	cfg := state.modelCfg
	if len(args) > 0 && args[0].Type() == js.TypeObject {
		cfg = parseModelConfig(args[0], cfg)
	}

	if cfg.NLayer <= 0 || cfg.NEmbd <= 0 || cfg.BlockSize <= 0 || cfg.NHead <= 0 {
		return errResult("all model dimensions must be > 0")
	}
	if cfg.NEmbd%cfg.NHead != 0 {
		return errResult("n_embd must be divisible by n_head")
	}

	state.mu.Lock()
	defer state.mu.Unlock()

	if len(state.docs) == 0 || state.tokenizer == nil {
		return errResult("dataset not loaded")
	}
	if state.rng == nil {
		state.rng = rand.New(rand.NewSource(state.seed))
	}

	state.modelCfg = cfg
	state.model = microgpt.NewModel(cfg, state.tokenizer.VocabSize(), state.rng)

	return okResult(map[string]any{
		"numParams": len(state.model.Params),
		"config": map[string]any{
			"nLayer":    cfg.NLayer,
			"nEmbd":     cfg.NEmbd,
			"blockSize": cfg.BlockSize,
			"nHead":     cfg.NHead,
		},
	})
}

func trainAsync(_ js.Value, args []js.Value) any {
	promiseCtor := js.Global().Get("Promise")
	var handler js.Func
	handler = js.FuncOf(func(_ js.Value, pArgs []js.Value) any {
		defer handler.Release()
		resolve := pArgs[0]
		reject := pArgs[1]

		opts := defaultTrainOptions()
		if len(args) > 0 && args[0].Type() == js.TypeObject {
			opts = parseTrainOptions(args[0], opts)
		}

		var progressCB js.Value
		if len(args) > 1 && args[1].Type() == js.TypeFunction {
			progressCB = args[1]
		}

		go func() {
			res, err := runTraining(opts, progressCB)
			if err != nil {
				reject.Invoke(err.Error())
				return
			}
			resolve.Invoke(js.ValueOf(res))
		}()
		return nil
	})
	return promiseCtor.New(handler)
}

func generateAsync(_ js.Value, args []js.Value) any {
	promiseCtor := js.Global().Get("Promise")
	var handler js.Func
	handler = js.FuncOf(func(_ js.Value, pArgs []js.Value) any {
		defer handler.Release()
		resolve := pArgs[0]
		reject := pArgs[1]

		count := 8
		temp := 0.5
		if len(args) > 0 && args[0].Type() == js.TypeObject {
			v := args[0]
			if x := pickNumber(v, "count", "numSamples", "samples"); x > 0 {
				count = int(x)
			}
			if x := pickNumber(v, "temperature", "temp"); x > 0 {
				temp = x
			}
		}

		go func() {
			res, err := runGenerate(count, temp)
			if err != nil {
				reject.Invoke(err.Error())
				return
			}
			resolve.Invoke(js.ValueOf(res))
		}()
		return nil
	})
	return promiseCtor.New(handler)
}

func stopTraining(_ js.Value, _ []js.Value) any {
	state.stopRequested.Store(true)
	return okResult(map[string]any{"stopping": true})
}

func runTraining(opts trainOptions, progressCB js.Value) (map[string]any, error) {
	if opts.NumSteps <= 0 {
		return nil, fmt.Errorf("steps must be > 0, got %d", opts.NumSteps)
	}
	if opts.SampleEvery <= 0 {
		opts.SampleEvery = 50
	}
	if opts.LiveSampleTemp <= 0 {
		opts.LiveSampleTemp = 0.5
	}
	if opts.FinalSamples <= 0 {
		opts.FinalSamples = 10
	}

	if state.training.Swap(true) {
		return nil, errors.New("training already in progress")
	}
	defer state.training.Store(false)
	state.stopRequested.Store(false)

	state.mu.Lock()
	model := state.model
	tokenizer := state.tokenizer
	docs := state.docs
	rng := state.rng
	cfg := state.modelCfg
	state.mu.Unlock()

	if model == nil {
		return nil, errors.New("model not initialized")
	}
	if tokenizer == nil || len(docs) == 0 {
		return nil, errors.New("dataset not loaded")
	}
	if rng == nil {
		return nil, errors.New("rng not initialized")
	}

	adam := microgpt.NewAdam(len(model.Params), opts.LearningRate, opts.Beta1, opts.Beta2, opts.EpsAdam)
	start := time.Now()

	for step := 0; step < opts.NumSteps; step++ {
		if state.stopRequested.Load() {
			return map[string]any{
				"stopped":     true,
				"stepsDone":   step,
				"totalSteps":  opts.NumSteps,
				"totalTimeMs": time.Since(start).Milliseconds(),
			}, nil
		}

		stepStart := time.Now()

		doc := docs[step%len(docs)]
		tokens := tokenizer.EncodeWithBOS(doc)
		n := cfg.BlockSize
		if len(tokens)-1 < n {
			n = len(tokens) - 1
		}

		cache := microgpt.NewKVCache(cfg.NLayer)
		losses := make(microgpt.Vec, 0, n)
		for posID := 0; posID < n; posID++ {
			tokenID := tokens[posID]
			targetID := tokens[posID+1]
			logits := model.ForwardToken(tokenID, posID, cache)
			probs := microgpt.Softmax(logits)
			losses = append(losses, probs[targetID].Log().Neg())
		}

		loss := microgpt.NewValue(0)
		for _, lt := range losses {
			loss = loss.Add(lt)
		}
		loss = loss.MulScalar(1.0 / float64(n))

		loss.Backward()
		adam.Step(model.Params, step, opts.NumSteps)

		sample := ""
		if step%opts.SampleEvery == 0 || step == opts.NumSteps-1 {
			out, err := microgpt.GenerateSamples(model, tokenizer, microgpt.SampleOptions{
				NumSamples:  1,
				Temperature: opts.LiveSampleTemp,
			}, rng)
			if err == nil && len(out) > 0 {
				sample = out[0]
			}
		}

		if progressCB.Type() == js.TypeFunction {
			payload := map[string]any{
				"step":       step + 1,
				"totalSteps": opts.NumSteps,
				"loss":       loss.Data,
				"stepTimeMs": time.Since(stepStart).Milliseconds(),
				"elapsedMs":  time.Since(start).Milliseconds(),
				"sample":     sample,
			}
			safeInvoke(progressCB, payload)
		}

		if step%5 == 0 {
			time.Sleep(time.Millisecond)
		}
	}

	samples, err := microgpt.GenerateSamples(model, tokenizer, microgpt.SampleOptions{
		NumSamples:  opts.FinalSamples,
		Temperature: opts.LiveSampleTemp,
	}, rng)
	if err != nil {
		return nil, err
	}

	return map[string]any{
		"stopped":     false,
		"samples":     stringSliceToAny(samples),
		"totalSteps":  opts.NumSteps,
		"totalTimeMs": time.Since(start).Milliseconds(),
	}, nil
}

func runGenerate(count int, temp float64) (map[string]any, error) {
	if count <= 0 {
		return nil, fmt.Errorf("count must be > 0, got %d", count)
	}
	if temp <= 0 {
		return nil, fmt.Errorf("temperature must be > 0, got %f", temp)
	}
	if state.training.Load() {
		return nil, errors.New("cannot generate while training is running")
	}

	state.mu.Lock()
	model := state.model
	tokenizer := state.tokenizer
	rng := state.rng
	state.mu.Unlock()

	if model == nil || tokenizer == nil || rng == nil {
		return nil, errors.New("model is not ready")
	}

	out, err := microgpt.GenerateSamples(model, tokenizer, microgpt.SampleOptions{
		NumSamples:  count,
		Temperature: temp,
	}, rng)
	if err != nil {
		return nil, err
	}

	return map[string]any{
		"samples": stringSliceToAny(out),
	}, nil
}

func parseDocs(text string) []string {
	lines := strings.Split(text, "\n")
	out := make([]string, 0, len(lines))
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line != "" {
			out = append(out, line)
		}
	}
	return out
}

func parseModelConfig(v js.Value, base microgpt.ModelConfig) microgpt.ModelConfig {
	cfg := base
	if x := pickNumber(v, "nLayer", "n_layer"); x > 0 {
		cfg.NLayer = int(x)
	}
	if x := pickNumber(v, "nEmbd", "n_embd"); x > 0 {
		cfg.NEmbd = int(x)
	}
	if x := pickNumber(v, "blockSize", "block_size"); x > 0 {
		cfg.BlockSize = int(x)
	}
	if x := pickNumber(v, "nHead", "n_head"); x > 0 {
		cfg.NHead = int(x)
	}
	return cfg
}

func parseTrainOptions(v js.Value, base trainOptions) trainOptions {
	opts := base
	if x := pickNumber(v, "steps", "numSteps", "num_steps"); x > 0 {
		opts.NumSteps = int(x)
	}
	if x := pickNumber(v, "learningRate", "learning_rate"); x > 0 {
		opts.LearningRate = x
	}
	if x := pickNumber(v, "beta1"); x > 0 {
		opts.Beta1 = x
	}
	if x := pickNumber(v, "beta2"); x > 0 {
		opts.Beta2 = x
	}
	if x := pickNumber(v, "epsAdam", "eps_adam"); x > 0 {
		opts.EpsAdam = x
	}
	if x := pickNumber(v, "liveSampleTemp", "live_sample_temp", "temperature"); x > 0 {
		opts.LiveSampleTemp = x
	}
	if x := pickNumber(v, "sampleEvery", "sample_every"); x > 0 {
		opts.SampleEvery = int(x)
	}
	if x := pickNumber(v, "finalSamples", "final_samples"); x > 0 {
		opts.FinalSamples = int(x)
	}
	return opts
}

func isNumber(v js.Value) bool {
	return v.Type() == js.TypeNumber
}

func pickNumber(v js.Value, keys ...string) float64 {
	for _, key := range keys {
		field := v.Get(key)
		if field.IsUndefined() || field.IsNull() || field.Type() != js.TypeNumber {
			continue
		}
		return field.Float()
	}
	return 0
}

func safeInvoke(fn js.Value, payload any) {
	defer func() {
		_ = recover()
	}()
	fn.Invoke(js.ValueOf(payload))
}

func okResult(data map[string]any) map[string]any {
	out := map[string]any{"ok": true}
	for k, v := range data {
		out[k] = v
	}
	return out
}

func errResult(msg string) map[string]any {
	return map[string]any{
		"ok":    false,
		"error": msg,
	}
}

func stringSliceToAny(xs []string) []any {
	out := make([]any, len(xs))
	for i, x := range xs {
		out[i] = x
	}
	return out
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
