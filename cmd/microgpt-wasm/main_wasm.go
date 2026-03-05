//go:build js && wasm

package main

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"strings"
	"sync"
	"sync/atomic"
	"syscall/js"
	"time"

	microgpt "go-microgpt"
)

// mode selects between classic autograd and optimized tensor training.
const (
	modeClassic = "classic"
	modeTensor  = "tensor"
)

type appState struct {
	mu sync.Mutex

	seed      int64
	rng       *rand.Rand
	docs      []string
	tokenizer *microgpt.Tokenizer

	// One of these is non-nil depending on mode.
	classicModel *microgpt.Model
	tensorModel  *microgpt.TensorModel

	mode      string // "classic" or "tensor"
	modelCfg  microgpt.ModelConfig
	batch     int
	nEmbd     int
	nHead     int
	blockSize int

	training      atomic.Bool
	stopRequested atomic.Bool
}

var state = &appState{
	seed:      42,
	mode:      modeTensor,
	nEmbd:     16,
	nHead:     4,
	blockSize: 16,
	batch:     8,
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
		"version":     "0.3.0-wasm-dual",
		"loadDataset": js.FuncOf(loadDataset),
		"initModel":   js.FuncOf(initModel),
		"train":       js.FuncOf(trainAsync),
		"generate":    js.FuncOf(generateAsync),
		"stop":        js.FuncOf(stopTraining),
	}

	js.Global().Set("MicroGPTKernel", js.ValueOf(kernel))
	println("MicroGPT wasm kernel loaded (dual mode)")
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
	state.classicModel = nil
	state.tensorModel = nil
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
		"sampleDocs": stringSliceToAny(docs[:min(5, len(docs))]),
	})
}

func initModel(_ js.Value, args []js.Value) any {
	if state.training.Load() {
		return errResult("cannot initialize model while training")
	}

	// Parse config from JS.
	nEmbd := state.nEmbd
	nHead := state.nHead
	blockSize := state.blockSize
	nLayer := state.modelCfg.NLayer
	batch := state.batch
	mode := state.mode

	if len(args) > 0 && args[0].Type() == js.TypeObject {
		v := args[0]
		if x := pickNumber(v, "nEmbd", "n_embd"); x > 0 {
			nEmbd = int(x)
		}
		if x := pickNumber(v, "nHead", "n_head"); x > 0 {
			nHead = int(x)
		}
		if x := pickNumber(v, "blockSize", "block_size"); x > 0 {
			blockSize = int(x)
		}
		if x := pickNumber(v, "nLayer", "n_layer"); x > 0 {
			nLayer = int(x)
		}
		if x := pickNumber(v, "batch", "batchSize"); x > 0 {
			batch = int(x)
		}
		if m := v.Get("mode"); m.Type() == js.TypeString {
			mode = m.String()
		}
	}

	if nEmbd <= 0 || blockSize <= 0 || nHead <= 0 {
		return errResult("all model dimensions must be > 0")
	}
	if nEmbd%nHead != 0 {
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

	state.nEmbd = nEmbd
	state.nHead = nHead
	state.blockSize = blockSize
	state.batch = batch
	state.mode = mode
	state.modelCfg = microgpt.ModelConfig{
		NLayer:    nLayer,
		NEmbd:     nEmbd,
		BlockSize: blockSize,
		NHead:     nHead,
	}

	var numParams int

	if mode == modeClassic {
		state.classicModel = microgpt.NewModel(state.modelCfg, state.tokenizer.VocabSize(), state.rng)
		state.tensorModel = nil
		numParams = len(state.classicModel.Params)
	} else {
		cfg := microgpt.TensorConfig{
			DModel:    nEmbd,
			NHeads:    nHead,
			DFF:       4 * nEmbd,
			VocabSize: state.tokenizer.VocabSize(),
			BlockSize: blockSize,
			Batch:     batch,
		}
		state.tensorModel = microgpt.NewTensorModel(cfg, state.rng)
		state.classicModel = nil
		numParams = len(state.tensorModel.AllParams())
	}

	return okResult(map[string]any{
		"numParams": numParams,
		"mode":      mode,
		"config": map[string]any{
			"nLayer":    nLayer,
			"nEmbd":     nEmbd,
			"nHead":     nHead,
			"blockSize": blockSize,
			"batch":     batch,
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
			var res map[string]any
			var err error

			state.mu.Lock()
			mode := state.mode
			state.mu.Unlock()

			if mode == modeClassic {
				res, err = runTrainingClassic(opts, progressCB)
			} else {
				res, err = runTrainingTensor(opts, progressCB)
			}

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

		prompt := ""
		count := 8
		temp := 0.5
		if len(args) > 0 && args[0].Type() == js.TypeObject {
			v := args[0]
			if x := v.Get("prompt"); x.Type() == js.TypeString {
				prompt = x.String()
			}
			if x := pickNumber(v, "count", "numSamples", "samples"); x > 0 {
				count = int(x)
			}
			if x := pickNumber(v, "temperature", "temp"); x > 0 {
				temp = x
			}
		}

		go func() {
			res, err := runGenerate(prompt, count, temp)
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

// ---------------------------------------------------------------------------
// Classic autograd training
// ---------------------------------------------------------------------------

func runTrainingClassic(opts trainOptions, progressCB js.Value) (map[string]any, error) {
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
	model := state.classicModel
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
	lossHistory := make([]float64, 0, opts.NumSteps)
	stepTimeHistory := make([]float64, 0, opts.NumSteps)

	for step := range opts.NumSteps {
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
		n := min(cfg.BlockSize, len(tokens)-1)

		cache := microgpt.NewKVCache(cfg.NLayer)
		losses := make(microgpt.Vec, 0, n)
		for posID := range n {
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

		stepTimeMs := float64(time.Since(stepStart).Milliseconds())
		lossHistory = append(lossHistory, loss.Data)
		stepTimeHistory = append(stepTimeHistory, stepTimeMs)

		if progressCB.Type() == js.TypeFunction {
			payload := map[string]any{
				"step":       step + 1,
				"totalSteps": opts.NumSteps,
				"loss":       loss.Data,
				"stepTimeMs": int64(stepTimeMs),
				"elapsedMs":  time.Since(start).Milliseconds(),
				"sample":     sample,
			}

			if step%5 == 0 || step == opts.NumSteps-1 {
				payload["lossSeries"] = floatSliceToAny(sparklineSeries(lossHistory, 140, true))
				payload["stepTimeSeries"] = floatSliceToAny(sparklineSeries(stepTimeHistory, 140, false))
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
		"stopped":        false,
		"samples":        stringSliceToAny(samples),
		"lossSeries":     floatSliceToAny(sparklineSeries(lossHistory, 140, true)),
		"stepTimeSeries": floatSliceToAny(sparklineSeries(stepTimeHistory, 140, false)),
		"totalSteps":     opts.NumSteps,
		"totalTimeMs":    time.Since(start).Milliseconds(),
	}, nil
}

// ---------------------------------------------------------------------------
// Tensor-optimised training
// ---------------------------------------------------------------------------

func runTrainingTensor(opts trainOptions, progressCB js.Value) (map[string]any, error) {
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
	model := state.tensorModel
	tokenizer := state.tokenizer
	docs := state.docs
	rng := state.rng
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

	cfg := model.Cfg
	tc := microgpt.NewTensorCache(cfg)
	grads := microgpt.NewTensorGrads(cfg)
	paramSlices := model.ParamSlices()
	gradSlices := grads.GradSlices()
	adam := microgpt.NewTensorAdam(
		paramSlices,
		float32(opts.LearningRate),
		float32(opts.Beta1),
		float32(opts.Beta2),
		float32(opts.EpsAdam),
	)

	start := time.Now()
	lossHistory := make([]float64, 0, opts.NumSteps)
	stepTimeHistory := make([]float64, 0, opts.NumSteps)

	for step := range opts.NumSteps {
		if state.stopRequested.Load() {
			return map[string]any{
				"stopped":     true,
				"stepsDone":   step,
				"totalSteps":  opts.NumSteps,
				"totalTimeMs": time.Since(start).Milliseconds(),
			}, nil
		}

		stepStart := time.Now()

		// Fill batch with random documents.
		for b := range cfg.Batch {
			docIdx := rng.Intn(len(docs))
			tokens := tokenizer.EncodeWithBOS(docs[docIdx])
			seqLen := min(len(tokens), cfg.BlockSize)
			tc.SeqLens[b] = seqLen
			for t := range cfg.BlockSize {
				if t < seqLen {
					tc.Tokens[b*cfg.BlockSize+t] = tokens[t]
				} else {
					tc.Tokens[b*cfg.BlockSize+t] = 0
				}
			}
		}

		// Forward -> Backward -> Adam.
		loss := microgpt.Forward(model, tc)
		grads.Zero()
		microgpt.Backward(model, tc, grads)
		adam.Step(paramSlices, gradSlices, step+1, opts.NumSteps)
		microgpt.SyncTransposedWeights(model)

		// Live sample.
		sample := ""
		if step%opts.SampleEvery == 0 || step == opts.NumSteps-1 {
			sample = microgpt.GenerateFastWithPrompt(model, tokenizer, "", opts.LiveSampleTemp, rng)
		}

		stepTimeMs := float64(time.Since(stepStart).Milliseconds())
		lossHistory = append(lossHistory, float64(loss))
		stepTimeHistory = append(stepTimeHistory, stepTimeMs)

		if progressCB.Type() == js.TypeFunction {
			payload := map[string]any{
				"step":       step + 1,
				"totalSteps": opts.NumSteps,
				"loss":       float64(loss),
				"stepTimeMs": int64(stepTimeMs),
				"elapsedMs":  time.Since(start).Milliseconds(),
				"sample":     sample,
			}

			if step%5 == 0 || step == opts.NumSteps-1 {
				payload["lossSeries"] = floatSliceToAny(sparklineSeries(lossHistory, 140, true))
				payload["stepTimeSeries"] = floatSliceToAny(sparklineSeries(stepTimeHistory, 140, false))
			}

			safeInvoke(progressCB, payload)
		}

		if step%5 == 0 {
			time.Sleep(time.Millisecond)
		}
	}

	// Final samples.
	samples := microgpt.GenerateFast(model, tokenizer, microgpt.SampleOptions{
		NumSamples:  opts.FinalSamples,
		Temperature: opts.LiveSampleTemp,
	}, rng)

	return map[string]any{
		"stopped":        false,
		"samples":        stringSliceToAny(samples),
		"lossSeries":     floatSliceToAny(sparklineSeries(lossHistory, 140, true)),
		"stepTimeSeries": floatSliceToAny(sparklineSeries(stepTimeHistory, 140, false)),
		"totalSteps":     opts.NumSteps,
		"totalTimeMs":    time.Since(start).Milliseconds(),
	}, nil
}

// ---------------------------------------------------------------------------
// Generation (dispatches by mode)
// ---------------------------------------------------------------------------

func runGenerate(prompt string, count int, temp float64) (map[string]any, error) {
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
	mode := state.mode
	classicModel := state.classicModel
	tensorModel := state.tensorModel
	tokenizer := state.tokenizer
	rng := state.rng
	state.mu.Unlock()

	if tokenizer == nil || rng == nil {
		return nil, errors.New("model is not ready")
	}

	out := make([]string, 0, count)

	if mode == modeClassic {
		if classicModel == nil {
			return nil, errors.New("model is not ready")
		}
		for range count {
			out = append(out, generateWithPromptClassic(classicModel, tokenizer, prompt, temp, rng))
		}
	} else {
		if tensorModel == nil {
			return nil, errors.New("model is not ready")
		}
		for range count {
			out = append(out, microgpt.GenerateFastWithPrompt(tensorModel, tokenizer, prompt, temp, rng))
		}
	}

	return map[string]any{
		"prompt":  prompt,
		"samples": stringSliceToAny(out),
	}, nil
}

// generateWithPromptClassic uses the original autograd model for generation.
func generateWithPromptClassic(model *microgpt.Model, tokenizer *microgpt.Tokenizer, prompt string, temperature float64, rng *rand.Rand) string {
	cfg := model.Config()
	if temperature <= 0 {
		temperature = 0.5
	}

	lookup := make(map[rune]int, len(tokenizer.Chars))
	for i, ch := range tokenizer.Chars {
		lookup[ch] = i
	}

	keys := microgpt.NewKVCache(cfg.NLayer)
	tokenID := tokenizer.BOS
	posID := 0
	logits := model.ForwardToken(tokenID, posID, keys)
	posID++

	promptRunes := []rune(prompt)
	for _, ch := range promptRunes {
		id, ok := lookup[ch]
		if !ok || posID >= cfg.BlockSize {
			continue
		}
		tokenID = id
		logits = model.ForwardToken(tokenID, posID, keys)
		posID++
	}

	outRunes := make([]rune, 0, cfg.BlockSize)
	outRunes = append(outRunes, promptRunes...)
	for posID < cfg.BlockSize {
		tokenID = sampleTokenClassic(logits, temperature, rng)
		if tokenID == tokenizer.BOS {
			break
		}
		outRunes = append(outRunes, tokenizer.Chars[tokenID])
		logits = model.ForwardToken(tokenID, posID, keys)
		posID++
	}

	return string(outRunes)
}

func sampleTokenClassic(logits microgpt.Vec, temperature float64, rng *rand.Rand) int {
	scaled := make(microgpt.Vec, len(logits))
	invTemp := 1.0 / temperature
	for i, l := range logits {
		scaled[i] = l.MulScalar(invTemp)
	}
	probs := microgpt.Softmax(scaled)
	weights := make([]float64, len(probs))
	for i, p := range probs {
		weights[i] = p.Data
	}
	return microgpt.SampleWeighted(rng, weights)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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

func floatSliceToAny(xs []float64) []any {
	out := make([]any, len(xs))
	for i, x := range xs {
		out[i] = x
	}
	return out
}

func sparklineSeries(values []float64, maxPoints int, useEMA bool) []float64 {
	if len(values) == 0 || maxPoints <= 0 {
		return []float64{}
	}

	step := max(int(math.Ceil(float64(len(values))/float64(maxPoints))), 1)

	reduced := make([]float64, 0, maxPoints)
	for i := 0; i < len(values); i += step {
		reduced = append(reduced, values[i])
	}
	if reduced[len(reduced)-1] != values[len(values)-1] {
		reduced = append(reduced, values[len(values)-1])
	}

	series := reduced
	if useEMA && len(reduced) > 1 {
		ema := reduced[0]
		series = make([]float64, len(reduced))
		series[0] = ema
		for i := 1; i < len(reduced); i++ {
			ema = 0.93*ema + 0.07*reduced[i]
			series[i] = ema
		}
	}

	minV, maxV := series[0], series[0]
	for _, v := range series[1:] {
		if v < minV {
			minV = v
		}
		if v > maxV {
			maxV = v
		}
	}
	span := maxV - minV
	if span == 0 {
		out := make([]float64, len(series))
		for i := range out {
			out[i] = 0.5
		}
		return out
	}

	out := make([]float64, len(series))
	for i, v := range series {
		out[i] = (v - minV) / span
	}
	return out
}
