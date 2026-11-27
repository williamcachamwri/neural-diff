package safetensors

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
)

// MaxHeaderBytes caps `.safetensors` JSON header allocation at 64 MiB for safety.
const MaxHeaderBytes = 64 * 1024 * 1024

// TensorSpec describes a single tensor parsed from a `.safetensors` header without loading weights.
type TensorSpec struct {
	Name        string  `json:"name"`
	DType       string  `json:"dtype"`
	Shape       []int64 `json:"shape"`
	DataOffsets [2]int64 `json:"data_offsets"`
	ParamCount  int64   `json:"parameter_count"`
	ByteLength  int64   `json:"byte_length"`
	ShardFile   string  `json:"shard_file,omitempty"`
}

// CheckpointSummary matches the metadata structure of `neural-diff inspect` while adding fast tensor listings.
type CheckpointSummary struct {
	ModelID         string            `json:"model_id"`
	ResolvedPath    string            `json:"resolved_path"`
	Format          string            `json:"format"`
	Architecture    string            `json:"architecture,omitempty"`
	ModelType       string            `json:"model_type,omitempty"`
	NumHiddenLayers *int              `json:"num_hidden_layers,omitempty"`
	HiddenSize      *int              `json:"hidden_size,omitempty"`
	VocabSize       *int              `json:"vocab_size,omitempty"`
	TensorCount     int               `json:"tensor_count"`
	ParameterCount  int64             `json:"total_parameters"`
	FileSizeBytes   int64             `json:"total_bytes"`
	DTypes          map[string]int    `json:"dtypes"`
	Metadata        map[string]string `json:"metadata,omitempty"`
	HeaderSHA256    string            `json:"header_sha256"`
	Tensors         []TensorSpec      `json:"tensors,omitempty"`
}

// QuickDiffSummary compares two `.safetensors` checkpoints strictly via binary headers in milliseconds.
type QuickDiffSummary struct {
	ModelA              string             `json:"model_a"`
	ModelB              string             `json:"model_b"`
	Compatible          bool               `json:"compatible"`
	Status              string             `json:"status"`
	AlignedCount        int                `json:"aligned_tensors"`
	OnlyInACount        int                `json:"only_in_a_count"`
	OnlyInBCount        int                `json:"only_in_b_count"`
	ShapeMismatchCount  int                `json:"shape_mismatch_count"`
	DTypeMismatchCount  int                `json:"dtype_mismatch_count"`
	ParamsA             int64              `json:"parameter_count_a"`
	ParamsB             int64              `json:"parameter_count_b"`
	ParamsDelta         int64              `json:"parameter_delta"`
	OnlyInA             []string           `json:"only_in_a"`
	OnlyInB             []string           `json:"only_in_b"`
	ShapeMismatches     []TensorShapeDelta `json:"shape_mismatches"`
	DTypeMismatches     []TensorDTypeDelta `json:"dtype_mismatches"`
}

// TensorShapeDelta records a shape discrepancy between checkpoints A and B.
type TensorShapeDelta struct {
	Name   string  `json:"name"`
	ShapeA []int64 `json:"shape_a"`
	ShapeB []int64 `json:"shape_b"`
}

// TensorDTypeDelta records a dtype change between checkpoints A and B.
type TensorDTypeDelta struct {
	Name   string `json:"name"`
	DTypeA string `json:"dtype_a"`
	DTypeB string `json:"dtype_b"`
}

type rawTensorEntry struct {
	DType       string   `json:"dtype"`
	Shape       []int64  `json:"shape"`
	DataOffsets [2]int64 `json:"data_offsets"`
}

type shardIndexFile struct {
	WeightMap map[string]string `json:"weight_map"`
}

// InspectSafetensors inspects a single `.safetensors` file or Hugging Face checkpoint directory natively in Go.
// Returns an error if the target is not a local `.safetensors` file/directory so callers can fall back to Python.
func InspectSafetensors(target string, includeTensors bool) (*CheckpointSummary, error) {
	clean := filepath.Clean(target)
	info, err := os.Stat(clean)
	if err != nil {
		return nil, err
	}
	absPath, err := filepath.Abs(clean)
	if err != nil {
		absPath = clean
	}

	summary := &CheckpointSummary{
		ModelID:      filepath.Base(absPath),
		ResolvedPath: absPath,
		DTypes:       make(map[string]int),
		Metadata:     make(map[string]string),
	}

	hasher := sha256.New()
	var specs []TensorSpec

	if !info.IsDir() {
		if !strings.HasSuffix(strings.ToLower(info.Name()), ".safetensors") {
			return nil, fmt.Errorf("not a .safetensors file: %s", info.Name())
		}
		summary.Format = "safetensors"
		summary.FileSizeBytes = info.Size()
		fileSpecs, meta, hdrBytes, err := parseSingleSafetensorsFile(absPath, "")
		if err != nil {
			return nil, err
		}
		hasher.Write(hdrBytes)
		for k, v := range meta {
			summary.Metadata[k] = v
		}
		specs = append(specs, fileSpecs...)
	} else {
		// Directory: inspect optional config.json + single or sharded safetensors.
		loadOptionalHFConfig(absPath, summary)
		shards, formatName, err := resolveDirectorySafetensors(absPath)
		if err != nil {
			return nil, err
		}
		summary.Format = formatName
		seenNames := make(map[string]bool)
		for _, shardPath := range shards {
			st, err := os.Stat(shardPath)
			if err != nil {
				return nil, err
			}
			summary.FileSizeBytes += st.Size()
			relName := filepath.Base(shardPath)
			fileSpecs, meta, hdrBytes, err := parseSingleSafetensorsFile(shardPath, relName)
			if err != nil {
				return nil, err
			}
			hasher.Write([]byte(relName))
			hasher.Write([]byte{0})
			hasher.Write(hdrBytes)
			for k, v := range meta {
				if _, exists := summary.Metadata[k]; !exists {
					summary.Metadata[k] = v
				}
			}
			for _, sp := range fileSpecs {
				if !seenNames[sp.Name] {
					seenNames[sp.Name] = true
					specs = append(specs, sp)
				}
			}
		}
	}

	sort.Slice(specs, func(i, j int) bool {
		return specs[i].Name < specs[j].Name
	})

	for _, sp := range specs {
		summary.ParameterCount += sp.ParamCount
		summary.DTypes[sp.DType]++
	}
	summary.TensorCount = len(specs)
	summary.HeaderSHA256 = hex.EncodeToString(hasher.Sum(nil))
	if includeTensors {
		summary.Tensors = specs
	}
	return summary, nil
}

// CompareSafetensorsHeaders computes an instant structural diff between two `.safetensors` checkpoints.
func CompareSafetensorsHeaders(pathA, pathB string) (*QuickDiffSummary, error) {
	sumA, err := InspectSafetensors(pathA, true)
	if err != nil {
		return nil, fmt.Errorf("model A: %w", err)
	}
	sumB, err := InspectSafetensors(pathB, true)
	if err != nil {
		return nil, fmt.Errorf("model B: %w", err)
	}

	mapA := make(map[string]TensorSpec, len(sumA.Tensors))
	for _, t := range sumA.Tensors {
		mapA[t.Name] = t
	}
	mapB := make(map[string]TensorSpec, len(sumB.Tensors))
	for _, t := range sumB.Tensors {
		mapB[t.Name] = t
	}

	out := &QuickDiffSummary{
		ModelA:          sumA.ModelID,
		ModelB:          sumB.ModelID,
		ParamsA:         sumA.ParameterCount,
		ParamsB:         sumB.ParameterCount,
		ParamsDelta:     sumB.ParameterCount - sumA.ParameterCount,
		OnlyInA:         []string{},
		OnlyInB:         []string{},
		ShapeMismatches: []TensorShapeDelta{},
		DTypeMismatches: []TensorDTypeDelta{},
	}

	for _, tA := range sumA.Tensors {
		tB, ok := mapB[tA.Name]
		if !ok {
			out.OnlyInA = append(out.OnlyInA, tA.Name)
			continue
		}
		if !reflect.DeepEqual(tA.Shape, tB.Shape) {
			out.ShapeMismatches = append(out.ShapeMismatches, TensorShapeDelta{
				Name:   tA.Name,
				ShapeA: tA.Shape,
				ShapeB: tB.Shape,
			})
			continue
		}
		out.AlignedCount++
		if tA.DType != tB.DType {
			out.DTypeMismatches = append(out.DTypeMismatches, TensorDTypeDelta{
				Name:   tA.Name,
				DTypeA: tA.DType,
				DTypeB: tB.DType,
			})
		}
	}

	for _, tB := range sumB.Tensors {
		if _, ok := mapA[tB.Name]; !ok {
			out.OnlyInB = append(out.OnlyInB, tB.Name)
		}
	}

	out.OnlyInACount = len(out.OnlyInA)
	out.OnlyInBCount = len(out.OnlyInB)
	out.ShapeMismatchCount = len(out.ShapeMismatches)
	out.DTypeMismatchCount = len(out.DTypeMismatches)

	switch {
	case out.AlignedCount == 0:
		out.Compatible = false
		out.Status = "INCOMPATIBLE"
	case out.OnlyInACount == 0 && out.OnlyInBCount == 0 && out.ShapeMismatchCount == 0:
		out.Compatible = true
		out.Status = "COMPATIBLE"
	default:
		out.Compatible = true
		out.Status = "PARTIAL"
	}

	return out, nil
}

func resolveDirectorySafetensors(dir string) ([]string, string, error) {
	indexFile := filepath.Join(dir, "model.safetensors.index.json")
	if st, err := os.Stat(indexFile); err == nil && !st.IsDir() {
		raw, err := os.ReadFile(indexFile)
		if err != nil {
			return nil, "", err
		}
		var idx shardIndexFile
		if err := json.Unmarshal(raw, &idx); err != nil {
			return nil, "", fmt.Errorf("malformed safetensors index: %w", err)
		}
		shardSet := make(map[string]bool)
		var shards []string
		for _, rel := range idx.WeightMap {
			cleanRel := filepath.Base(rel)
			if cleanRel != rel || strings.Contains(rel, "..") {
				return nil, "", fmt.Errorf("unsafe shard path in index: %s", rel)
			}
			full := filepath.Join(dir, cleanRel)
			if !shardSet[full] {
				shardSet[full] = true
				shards = append(shards, full)
			}
		}
		sort.Strings(shards)
		if len(shards) == 0 {
			return nil, "", errors.New("safetensors index has empty weight_map")
		}
		return shards, "hf_sharded_safetensors", nil
	}

	primary := filepath.Join(dir, "model.safetensors")
	if st, err := os.Stat(primary); err == nil && !st.IsDir() {
		return []string{primary}, "hf_safetensors", nil
	}

	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, "", err
	}
	var found []string
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		lower := strings.ToLower(e.Name())
		if strings.HasSuffix(lower, ".safetensors") && !strings.HasPrefix(lower, "adapter_") {
			found = append(found, filepath.Join(dir, e.Name()))
		}
	}
	sort.Strings(found)
	if len(found) == 0 {
		return nil, "", errors.New("no .safetensors weights found in directory")
	}
	return found, "safetensors_dir", nil
}

func parseSingleSafetensorsFile(path string, shardName string) ([]TensorSpec, map[string]string, []byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, nil, err
	}
	defer f.Close()

	st, err := f.Stat()
	if err != nil {
		return nil, nil, nil, err
	}
	if st.Size() < 8 {
		return nil, nil, nil, fmt.Errorf("truncated safetensors file (< 8 bytes): %s", path)
	}

	var lenBuf [8]byte
	if _, err := io.ReadFull(f, lenBuf[:]); err != nil {
		return nil, nil, nil, err
	}
	hdrLen := binary.LittleEndian.Uint64(lenBuf[:])
	if hdrLen == 0 || hdrLen > MaxHeaderBytes {
		return nil, nil, nil, fmt.Errorf("invalid safetensors header length %d in %s", hdrLen, path)
	}
	if int64(hdrLen)+8 > st.Size() {
		return nil, nil, nil, fmt.Errorf("safetensors header exceeds file size in %s", path)
	}

	hdrBytes := make([]byte, hdrLen)
	if _, err := io.ReadFull(f, hdrBytes); err != nil {
		return nil, nil, nil, err
	}

	var rawMap map[string]json.RawMessage
	if err := json.Unmarshal(hdrBytes, &rawMap); err != nil {
		return nil, nil, nil, fmt.Errorf("invalid safetensors JSON header: %w", err)
	}

	meta := make(map[string]string)
	specs := make([]TensorSpec, 0, len(rawMap))
	payloadBytes := st.Size() - 8 - int64(hdrLen)

	for key, rawVal := range rawMap {
		if key == "__metadata__" {
			_ = json.Unmarshal(rawVal, &meta)
			continue
		}
		var entry rawTensorEntry
		if err := json.Unmarshal(rawVal, &entry); err != nil {
			return nil, nil, nil, fmt.Errorf("malformed tensor entry %q: %w", key, err)
		}
		if entry.DataOffsets[0] < 0 || entry.DataOffsets[1] < entry.DataOffsets[0] || entry.DataOffsets[1] > payloadBytes {
			return nil, nil, nil, fmt.Errorf("out-of-bounds data_offsets for tensor %q", key)
		}
		params := int64(1)
		if len(entry.Shape) == 0 {
			params = 1
		} else {
			for _, dim := range entry.Shape {
				if dim < 0 {
					return nil, nil, nil, fmt.Errorf("negative dimension in tensor %q", key)
				}
				params *= dim
			}
		}
		specs = append(specs, TensorSpec{
			Name:        key,
			DType:       entry.DType,
			Shape:       entry.Shape,
			DataOffsets: entry.DataOffsets,
			ParamCount:  params,
			ByteLength:  entry.DataOffsets[1] - entry.DataOffsets[0],
			ShardFile:   shardName,
		})
	}

	return specs, meta, hdrBytes, nil
}

func loadOptionalHFConfig(dir string, summary *CheckpointSummary) {
	cfgPath := filepath.Join(dir, "config.json")
	raw, err := os.ReadFile(cfgPath)
	if err != nil {
		return
	}
	var cfg struct {
		Architectures   []string `json:"architectures"`
		ModelType       string   `json:"model_type"`
		NumHiddenLayers *int     `json:"num_hidden_layers"`
		HiddenSize      *int     `json:"hidden_size"`
		VocabSize       *int     `json:"vocab_size"`
	}
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return
	}
	if len(cfg.Architectures) > 0 {
		summary.Architecture = cfg.Architectures[0]
	}
	summary.ModelType = cfg.ModelType
	summary.NumHiddenLayers = cfg.NumHiddenLayers
	summary.HiddenSize = cfg.HiddenSize
	summary.VocabSize = cfg.VocabSize
}
