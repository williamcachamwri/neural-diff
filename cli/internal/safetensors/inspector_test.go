package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func writeSyntheticSafetensors(t *testing.T, path string, header map[string]any, payloadSize int) {
	t.Helper()
	hdrBytes, err := json.Marshal(header)
	if err != nil {
		t.Fatalf("marshal header: %v", err)
	}
	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("create file: %v", err)
	}
	defer f.Close()

	var lenBuf [8]byte
	binary.LittleEndian.PutUint64(lenBuf[:], uint64(len(hdrBytes)))
	if _, err := f.Write(lenBuf[:]); err != nil {
		t.Fatalf("write len: %v", err)
	}
	if _, err := f.Write(hdrBytes); err != nil {
		t.Fatalf("write header: %v", err)
	}
	payload := make([]byte, payloadSize)
	if _, err := f.Write(payload); err != nil {
		t.Fatalf("write payload: %v", err)
	}
}

func TestInspectAndQuickDiffSafetensors(t *testing.T) {
	tmp := t.TempDir()
	dirA := filepath.Join(tmp, "model_a")
	dirB := filepath.Join(tmp, "model_b")
	if err := os.MkdirAll(dirA, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(dirB, 0o755); err != nil {
		t.Fatal(err)
	}

	_ = os.WriteFile(filepath.Join(dirA, "config.json"), []byte(`{"architectures":["LlamaForCausalLM"],"num_hidden_layers":2,"hidden_size":16,"vocab_size":64}`), 0o644)
	_ = os.WriteFile(filepath.Join(dirB, "config.json"), []byte(`{"architectures":["LlamaForCausalLM"],"num_hidden_layers":2,"hidden_size":16,"vocab_size":64}`), 0o644)

	hdrA := map[string]any{
		"__metadata__": map[string]string{"format": "pt"},
		"embed.weight": map[string]any{
			"dtype":        "BF16",
			"shape":        []int{64, 16},
			"data_offsets": []int{0, 2048},
		},
		"layers.0.q_proj.weight": map[string]any{
			"dtype":        "BF16",
			"shape":        []int{16, 16},
			"data_offsets": []int{2048, 2560},
		},
	}
	writeSyntheticSafetensors(t, filepath.Join(dirA, "model.safetensors"), hdrA, 2560)
	writeSyntheticSafetensors(t, filepath.Join(dirB, "model.safetensors"), hdrA, 2560)

	sumA, err := InspectSafetensors(dirA, true)
	if err != nil {
		t.Fatalf("InspectSafetensors failed: %v", err)
	}
	if sumA.TensorCount != 2 {
		t.Fatalf("expected 2 tensors, got %d", sumA.TensorCount)
	}
	if sumA.ParameterCount != 64*16+16*16 {
		t.Fatalf("unexpected parameter count: %d", sumA.ParameterCount)
	}
	if sumA.Architecture != "LlamaForCausalLM" {
		t.Fatalf("unexpected architecture: %s", sumA.Architecture)
	}

	diffEq, err := CompareSafetensorsHeaders(dirA, dirB)
	if err != nil {
		t.Fatalf("CompareSafetensorsHeaders failed: %v", err)
	}
	if diffEq.Status != "COMPATIBLE" || diffEq.AlignedCount != 2 {
		t.Fatalf("expected COMPATIBLE with 2 aligned tensors, got %+v", diffEq)
	}

	// Modify B with a shape change and an added tensor
	hdrB := map[string]any{
		"embed.weight": map[string]any{
			"dtype":        "F16",
			"shape":        []int{64, 16},
			"data_offsets": []int{0, 2048},
		},
		"layers.0.q_proj.weight": map[string]any{
			"dtype":        "BF16",
			"shape":        []int{32, 16},
			"data_offsets": []int{2048, 3072},
		},
		"layers.1.v_proj.weight": map[string]any{
			"dtype":        "BF16",
			"shape":        []int{16, 16},
			"data_offsets": []int{3072, 3584},
		},
	}
	writeSyntheticSafetensors(t, filepath.Join(dirB, "model.safetensors"), hdrB, 3584)

	diffMod, err := CompareSafetensorsHeaders(dirA, dirB)
	if err != nil {
		t.Fatalf("CompareSafetensorsHeaders mod failed: %v", err)
	}
	if diffMod.Status != "PARTIAL" {
		t.Fatalf("expected PARTIAL status, got %s", diffMod.Status)
	}
	if diffMod.ShapeMismatchCount != 1 || diffMod.DTypeMismatchCount != 1 || diffMod.OnlyInBCount != 1 {
		t.Fatalf("unexpected diff counts: %+v", diffMod)
	}
}

func TestRejectOutOfBoundsOffsets(t *testing.T) {
	tmp := t.TempDir()
	badFile := filepath.Join(tmp, "corrupt.safetensors")
	hdr := map[string]any{
		"bad.weight": map[string]any{
			"dtype":        "F32",
			"shape":        []int{16, 16},
			"data_offsets": []int{0, 4096},
		},
	}
	// Payload is only 64 bytes while header claims 4096 bytes.
	writeSyntheticSafetensors(t, badFile, hdr, 64)

	if _, err := InspectSafetensors(badFile, true); err == nil {
		t.Fatal("expected out-of-bounds data_offsets error, got nil")
	}
}
