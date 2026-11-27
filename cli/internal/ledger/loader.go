package ledger

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
)

// SuiteSummary matches the deterministic JSON schema of `neural-diff suites`.
type SuiteSummary struct {
	SuiteID    string         `json:"suite_id"`
	Version    string         `json:"version"`
	TotalCases int            `json:"total_cases"`
	Categories map[string]int `json:"categories"`
}

type rawSuiteFile struct {
	SuiteID     string `json:"suite_id"`
	Version     string `json:"version"`
	Description string `json:"description"`
	Cases       []struct {
		CaseID   string `json:"case_id"`
		Category string `json:"category"`
		Prompt   string `json:"prompt"`
	} `json:"cases"`
}

// EvolutionBenchmark represents the persisted multi-generation evolution ledger (`evolution_benchmark.json`).
type EvolutionBenchmark struct {
	ExperimentType    string   `json:"experiment_type"`
	RunID             string   `json:"run_id"`
	CacheKey          string   `json:"cache_key"`
	WallClockSeconds  float64  `json:"wall_clock_seconds"`
	Overview          Overview `json:"overview"`
	Checkpoints       []CheckpointEntry      `json:"checkpoints"`
	TensorTransitions []TensorTransition     `json:"tensor_transitions"`
	BehavioralMetrics []BehavioralCheckpoint `json:"behavioral_evolution"`
	Representation    []RepresentationEntry  `json:"representation_evolution"`
	RegressionSummary RegressionSummary      `json:"regression_summary"`
	RegressionOrigins []RegressionEvent      `json:"regression_origins"`
}

// Overview holds top-level evolution DAG and regression KPIs.
type Overview struct {
	TotalCheckpoints                  int     `json:"total_checkpoints"`
	HealthyCheckpoints                int     `json:"healthy_checkpoints"`
	TotalTransformations              int     `json:"total_transformations"`
	BranchCount                       int     `json:"branch_count"`
	AnalyzedBranchCount               int     `json:"analyzed_branch_count"`
	BranchesTruncated                 bool    `json:"branches_truncated"`
	DetectedRegressions               int     `json:"detected_regressions"`
	StatisticallySignificantCount     int     `json:"statistically_significant_regressions"`
	ExploratoryRegressions            int     `json:"exploratory_regressions"`
	MostModifiedLayers                []int   `json:"most_modified_layers"`
	LeastModifiedLayers               []int   `json:"least_modified_layers"`
	PeakCumulativeL2                  float64 `json:"peak_cumulative_l2"`
	MinBaseCosineSimilarity           float64 `json:"min_base_cosine_similarity"`
}

// CheckpointEntry describes a registered checkpoint and its 5 orthogonal SHA-256 components.
type CheckpointEntry struct {
	CheckpointID          string                `json:"checkpoint_id"`
	VersionLabel          string                `json:"version_label"`
	TransformationType    string                `json:"transformation_type"`
	ParameterCount        int64                 `json:"parameter_count"`
	FileSizeBytes         int64                 `json:"file_size_bytes"`
	SHA256Fingerprint     string                `json:"sha256_fingerprint"`
	TensorMetadataHash    string                `json:"tensor_metadata_hash"`
	ComponentFingerprints ComponentFingerprints `json:"component_fingerprints"`
}

// ComponentFingerprints stores the 5-domain orthogonal SHA-256 hashes.
type ComponentFingerprints struct {
	WeightsSHA256            string  `json:"weights_sha256"`
	ArchitectureConfigSHA256 string  `json:"architecture_config_sha256"`
	TokenizerSHA256          string  `json:"tokenizer_sha256"`
	AdapterSHA256            *string `json:"adapter_sha256"`
	MetadataSHA256           string  `json:"metadata_sha256"`
}

// TensorTransition records consecutive or cumulative weight drift between checkpoints.
type TensorTransition struct {
	Scope                  string   `json:"scope"`
	SourceLabel            string   `json:"source_label"`
	TargetLabel            string   `json:"target_label"`
	TransformationType     string   `json:"transformation_type"`
	L2Distance             float64  `json:"l2_distance"`
	RelativeL2Distance     float64  `json:"relative_l2_distance"`
	SumOfStepL2AlongPath   *float64 `json:"sum_of_step_l2_along_path"`
	PathCancellationRatio  *float64 `json:"path_cancellation_ratio"`
	GlobalCosineSimilarity float64  `json:"global_cosine_similarity"`
	ModifiedParameters     int64    `json:"modified_parameters"`
	ModifiedParameterPct   float64  `json:"modified_parameter_pct"`
	SparsityRatioTarget    float64  `json:"sparsity_ratio_target"`
	SparsityDelta          float64  `json:"sparsity_delta"`
}

// BehavioralCheckpoint records deterministic benchmark suite pass rates and rubric scores per checkpoint.
type BehavioralCheckpoint struct {
	CheckpointID     string     `json:"checkpoint_id"`
	VersionLabel     string     `json:"version_label"`
	SampleSize       int        `json:"sample_size"`
	PassRate         float64    `json:"pass_rate"`
	PassRateWilson95 [2]float64 `json:"pass_rate_wilson_ci95"`
	MeanScore        float64    `json:"mean_score"`
}

// RepresentationEntry records Centered Linear CKA and Logit JS divergence to base.
type RepresentationEntry struct {
	VersionLabel        string  `json:"version_label"`
	MeanCKAToBase       float64 `json:"mean_cka_to_base"`
	RepresentationDrift float64 `json:"representation_drift"`
	MeanLogitJSToBase   float64 `json:"mean_logit_js_to_base"`
}

// RegressionSummary separates behavioral pass-rate drops, score drops, and verified causal interventions.
type RegressionSummary struct {
	TotalEvents                       int `json:"total_events"`
	BehavioralRegressionObservedCount int `json:"behavioral_regression_observed_count"`
	ScoreRegressionObservedCount      int `json:"score_regression_observed_count"`
	CausalInterventionVerifiedCount   int `json:"causal_intervention_verified_count"`
	ExploratoryCount                  int `json:"exploratory_count"`
	StatisticallySignificantCount     int `json:"statistically_significant_count"`
}

// RegressionEvent represents a localized capability regression across an evolution edge.
type RegressionEvent struct {
	PreviousLabel                string  `json:"previous_label"`
	FirstAffectedLabel           string  `json:"first_affected_label"`
	TransformationType           string  `json:"transformation_type"`
	AffectedCategory             string  `json:"affected_category"`
	RegressionKind               string  `json:"regression_kind"`
	BehavioralRegressionObserved bool    `json:"behavioral_regression_observed"`
	ScoreRegressionObserved      bool    `json:"score_regression_observed"`
	CausalInterventionVerified   bool    `json:"causal_intervention_verified"`
	PassRateBefore               float64 `json:"pass_rate_before"`
	PassRateAfter                float64 `json:"pass_rate_after"`
	ScoreBefore                  float64 `json:"score_before"`
	ScoreAfter                   float64 `json:"score_after"`
	SampleSize                   int     `json:"sample_size"`
	IsExploratory                bool    `json:"is_exploratory"`
	StatisticallySignificant     bool    `json:"statistically_significant"`
	HolmBonferroniPValue         float64 `json:"holm_bonferroni_p_value"`
	EpistemicStatus              string  `json:"epistemic_status"`
}

// WorkspaceStatus aggregates instant workspace health without booting Python.
type WorkspaceStatus struct {
	RepoRoot            string        `json:"repo_root"`
	PythonEnginePath    string        `json:"python_engine_path"`
	SQLitePresent       bool          `json:"sqlite_present"`
	SQLiteSizeBytes     int64         `json:"sqlite_size_bytes"`
	EvidenceManifestOK  bool          `json:"evidence_manifest_present"`
	ManifestArtifacts   int           `json:"manifest_artifact_count"`
	DefaultSuite        *SuiteSummary `json:"default_suite,omitempty"`
	ActiveEvolutionRun  string        `json:"active_evolution_run,omitempty"`
	EvolutionCheckpoints int          `json:"evolution_checkpoints"`
	DemoFixturesReady   bool          `json:"demo_fixtures_ready"`
}

// FindRepoRoot walks upward from cwd or the executable location to locate `pyproject.toml`.
func FindRepoRoot() (string, error) {
	candidates := []string{}
	if wd, err := os.Getwd(); err == nil {
		candidates = append(candidates, wd)
	}
	if exe, err := os.Executable(); err == nil {
		candidates = append(candidates, filepath.Dir(exe))
	}
	if envRoot := os.Getenv("NEURAL_DIFF_ROOT"); envRoot != "" {
		candidates = append([]string{envRoot}, candidates...)
	}

	for _, start := range candidates {
		dir, err := filepath.Abs(start)
		if err != nil {
			continue
		}
		for {
			if fileExists(filepath.Join(dir, "pyproject.toml")) && fileExists(filepath.Join(dir, "core", "cli.py")) {
				return dir, nil
			}
			parent := filepath.Dir(dir)
			if parent == dir {
				break
			}
			dir = parent
		}
	}
	return "", errors.New("could not locate NEURAL DIFF workspace root")
}

// LoadDefaultSuite reads `benchmarks/datasets/default_suite_v1.json` natively in < 1ms.
func LoadDefaultSuite(repoRoot string) (*SuiteSummary, error) {
	p := filepath.Join(repoRoot, "benchmarks", "datasets", "default_suite_v1.json")
	raw, err := os.ReadFile(p)
	if err != nil {
		return nil, err
	}
	var parsed rawSuiteFile
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return nil, fmt.Errorf("invalid benchmark suite JSON: %w", err)
	}
	counts := make(map[string]int)
	for _, c := range parsed.Cases {
		counts[c.Category]++
	}
	return &SuiteSummary{
		SuiteID:    parsed.SuiteID,
		Version:    parsed.Version,
		TotalCases: len(parsed.Cases),
		Categories: counts,
	}, nil
}

// LoadEvolutionBenchmark loads an evolution ledger JSON file (defaults to `evidence/evolution/evolution_benchmark.json`).
func LoadEvolutionBenchmark(repoRoot string, explicitFile string) (*EvolutionBenchmark, string, error) {
	target := explicitFile
	if target == "" {
		target = filepath.Join(repoRoot, "evidence", "evolution", "evolution_benchmark.json")
	}
	raw, err := os.ReadFile(target)
	if err != nil {
		return nil, target, err
	}
	var bench EvolutionBenchmark
	if err := json.Unmarshal(raw, &bench); err != nil {
		return nil, target, fmt.Errorf("invalid evolution JSON %s: %w", target, err)
	}
	return &bench, target, nil
}

// DiscoverCheckpoints returns local `.safetensors` directories useful for interactive selection.
func DiscoverCheckpoints(repoRoot string) []string {
	var found []string
	roots := []string{
		filepath.Join(repoRoot, "data", "demo_fixtures"),
		filepath.Join(repoRoot, "data", "evolution_smollm2"),
		filepath.Join(repoRoot, "models_cache"),
	}
	for _, r := range roots {
		entries, err := os.ReadDir(r)
		if err != nil {
			continue
		}
		for _, e := range entries {
			if !e.IsDir() {
				continue
			}
			sub := filepath.Join(r, e.Name())
			if fileExists(filepath.Join(sub, "model.safetensors")) || fileExists(filepath.Join(sub, "model.safetensors.index.json")) {
				rel, err := filepath.Rel(repoRoot, sub)
				if err == nil {
					found = append(found, filepath.ToSlash(rel))
				} else {
					found = append(found, sub)
				}
			}
		}
	}
	sort.Strings(found)
	return found
}

// InspectWorkspace gathers instant status telemetry without importing Python/PyTorch.
func InspectWorkspace(repoRoot string, pythonExe string) WorkspaceStatus {
	st := WorkspaceStatus{
		RepoRoot:         repoRoot,
		PythonEnginePath: pythonExe,
	}
	dbPath := filepath.Join(repoRoot, "data", "neural_diff.db")
	if info, err := os.Stat(dbPath); err == nil && !info.IsDir() {
		st.SQLitePresent = true
		st.SQLiteSizeBytes = info.Size()
	}
	manifestPath := filepath.Join(repoRoot, "evidence", "EVIDENCE_MANIFEST.json")
	if raw, err := os.ReadFile(manifestPath); err == nil {
		var m struct {
			Artifacts []json.RawMessage `json:"artifacts"`
		}
		if json.Unmarshal(raw, &m) == nil {
			st.EvidenceManifestOK = true
			st.ManifestArtifacts = len(m.Artifacts)
		}
	}
	if suite, err := LoadDefaultSuite(repoRoot); err == nil {
		st.DefaultSuite = suite
	}
	if evo, _, err := LoadEvolutionBenchmark(repoRoot, ""); err == nil {
		st.ActiveEvolutionRun = evo.RunID
		st.EvolutionCheckpoints = len(evo.Checkpoints)
	}
	if fileExists(filepath.Join(repoRoot, "data", "demo_fixtures", "synth_model_a", "model.safetensors")) {
		st.DemoFixturesReady = true
	}
	return st
}

func fileExists(p string) bool {
	info, err := os.Stat(p)
	return err == nil && !info.IsDir()
}
