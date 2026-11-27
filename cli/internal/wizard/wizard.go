package wizard

import (
	"fmt"
	"strings"

	"charm.land/huh/v2"

	"github.com/williamcachamwri/neural-diff/cli/internal/ledger"
)

// ComparePlan captures user choices from the interactive `charm.land/huh/v2` comparison wizard.
type ComparePlan struct {
	ModelA     string
	ModelB     string
	Modules    []string
	Device     string
	OutputDir  string
	LocalOnly  bool
	ExecuteNow bool
}

// CLIArgs converts a `ComparePlan` into deterministic `core.cli compare` arguments.
func (p ComparePlan) CLIArgs() []string {
	mods := strings.Join(p.Modules, ",")
	if mods == "" {
		mods = "structural,tensor,neuron,regression"
	}
	args := []string{
		"compare",
		"--model-a", p.ModelA,
		"--model-b", p.ModelB,
		"--modules", mods,
		"--device", p.Device,
	}
	if p.LocalOnly {
		args = append(args, "--local-only")
	}
	if strings.TrimSpace(p.OutputDir) != "" {
		args = append(args, "--output-dir", strings.TrimSpace(p.OutputDir))
	}
	return args
}

// CommandString formats the exact reproducible shell command.
func (p ComparePlan) CommandString() string {
	args := p.CLIArgs()
	quoted := make([]string, 0, len(args)+1)
	quoted = append(quoted, "ndiff")
	for _, a := range args {
		if strings.ContainsAny(a, " \t") {
			quoted = append(quoted, fmt.Sprintf("%q", a))
		} else {
			quoted = append(quoted, a)
		}
	}
	return strings.Join(quoted, " ")
}

// RunCompareWizard runs a minimalist `huh/v2` interactive form for configuring a checkpoint comparison.
func RunCompareWizard(repoRoot string) (*ComparePlan, error) {
	discovered := ledger.DiscoverCheckpoints(repoRoot)
	defaultA := "data/demo_fixtures/synth_model_a"
	defaultB := "data/demo_fixtures/synth_model_b"
	if len(discovered) >= 2 {
		defaultA = discovered[0]
		defaultB = discovered[1]
	}

	plan := &ComparePlan{
		ModelA:     defaultA,
		ModelB:     defaultB,
		Modules:    []string{"structural", "tensor", "neuron", "regression"},
		Device:     "auto",
		OutputDir:  "",
		LocalOnly:  true,
		ExecuteNow: true,
	}

	form := huh.NewForm(
		huh.NewGroup(
			huh.NewInput().
				Title("Baseline Checkpoint (Model A)").
				Description("Local .safetensors directory or Hugging Face repo ID").
				Value(&plan.ModelA),
			huh.NewInput().
				Title("Candidate Checkpoint (Model B)").
				Description("Target checkpoint to compare against Model A").
				Value(&plan.ModelB),
			huh.NewMultiSelect[string]().
				Title("Analysis Modules").
				Options(
					huh.NewOption("Structural Architecture Diff", "structural").Selected(true),
					huh.NewOption("Float64 Tensor Weight Metrics", "tensor").Selected(true),
					huh.NewOption("Layer & Attention Head Slicing", "neuron").Selected(true),
					huh.NewOption("Observational Activation Hooks", "activation"),
					huh.NewOption("Deterministic Behavioral Suite", "behavioral"),
					huh.NewOption("Regression Gate Verdict", "regression").Selected(true),
				).
				Value(&plan.Modules),
		),
		huh.NewGroup(
			huh.NewSelect[string]().
				Title("Compute Device").
				Options(
					huh.NewOption("Auto (CUDA -> MPS -> CPU)", "auto"),
					huh.NewOption("CPU (Deterministic Portable)", "cpu"),
					huh.NewOption("CUDA GPU", "cuda"),
				).
				Value(&plan.Device),
			huh.NewInput().
				Title("Report Export Directory (Optional)").
				Placeholder("leave blank for terminal output only").
				Value(&plan.OutputDir),
			huh.NewConfirm().
				Title("Offline / Local Checkpoints Only?").
				Affirmative("Local only").
				Negative("Allow Hub").
				Value(&plan.LocalOnly),
			huh.NewConfirm().
				Title("Execute Comparison Now?").
				Affirmative("Run now").
				Negative("Print CLI command only").
				Value(&plan.ExecuteNow),
		),
	).WithTheme(huh.ThemeFunc(huh.ThemeBase16))

	if err := form.Run(); err != nil {
		return nil, err
	}
	return plan, nil
}
