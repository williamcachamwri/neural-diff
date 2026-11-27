package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"

	"charm.land/fang/v2"
	"github.com/spf13/cobra"

	"github.com/williamcachamwri/neural-diff/cli/internal/engine"
	"github.com/williamcachamwri/neural-diff/cli/internal/ledger"
	"github.com/williamcachamwri/neural-diff/cli/internal/render"
	"github.com/williamcachamwri/neural-diff/cli/internal/safetensors"
	"github.com/williamcachamwri/neural-diff/cli/internal/theme"
	"github.com/williamcachamwri/neural-diff/cli/internal/tui"
	"github.com/williamcachamwri/neural-diff/cli/internal/wizard"
)

var (
	version = "0.1.0"
	commit  = "HEAD"
)

type globalFlags struct {
	jsonOut   bool
	prettyOut bool
	noColor   bool
}

func (g globalFlags) emitJSON() bool {
	if g.jsonOut || os.Getenv("NEURAL_DIFF_CLI_JSON") == "1" {
		return true
	}
	if g.prettyOut {
		return false
	}
	return !theme.IsStdoutTTY()
}

func printJSON(v any) error {
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(v)
}

func buildRootCmd() *cobra.Command {
	var gf globalFlags

	root := &cobra.Command{
		Use:   "ndiff",
		Short: "NEURAL DIFF — Structural, Neural & Behavioral Checkpoint Diffing",
		Long: "Minimalist, fast terminal interface for NEURAL DIFF.\n" +
			"Inspects .safetensors headers, Evolution DAGs, and benchmark suites in milliseconds,\n" +
			"and bridges seamlessly to the float64 PyTorch engine for full neural & behavioral diffs.",
		SilenceUsage:  true,
		SilenceErrors: true,
		PersistentPreRun: func(_ *cobra.Command, _ []string) {
			if gf.noColor {
				_ = os.Setenv("NO_COLOR", "1")
			}
		},
	}

	root.PersistentFlags().BoolVar(&gf.jsonOut, "json", false, "Emit deterministic machine-readable JSON")
	root.PersistentFlags().BoolVar(&gf.prettyOut, "pretty", false, "Force Lip Gloss v2 formatted output even when piped")
	root.PersistentFlags().BoolVar(&gf.noColor, "no-color", false, "Disable ANSI color output")

	// 1. status (instant Go telemetry)
	cmdStatus := &cobra.Command{
		Use:   "status",
		Short: "Show instant workspace, SQLite, manifest, and benchmark readiness",
		RunE: func(_ *cobra.Command, _ []string) error {
			repoRoot, _ := ledger.FindRepoRoot()
			py := engine.ResolvePythonExecutable(repoRoot)
			st := ledger.InspectWorkspace(repoRoot, py)
			if gf.emitJSON() {
				return printJSON(st)
			}
			fmt.Print(render.RenderStatus(st))
			return nil
		},
	}

	// 2. suites (instant Go reader, identical JSON schema to `neural-diff suites`)
	cmdSuites := &cobra.Command{
		Use:   "suites",
		Short: "List registered deterministic behavioral benchmark suites",
		RunE: func(cmd *cobra.Command, _ []string) error {
			repoRoot, err := ledger.FindRepoRoot()
			if err == nil {
				if suite, sErr := ledger.LoadDefaultSuite(repoRoot); sErr == nil {
					if gf.emitJSON() {
						return printJSON(suite)
					}
					fmt.Print(render.RenderSuites(suite))
					return nil
				}
			}
			return delegateToPython(cmd.Context(), repoRoot, []string{"suites"}, gf, "suites", "Loading benchmark suite")
		},
	}

	// 3. inspect (instant Go `.safetensors` binary header reader; Python fallback for Hub / PyTorch pickle)
	var inspectLocalOnly bool
	var inspectFilter string
	var inspectTop int
	cmdInspect := &cobra.Command{
		Use:   "inspect <checkpoint>",
		Short: "Inspect checkpoint metadata and tensor headers without loading weights",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			target := args[0]
			if sum, err := safetensors.InspectSafetensors(target, !gf.emitJSON()); err == nil {
				if gf.emitJSON() {
					return printJSON(sum)
				}
				fmt.Print(render.RenderCheckpointSummary(sum, inspectFilter, inspectTop))
				return nil
			}
			repoRoot, _ := ledger.FindRepoRoot()
			pyArgs := []string{"inspect", target}
			if inspectLocalOnly {
				pyArgs = append(pyArgs, "--local-only")
			}
			return delegateToPython(cmd.Context(), repoRoot, pyArgs, gf, "inspect", "Inspecting checkpoint metadata")
		},
	}
	cmdInspect.Flags().BoolVar(&inspectLocalOnly, "local-only", false, "Do not download from Hugging Face Hub")
	cmdInspect.Flags().StringVar(&inspectFilter, "filter", "", "Filter displayed tensors by substring")
	cmdInspect.Flags().IntVar(&inspectTop, "top", 24, "Maximum tensor rows to display in table mode")

	// 4. quickdiff (instant Go `.safetensors` structural header diff in < 5ms)
	var qModelA, qModelB string
	cmdQuickDiff := &cobra.Command{
		Use:   "quickdiff",
		Short: "Instant zero-PyTorch structural diff across .safetensors headers",
		RunE: func(_ *cobra.Command, _ []string) error {
			diff, err := safetensors.CompareSafetensorsHeaders(qModelA, qModelB)
			if err != nil {
				return err
			}
			if gf.emitJSON() {
				return printJSON(diff)
			}
			fmt.Print(render.RenderQuickDiff(diff))
			return nil
		},
	}
	cmdQuickDiff.Flags().StringVar(&qModelA, "model-a", "", "Baseline .safetensors checkpoint directory or file")
	cmdQuickDiff.Flags().StringVar(&qModelB, "model-b", "", "Candidate .safetensors checkpoint directory or file")
	_ = cmdQuickDiff.MarkFlagRequired("model-a")
	_ = cmdQuickDiff.MarkFlagRequired("model-b")

	// 5. dag (instant Evolution DAG lineage tree & regression table)
	var dagFile string
	cmdDAG := &cobra.Command{
		Use:   "dag",
		Short: "Render the branched & merged Neural Evolution DAG and epistemic ledger",
		RunE: func(_ *cobra.Command, _ []string) error {
			repoRoot, _ := ledger.FindRepoRoot()
			evo, _, err := ledger.LoadEvolutionBenchmark(repoRoot, dagFile)
			if err != nil {
				return err
			}
			if gf.emitJSON() {
				return printJSON(evo)
			}
			fmt.Print(render.RenderEvolutionDAG(evo))
			return nil
		},
	}
	cmdDAG.Flags().StringVar(&dagFile, "file", "", "Path to evolution benchmark/export JSON")

	// 6. tui (Bubble Tea v2 interactive 5-tab Terminal Studio)
	var tuiEvoFile, tuiModelPath string
	cmdTUI := &cobra.Command{
		Use:   "tui",
		Short: "Launch interactive Bubble Tea v2 Terminal Studio (DAG, Checkpoints, Tensors)",
		RunE: func(_ *cobra.Command, _ []string) error {
			if !theme.IsStdoutTTY() {
				return errors.New("`ndiff tui` requires an interactive TTY terminal")
			}
			repoRoot, _ := ledger.FindRepoRoot()
			py := engine.ResolvePythonExecutable(repoRoot)
			return tui.RunStudio(tui.StudioConfig{
				RepoRoot:      repoRoot,
				PythonExe:     py,
				EvolutionFile: tuiEvoFile,
				ModelPath:     tuiModelPath,
			})
		},
	}
	cmdTUI.Flags().StringVar(&tuiEvoFile, "file", "", "Optional evolution benchmark JSON path")
	cmdTUI.Flags().StringVar(&tuiModelPath, "model", "", "Optional .safetensors checkpoint path for Tensor Explorer")

	// 7. wizard (Huh v2 interactive comparison form)
	cmdWizard := &cobra.Command{
		Use:   "wizard",
		Short: "Interactive comparison configurator powered by Charm Huh v2",
		RunE: func(cmd *cobra.Command, _ []string) error {
			if !theme.IsStdoutTTY() {
				return errors.New("`ndiff wizard` requires an interactive TTY terminal")
			}
			repoRoot, _ := ledger.FindRepoRoot()
			plan, err := wizard.RunCompareWizard(repoRoot)
			if err != nil {
				return err
			}
			fmt.Println()
			fmt.Println(theme.SectionTitle.Render("Reproducible Command:"))
			fmt.Println("  " + plan.CommandString())
			fmt.Println()
			if !plan.ExecuteNow {
				return nil
			}
			return delegateToPython(cmd.Context(), repoRoot, plan.CLIArgs(), gf, "compare", "Running neural & behavioral comparison")
		},
	}

	// 8. validate (Python engine bridge with spinner + clean output)
	var valA, valB string
	var valLocal bool
	cmdValidate := &cobra.Command{
		Use:   "validate",
		Short: "Validate structural & tokenizer compatibility between Model A and Model B",
		RunE: func(cmd *cobra.Command, _ []string) error {
			repoRoot, _ := ledger.FindRepoRoot()
			pyArgs := []string{"validate", "--model-a", valA, "--model-b", valB}
			if valLocal {
				pyArgs = append(pyArgs, "--local-only")
			}
			return delegateToPython(cmd.Context(), repoRoot, pyArgs, gf, "validate", "Validating checkpoint compatibility")
		},
	}
	cmdValidate.Flags().StringVar(&valA, "model-a", "", "Baseline checkpoint")
	cmdValidate.Flags().StringVar(&valB, "model-b", "", "Candidate checkpoint")
	cmdValidate.Flags().BoolVar(&valLocal, "local-only", false, "Offline mode")
	_ = cmdValidate.MarkFlagRequired("model-a")
	_ = cmdValidate.MarkFlagRequired("model-b")

	// 9. compare
	var cmpA, cmpB, cmpModules, cmpDevice, cmpOutDir string
	var cmpAtol, cmpRtol float64
	var cmpSeed, cmpMaxTokens int
	var cmpReal, cmpLocal bool
	cmdCompare := &cobra.Command{
		Use:   "compare",
		Short: "Run multi-layer checkpoint comparison and export reports",
		RunE: func(cmd *cobra.Command, _ []string) error {
			repoRoot, _ := ledger.FindRepoRoot()
			pyArgs := []string{
				"compare",
				"--model-a", cmpA,
				"--model-b", cmpB,
				"--modules", cmpModules,
				"--device", cmpDevice,
				"--atol", fmt.Sprintf("%g", cmpAtol),
				"--rtol", fmt.Sprintf("%g", cmpRtol),
				"--seed", fmt.Sprintf("%d", cmpSeed),
				"--max-new-tokens", fmt.Sprintf("%d", cmpMaxTokens),
			}
			if cmpReal {
				pyArgs = append(pyArgs, "--real-experiment")
			}
			if cmpLocal {
				pyArgs = append(pyArgs, "--local-only")
			}
			if cmpOutDir != "" {
				pyArgs = append(pyArgs, "--output-dir", cmpOutDir)
			}
			return delegateToPython(cmd.Context(), repoRoot, pyArgs, gf, "compare", "Executing float64 neural diff pipeline")
		},
	}
	cmdCompare.Flags().StringVar(&cmpA, "model-a", "", "Baseline checkpoint")
	cmdCompare.Flags().StringVar(&cmpB, "model-b", "", "Candidate checkpoint")
	cmdCompare.Flags().StringVar(&cmpModules, "modules", "structural,tensor,neuron,regression", "Comma-separated analysis modules")
	cmdCompare.Flags().StringVar(&cmpDevice, "device", "auto", "Compute device (auto|cpu|cuda|mps)")
	cmdCompare.Flags().Float64Var(&cmpAtol, "atol", 1e-6, "Absolute tolerance")
	cmdCompare.Flags().Float64Var(&cmpRtol, "rtol", 1e-5, "Relative tolerance")
	cmdCompare.Flags().IntVar(&cmpSeed, "seed", 42, "Deterministic random seed")
	cmdCompare.Flags().IntVar(&cmpMaxTokens, "max-new-tokens", 128, "Max generation tokens")
	cmdCompare.Flags().BoolVar(&cmpReal, "real-experiment", false, "Tag run as REAL MODEL EXPERIMENT")
	cmdCompare.Flags().BoolVar(&cmpLocal, "local-only", false, "Offline mode")
	cmdCompare.Flags().StringVar(&cmpOutDir, "output-dir", "", "Directory to export JSON/MD/HTML reports")
	_ = cmdCompare.MarkFlagRequired("model-a")
	_ = cmdCompare.MarkFlagRequired("model-b")

	// 10. behavior
	var behA, behB, behCat, behDev, behOut string
	var behMaxCases, behSeed int
	cmdBehavior := &cobra.Command{
		Use:   "behavior",
		Short: "Run deterministic paired behavioral diff suite",
		RunE: func(cmd *cobra.Command, _ []string) error {
			repoRoot, _ := ledger.FindRepoRoot()
			pyArgs := []string{
				"behavior",
				"--model-a", behA,
				"--model-b", behB,
				"--seed", fmt.Sprintf("%d", behSeed),
				"--device", behDev,
			}
			if behCat != "" {
				pyArgs = append(pyArgs, "--category", behCat)
			}
			if behMaxCases > 0 {
				pyArgs = append(pyArgs, "--max-cases", fmt.Sprintf("%d", behMaxCases))
			}
			if behOut != "" {
				pyArgs = append(pyArgs, "--output-dir", behOut)
			}
			return delegateToPython(cmd.Context(), repoRoot, pyArgs, gf, "behavior", "Evaluating paired behavioral benchmark")
		},
	}
	cmdBehavior.Flags().StringVar(&behA, "model-a", "", "Baseline checkpoint")
	cmdBehavior.Flags().StringVar(&behB, "model-b", "", "Candidate checkpoint")
	cmdBehavior.Flags().StringVar(&behCat, "category", "", "Optional single category filter")
	cmdBehavior.Flags().IntVar(&behMaxCases, "max-cases", 0, "Optional case cap")
	cmdBehavior.Flags().IntVar(&behSeed, "seed", 42, "Deterministic seed")
	cmdBehavior.Flags().StringVar(&behDev, "device", "auto", "Compute device")
	cmdBehavior.Flags().StringVar(&behOut, "output-dir", "", "Report output directory")
	_ = cmdBehavior.MarkFlagRequired("model-a")
	_ = cmdBehavior.MarkFlagRequired("model-b")

	// 11. svd
	var svdA, svdB, svdTensor string
	var svdLocal bool
	cmdSVD := &cobra.Command{
		Use:   "svd",
		Short: "Analyze SVD singular spectrum and Shannon effective rank of a 2D weight delta",
		RunE: func(cmd *cobra.Command, _ []string) error {
			repoRoot, _ := ledger.FindRepoRoot()
			pyArgs := []string{"svd", "--model-a", svdA, "--model-b", svdB, "--tensor", svdTensor}
			if svdLocal {
				pyArgs = append(pyArgs, "--local-only")
			}
			return delegateToPython(cmd.Context(), repoRoot, pyArgs, gf, "svd", "Computing float64 SVD singular spectrum")
		},
	}
	cmdSVD.Flags().StringVar(&svdA, "model-a", "", "Baseline checkpoint")
	cmdSVD.Flags().StringVar(&svdB, "model-b", "", "Candidate checkpoint")
	cmdSVD.Flags().StringVar(&svdTensor, "tensor", "", "Exact 2D tensor parameter name")
	cmdSVD.Flags().BoolVar(&svdLocal, "local-only", false, "Offline mode")
	_ = cmdSVD.MarkFlagRequired("model-a")
	_ = cmdSVD.MarkFlagRequired("model-b")
	_ = cmdSVD.MarkFlagRequired("tensor")

	// 12. evolution subcommands
	cmdEvo := &cobra.Command{
		Use:   "evolution",
		Short: "Manage Neural Evolution Tracking projects, checkpoints, and multi-generation runs",
	}

	var epName, epDesc, epDB string
	cmdEvoCreate := &cobra.Command{
		Use:   "create-project",
		Short: "Create a new evolution tracking project",
		RunE: func(cmd *cobra.Command, _ []string) error {
			repoRoot, _ := ledger.FindRepoRoot()
			pyArgs := []string{"evolution", "create-project", "--name", epName, "--description", epDesc}
			if epDB != "" {
				pyArgs = append(pyArgs, "--db-path", epDB)
			}
			return delegateToPython(cmd.Context(), repoRoot, pyArgs, gf, "evolution create-project", "Creating evolution project")
		},
	}
	cmdEvoCreate.Flags().StringVar(&epName, "name", "", "Project name")
	cmdEvoCreate.Flags().StringVar(&epDesc, "description", "", "Project description")
	cmdEvoCreate.Flags().StringVar(&epDB, "db-path", "", "Optional SQLite path")
	_ = cmdEvoCreate.MarkFlagRequired("name")

	var erProj, erSrc, erLabel, erParents, erTrans, erNotes, erDB string
	var erLocal bool
	cmdEvoReg := &cobra.Command{
		Use:   "register",
		Short: "Register a checkpoint with 5-component SHA-256 provenance",
		RunE: func(cmd *cobra.Command, _ []string) error {
			repoRoot, _ := ledger.FindRepoRoot()
			pyArgs := []string{"evolution", "register", "--project-id", erProj, "--source", erSrc}
			if erLabel != "" {
				pyArgs = append(pyArgs, "--label", erLabel)
			}
			if erParents != "" {
				pyArgs = append(pyArgs, "--parents", erParents)
			}
			if erTrans != "" {
				pyArgs = append(pyArgs, "--transformation", erTrans)
			}
			if erNotes != "" {
				pyArgs = append(pyArgs, "--notes", erNotes)
			}
			if erLocal {
				pyArgs = append(pyArgs, "--local-only")
			}
			if erDB != "" {
				pyArgs = append(pyArgs, "--db-path", erDB)
			}
			return delegateToPython(cmd.Context(), repoRoot, pyArgs, gf, "evolution register", "Computing 5-domain SHA-256 checkpoint fingerprint")
		},
	}
	cmdEvoReg.Flags().StringVar(&erProj, "project-id", "", "Evolution project ID")
	cmdEvoReg.Flags().StringVar(&erSrc, "source", "", "Checkpoint path or Hub repo ID")
	cmdEvoReg.Flags().StringVar(&erLabel, "label", "", "Version label (e.g. v1a)")
	cmdEvoReg.Flags().StringVar(&erParents, "parents", "", "Comma-separated parent checkpoint IDs")
	cmdEvoReg.Flags().StringVar(&erTrans, "transformation", "", "Transformation stage")
	cmdEvoReg.Flags().StringVar(&erNotes, "notes", "", "Experiment notes")
	cmdEvoReg.Flags().BoolVar(&erLocal, "local-only", false, "Offline mode")
	cmdEvoReg.Flags().StringVar(&erDB, "db-path", "", "Optional SQLite path")
	_ = cmdEvoReg.MarkFlagRequired("project-id")
	_ = cmdEvoReg.MarkFlagRequired("source")

	var eaProj, eaOut, eaDB string
	var eaSkipBeh, eaSkipRep, eaForce, eaLocal bool
	cmdEvoAnalyze := &cobra.Command{
		Use:   "analyze",
		Short: "Run multi-generation evolution analysis and export 5-format reports",
		RunE: func(cmd *cobra.Command, _ []string) error {
			repoRoot, _ := ledger.FindRepoRoot()
			pyArgs := []string{"evolution", "analyze", "--project-id", eaProj}
			if eaSkipBeh {
				pyArgs = append(pyArgs, "--skip-behavioral")
			}
			if eaSkipRep {
				pyArgs = append(pyArgs, "--skip-representation")
			}
			if eaForce {
				pyArgs = append(pyArgs, "--force-rerun")
			}
			if eaLocal {
				pyArgs = append(pyArgs, "--local-only")
			}
			if eaOut != "" {
				pyArgs = append(pyArgs, "--output-dir", eaOut)
			}
			if eaDB != "" {
				pyArgs = append(pyArgs, "--db-path", eaDB)
			}
			return delegateToPython(cmd.Context(), repoRoot, pyArgs, gf, "evolution analyze", "Analyzing multi-generation evolution DAG")
		},
	}
	cmdEvoAnalyze.Flags().StringVar(&eaProj, "project-id", "", "Evolution project ID")
	cmdEvoAnalyze.Flags().BoolVar(&eaSkipBeh, "skip-behavioral", false, "Skip behavioral evaluation")
	cmdEvoAnalyze.Flags().BoolVar(&eaSkipRep, "skip-representation", false, "Skip Linear CKA representation hooks")
	cmdEvoAnalyze.Flags().BoolVar(&eaForce, "force-rerun", false, "Bypass content-addressed run cache")
	cmdEvoAnalyze.Flags().BoolVar(&eaLocal, "local-only", false, "Offline mode")
	cmdEvoAnalyze.Flags().StringVar(&eaOut, "output-dir", "", "Export JSON/CSV/MD/HTML/PDF reports to directory")
	cmdEvoAnalyze.Flags().StringVar(&eaDB, "db-path", "", "Optional SQLite path")
	_ = cmdEvoAnalyze.MarkFlagRequired("project-id")

	cmdEvo.AddCommand(cmdEvoCreate, cmdEvoReg, cmdEvoAnalyze)

	root.AddCommand(
		cmdStatus,
		cmdSuites,
		cmdInspect,
		cmdQuickDiff,
		cmdDAG,
		cmdTUI,
		cmdWizard,
		cmdValidate,
		cmdCompare,
		cmdBehavior,
		cmdSVD,
		cmdEvo,
	)

	return root
}

func delegateToPython(ctx context.Context, repoRoot string, pyArgs []string, gf globalFlags, title string, spinLabel string) error {
	useSpinner := !gf.emitJSON()
	out, dur, err := engine.RunPythonCLI(ctx, repoRoot, pyArgs, useSpinner, spinLabel)
	if err != nil {
		return err
	}
	trimmed := strings.TrimSpace(string(out))
	if gf.emitJSON() {
		fmt.Println(trimmed)
		return nil
	}
	fmt.Print(render.RenderPythonJSONResult(fmt.Sprintf("%s (%s)", title, dur.Round(1e6)), []byte(trimmed)))
	return nil
}

func main() {
	root := buildRootCmd()
	err := fang.Execute(
		context.Background(),
		root,
		fang.WithVersion(version),
		fang.WithCommit(commit),
		fang.WithColorSchemeFunc(theme.FangColorScheme),
	)
	if err != nil {
		os.Exit(1)
	}
}
