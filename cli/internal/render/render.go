package render

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"charm.land/lipgloss/v2"
	"charm.land/lipgloss/v2/table"
	"charm.land/lipgloss/v2/tree"

	"github.com/williamcachamwri/neural-diff/cli/internal/ledger"
	"github.com/williamcachamwri/neural-diff/cli/internal/safetensors"
	"github.com/williamcachamwri/neural-diff/cli/internal/theme"
)

func newCleanTable(headers ...string) *table.Table {
	return table.New().
		Border(lipgloss.RoundedBorder()).
		BorderStyle(lipgloss.NewStyle().Foreground(theme.Zinc700)).
		Headers(headers...).
		StyleFunc(func(row, _ int) lipgloss.Style {
			if row == table.HeaderRow {
				return theme.TableHeader
			}
			if row%2 == 0 {
				return theme.TableCell
			}
			return theme.TableCellMuted
		})
}

// RenderStatus formats workspace health and ledger availability cleanly.
func RenderStatus(st ledger.WorkspaceStatus) string {
	var b strings.Builder
	b.WriteString(theme.HeaderBanner("Workspace Telemetry") + "\n\n")

	sqliteVal := "absent"
	if st.SQLitePresent {
		sqliteVal = fmt.Sprintf("ready (%s)", theme.HumanBytes(st.SQLiteSizeBytes))
	}
	manifestVal := "absent"
	if st.EvidenceManifestOK {
		manifestVal = fmt.Sprintf("sealed (%d SHA-256 artifacts)", st.ManifestArtifacts)
	}
	suiteVal := "unavailable"
	if st.DefaultSuite != nil {
		suiteVal = fmt.Sprintf("%s v%s (%d cases / %d categories)",
			st.DefaultSuite.SuiteID, st.DefaultSuite.Version, st.DefaultSuite.TotalCases, len(st.DefaultSuite.Categories))
	}
	evoVal := "none"
	if st.ActiveEvolutionRun != "" {
		evoVal = fmt.Sprintf("%s (%d checkpoints)", st.ActiveEvolutionRun, st.EvolutionCheckpoints)
	}
	fixturesVal := "not seeded"
	if st.DemoFixturesReady {
		fixturesVal = "ready (data/demo_fixtures)"
	}

	kv := newCleanTable("COMPONENT", "STATE", "DETAIL")
	kv.Row("Workspace Root", theme.StatusBadge("OK"), st.RepoRoot)
	kv.Row("Python Compute Engine", statusTag(st.PythonEnginePath != ""), nonEmpty(st.PythonEnginePath, "not detected"))
	kv.Row("SQLite WAL Store", statusTag(st.SQLitePresent), sqliteVal)
	kv.Row("Evidence Manifest", statusTag(st.EvidenceManifestOK), manifestVal)
	kv.Row("Behavioral Suite", statusTag(st.DefaultSuite != nil), suiteVal)
	kv.Row("Evolution Ledger", statusTag(st.ActiveEvolutionRun != ""), evoVal)
	kv.Row("Synthetic Fixtures", statusTag(st.DemoFixturesReady), fixturesVal)

	b.WriteString(kv.String() + "\n")
	b.WriteString(theme.Muted.Render("Hint: run `ndiff tui` for interactive DAG & tensor studio, or `ndiff wizard` to configure a run.") + "\n")
	return b.String()
}

// RenderSuites formats the deterministic behavioral benchmark suite summary.
func RenderSuites(suite *ledger.SuiteSummary) string {
	var b strings.Builder
	b.WriteString(theme.HeaderBanner("Deterministic Behavioral Suite") + "\n")
	b.WriteString(theme.Muted.Render(fmt.Sprintf("Suite: %s  •  Version: %s  •  Total Cases: %d",
		suite.SuiteID, suite.Version, suite.TotalCases)) + "\n\n")

	keys := make([]string, 0, len(suite.Categories))
	maxCount := 1
	for k, c := range suite.Categories {
		keys = append(keys, k)
		if c > maxCount {
			maxCount = c
		}
	}
	sort.Strings(keys)

	t := newCleanTable("CATEGORY", "CASES", "WEIGHT", "DISTRIBUTION")
	for _, k := range keys {
		c := suite.Categories[k]
		pct := 0.0
		if suite.TotalCases > 0 {
			pct = float64(c) / float64(suite.TotalCases)
		}
		t.Row(
			k,
			fmt.Sprintf("%d", c),
			fmt.Sprintf("%.1f%%", pct*100.0),
			theme.Bar(float64(c)/float64(maxCount), 14, theme.Cyan400),
		)
	}
	b.WriteString(t.String() + "\n")
	return b.String()
}

// RenderCheckpointSummary formats native `.safetensors` header inspection output.
func RenderCheckpointSummary(sum *safetensors.CheckpointSummary, filter string, top int) string {
	var b strings.Builder
	b.WriteString(theme.HeaderBanner("Checkpoint Header Inspection") + "\n\n")

	arch := nonEmpty(sum.Architecture, nonEmpty(sum.ModelType, "—"))
	dims := "—"
	if sum.NumHiddenLayers != nil && sum.HiddenSize != nil {
		dims = fmt.Sprintf("%d layers × %d hidden", *sum.NumHiddenLayers, *sum.HiddenSize)
		if sum.VocabSize != nil {
			dims += fmt.Sprintf("  (vocab %d)", *sum.VocabSize)
		}
	}

	dtypeParts := make([]string, 0, len(sum.DTypes))
	for dt, cnt := range sum.DTypes {
		dtypeParts = append(dtypeParts, fmt.Sprintf("%s:%d", dt, cnt))
	}
	sort.Strings(dtypeParts)

	metaLines := []string{
		fmt.Sprintf("%s  %s", theme.Label.Render("Model:       "), theme.Value.Render(sum.ModelID)),
		fmt.Sprintf("%s  %s", theme.Label.Render("Path:        "), theme.Muted.Render(sum.ResolvedPath)),
		fmt.Sprintf("%s  %s  •  %s  •  %s",
			theme.Label.Render("Footprint:   "),
			theme.Value.Render(theme.HumanParams(sum.ParameterCount)+" params"),
			theme.Value.Render(fmt.Sprintf("%d tensors", sum.TensorCount)),
			theme.Value.Render(theme.HumanBytes(sum.FileSizeBytes)),
		),
		fmt.Sprintf("%s  %s  (%s)", theme.Label.Render("Topology:    "), theme.Value.Render(arch), dims),
		fmt.Sprintf("%s  %s  •  DTypes: %s",
			theme.Label.Render("Format:      "),
			theme.Value.Render(sum.Format),
			theme.Value.Render(strings.Join(dtypeParts, ", ")),
		),
		fmt.Sprintf("%s  %s", theme.Label.Render("Header Hash: "), theme.MonoHash.Render(sum.HeaderSHA256)),
	}
	b.WriteString(theme.Card.Render(strings.Join(metaLines, "\n")) + "\n\n")

	if len(sum.Tensors) == 0 {
		return b.String()
	}

	needle := strings.ToLower(strings.TrimSpace(filter))
	filtered := make([]safetensors.TensorSpec, 0, len(sum.Tensors))
	for _, ts := range sum.Tensors {
		if needle == "" || strings.Contains(strings.ToLower(ts.Name), needle) {
			filtered = append(filtered, ts)
		}
	}

	limit := top
	if limit <= 0 || limit > len(filtered) {
		limit = len(filtered)
	}

	t := newCleanTable("TENSOR PARAMETER", "DTYPE", "SHAPE", "PARAMS", "BYTES")
	for i := 0; i < limit; i++ {
		ts := filtered[i]
		t.Row(
			ts.Name,
			ts.DType,
			formatShape(ts.Shape),
			theme.HumanParams(ts.ParamCount),
			theme.HumanBytes(ts.ByteLength),
		)
	}
	b.WriteString(t.String() + "\n")
	if len(filtered) > limit {
		b.WriteString(theme.Muted.Render(fmt.Sprintf("Showing %d of %d matching tensors (use --top %d to list all).",
			limit, len(filtered), len(filtered))) + "\n")
	}
	return b.String()
}

// RenderQuickDiff formats the millisecond `.safetensors` structural header comparison.
func RenderQuickDiff(diff *safetensors.QuickDiffSummary) string {
	var b strings.Builder
	b.WriteString(theme.HeaderBanner("Fast Structural Header Diff") + "  " + theme.StatusBadge(diff.Status) + "\n\n")

	summary := newCleanTable("METRIC", "MODEL A ("+diff.ModelA+")", "MODEL B ("+diff.ModelB+")", "DELTA / VERDICT")
	deltaSign := ""
	if diff.ParamsDelta > 0 {
		deltaSign = "+"
	}
	summary.Row(
		"Parameters",
		theme.HumanParams(diff.ParamsA),
		theme.HumanParams(diff.ParamsB),
		fmt.Sprintf("%s%s", deltaSign, theme.HumanParams(diff.ParamsDelta)),
	)
	summary.Row(
		"Aligned Tensors",
		fmt.Sprintf("%d", diff.AlignedCount+diff.OnlyInACount+diff.ShapeMismatchCount),
		fmt.Sprintf("%d", diff.AlignedCount+diff.OnlyInBCount+diff.ShapeMismatchCount),
		fmt.Sprintf("%d aligned", diff.AlignedCount),
	)
	summary.Row(
		"Structural Drift",
		fmt.Sprintf("-%d removed", diff.OnlyInACount),
		fmt.Sprintf("+%d added", diff.OnlyInBCount),
		fmt.Sprintf("%d shape / %d dtype diffs", diff.ShapeMismatchCount, diff.DTypeMismatchCount),
	)
	b.WriteString(summary.String() + "\n")

	if len(diff.ShapeMismatches) > 0 {
		b.WriteString("\n" + theme.SectionTitle.Render("Shape Mismatches") + "\n")
		st := newCleanTable("TENSOR", "SHAPE A", "SHAPE B")
		for i, m := range diff.ShapeMismatches {
			if i >= 15 {
				break
			}
			st.Row(m.Name, formatShape(m.ShapeA), formatShape(m.ShapeB))
		}
		b.WriteString(st.String() + "\n")
	}

	if len(diff.OnlyInA) > 0 || len(diff.OnlyInB) > 0 {
		b.WriteString("\n" + theme.SectionTitle.Render("Unaligned Keys (Sample)") + "\n")
		kt := newCleanTable("ORIGIN", "TENSOR PARAMETER")
		for i, k := range diff.OnlyInA {
			if i >= 8 {
				break
			}
			kt.Row("Only in A (-)", k)
		}
		for i, k := range diff.OnlyInB {
			if i >= 8 {
				break
			}
			kt.Row("Only in B (+)", k)
		}
		b.WriteString(kt.String() + "\n")
	}

	return b.String()
}

// BuildEvolutionTree constructs a `charm.land/lipgloss/v2/tree` lineage graph from consecutive transitions.
func BuildEvolutionTree(evo *ledger.EvolutionBenchmark) *tree.Tree {
	childrenMap := make(map[string][]ledger.TensorTransition)
	hasParent := make(map[string]bool)
	parentCount := make(map[string]int)

	for _, tr := range evo.TensorTransitions {
		if tr.Scope != "consecutive" {
			continue
		}
		childrenMap[tr.SourceLabel] = append(childrenMap[tr.SourceLabel], tr)
		hasParent[tr.TargetLabel] = true
		parentCount[tr.TargetLabel]++
	}

	var roots []string
	seenRoot := make(map[string]bool)
	for _, ck := range evo.Checkpoints {
		if !hasParent[ck.VersionLabel] && !seenRoot[ck.VersionLabel] {
			seenRoot[ck.VersionLabel] = true
			roots = append(roots, ck.VersionLabel)
		}
	}
	if len(roots) == 0 && len(evo.Checkpoints) > 0 {
		roots = append(roots, evo.Checkpoints[0].VersionLabel)
	}

	ckptType := make(map[string]string)
	for _, ck := range evo.Checkpoints {
		ckptType[ck.VersionLabel] = ck.TransformationType
	}

	roundedEnum := func(children tree.Children, index int) string {
		return tree.RoundedEnumerator(children, index) + " "
	}

	var buildSubtree func(label string, edge *ledger.TensorTransition, depth int) *tree.Tree
	buildSubtree = func(label string, edge *ledger.TensorTransition, depth int) *tree.Tree {
		nodeText := lipgloss.NewStyle().Bold(true).Foreground(theme.Zinc100).Render(label)
		if edge != nil {
			mergeNote := ""
			if parentCount[label] > 1 {
				mergeNote = fmt.Sprintf(" [merge %d parents]", parentCount[label])
			}
			meta := fmt.Sprintf("  (%s%s  ΔW L2=%.2f  cos=%.5f)",
				edge.TransformationType, mergeNote, edge.L2Distance, edge.GlobalCosineSimilarity)
			nodeText += theme.Muted.Render(meta)
		} else if stg, ok := ckptType[label]; ok && stg != "" {
			nodeText += theme.Muted.Render("  (" + stg + ")")
		}

		sub := tree.Root(nodeText).
			Enumerator(roundedEnum).
			EnumeratorStyle(lipgloss.NewStyle().Foreground(theme.Cyan500))

		if depth > 16 {
			return sub
		}
		for i := range childrenMap[label] {
			ch := childrenMap[label][i]
			sub.Child(buildSubtree(ch.TargetLabel, &ch, depth+1))
		}
		return sub
	}

	rootTree := tree.Root(theme.SectionTitle.Render(fmt.Sprintf("DAG Lineage · %s", evo.RunID))).
		Enumerator(roundedEnum).
		EnumeratorStyle(lipgloss.NewStyle().Foreground(theme.Cyan500))

	for _, r := range roots {
		rootTree.Child(buildSubtree(r, nil, 0))
	}
	return rootTree
}

// RenderEvolutionDAG renders the full Evolution DAG tree, consecutive drift table, and regression summary.
func RenderEvolutionDAG(evo *ledger.EvolutionBenchmark) string {
	var b strings.Builder
	b.WriteString(theme.HeaderBanner("Neural Evolution DAG & Provenance") + "\n")
	b.WriteString(theme.Muted.Render(fmt.Sprintf("%s  •  Cache Key: %s",
		evo.ExperimentType, theme.ShortHash(evo.CacheKey, 16))) + "\n\n")

	kpiLines := fmt.Sprintf(
		"Checkpoints: %d/%d healthy   •   Branches: %d   •   Peak Cumulative L2: %.2f   •   Min Base Cosine: %.6f\n"+
			"Regressions: %d total   •   Behavioral (pass-rate): %d   •   Score (rubric): %d   •   Causally Verified: %d",
		evo.Overview.HealthyCheckpoints,
		evo.Overview.TotalCheckpoints,
		evo.Overview.BranchCount,
		evo.Overview.PeakCumulativeL2,
		evo.Overview.MinBaseCosineSimilarity,
		evo.RegressionSummary.TotalEvents,
		evo.RegressionSummary.BehavioralRegressionObservedCount,
		evo.RegressionSummary.ScoreRegressionObservedCount,
		evo.RegressionSummary.CausalInterventionVerifiedCount,
	)
	b.WriteString(theme.Card.Render(kpiLines) + "\n\n")

	b.WriteString(BuildEvolutionTree(evo).String() + "\n\n")

	b.WriteString(theme.SectionTitle.Render("Weight Drift Transitions") + "\n")
	tt := newCleanTable("SCOPE", "TRANSITION", "STAGE", "L2 DIST", "COSINE", "MODIFIED %", "SPARSITY Δ")
	for _, tr := range evo.TensorTransitions {
		tt.Row(
			tr.Scope,
			tr.SourceLabel+" -> "+tr.TargetLabel,
			tr.TransformationType,
			fmt.Sprintf("%.3f", tr.L2Distance),
			fmt.Sprintf("%.6f", tr.GlobalCosineSimilarity),
			fmt.Sprintf("%.2f%%", tr.ModifiedParameterPct),
			fmt.Sprintf("%+.4f", tr.SparsityDelta),
		)
	}
	b.WriteString(tt.String() + "\n")

	if len(evo.RegressionOrigins) > 0 {
		b.WriteString("\n" + theme.SectionTitle.Render("Regression Origin Ledger (Top Transitions)") + "\n")
		rt := newCleanTable("TRANSITION", "CATEGORY", "FLAGS (B/S/C)", "PASS Δ", "SCORE Δ", "P_HOLM", "EPISTEMIC TIER")
		limit := len(evo.RegressionOrigins)
		if limit > 8 {
			limit = 8
		}
		for i := 0; i < limit; i++ {
			ev := evo.RegressionOrigins[i]
			flags := fmt.Sprintf("B:%s S:%s C:%s",
				shortBool(ev.BehavioralRegressionObserved),
				shortBool(ev.ScoreRegressionObserved),
				shortBool(ev.CausalInterventionVerified),
			)
			rt.Row(
				ev.PreviousLabel+" -> "+ev.FirstAffectedLabel,
				ev.AffectedCategory,
				flags,
				fmt.Sprintf("%.1f%% -> %.1f%%", ev.PassRateBefore*100, ev.PassRateAfter*100),
				fmt.Sprintf("%.3f -> %.3f", ev.ScoreBefore, ev.ScoreAfter),
				fmt.Sprintf("%.3f", ev.HolmBonferroniPValue),
				theme.StatusBadge(ev.EpistemicStatus),
			)
		}
		b.WriteString(rt.String() + "\n")
	}

	return b.String()
}

// RenderPythonJSONResult renders JSON payloads returned by `core.cli` into clean Lip Gloss summaries on TTY.
func RenderPythonJSONResult(subcommand string, rawJSON []byte) string {
	var generic map[string]any
	if err := json.Unmarshal(rawJSON, &generic); err != nil {
		return string(rawJSON)
	}

	var b strings.Builder
	b.WriteString(theme.HeaderBanner("Result · "+subcommand) + "\n\n")

	t := newCleanTable("FIELD", "VALUE")
	keys := make([]string, 0, len(generic))
	for k := range generic {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	for _, k := range keys {
		val := generic[k]
		switch v := val.(type) {
		case string:
			if strings.Contains(strings.ToLower(k), "verdict") || strings.Contains(strings.ToLower(k), "status") {
				t.Row(k, theme.StatusBadge(v))
			} else {
				t.Row(k, v)
			}
		case float64:
			if v == float64(int64(v)) {
				t.Row(k, fmt.Sprintf("%d", int64(v)))
			} else {
				t.Row(k, fmt.Sprintf("%.6f", v))
			}
		case bool:
			t.Row(k, fmt.Sprintf("%v", v))
		default:
			compact, _ := json.Marshal(v)
			s := string(compact)
			if len(s) > 88 {
				s = s[:85] + "..."
			}
			t.Row(k, s)
		}
	}
	b.WriteString(t.String() + "\n")
	return b.String()
}

func formatShape(shape []int64) string {
	if len(shape) == 0 {
		return "[]"
	}
	parts := make([]string, len(shape))
	for i, d := range shape {
		parts[i] = fmt.Sprintf("%d", d)
	}
	return "[" + strings.Join(parts, " × ") + "]"
}

func nonEmpty(s, fallback string) string {
	if strings.TrimSpace(s) == "" {
		return fallback
	}
	return s
}

func statusTag(ok bool) string {
	if ok {
		return theme.StatusBadge("OK")
	}
	return theme.StatusBadge("IDLE")
}

func shortBool(v bool) string {
	if v {
		return "Y"
	}
	return "N"
}
