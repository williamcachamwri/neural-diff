package tui

import (
	"fmt"
	"path/filepath"
	"strings"

	"charm.land/bubbles/v2/textinput"
	"charm.land/bubbles/v2/viewport"
	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
	"charm.land/lipgloss/v2/table"

	"github.com/williamcachamwri/neural-diff/cli/internal/ledger"
	"github.com/williamcachamwri/neural-diff/cli/internal/render"
	"github.com/williamcachamwri/neural-diff/cli/internal/safetensors"
	"github.com/williamcachamwri/neural-diff/cli/internal/theme"
)

var tabTitles = []string{
	"1 · Overview",
	"2 · Evolution DAG",
	"3 · Checkpoints & SHA-256",
	"4 · Regressions",
	"5 · Tensor Explorer",
}

// StudioConfig configures the initial state of the Bubble Tea v2 Terminal Studio.
type StudioConfig struct {
	RepoRoot      string
	PythonExe     string
	EvolutionFile string
	ModelPath     string
}

// StudioModel implements `tea.Model` (Bubble Tea v2 declarative `tea.View` + AltScreen).
type StudioModel struct {
	cfg           StudioConfig
	status        ledger.WorkspaceStatus
	evo           *ledger.EvolutionBenchmark
	evoPath       string
	ckptSummary   *safetensors.CheckpointSummary
	activeTab     int
	width         int
	height        int
	vp            viewport.Model
	filterInput   textinput.Model
	filtering     bool
	filterQuery   string
	statusMessage string
}

// NewStudioModel loads workspace telemetry, evolution DAG ledger, and `.safetensors` header in < 5ms.
func NewStudioModel(cfg StudioConfig) StudioModel {
	vp := viewport.New(viewport.WithWidth(100), viewport.WithHeight(24))
	vp.MouseWheelEnabled = true

	ti := textinput.New()
	ti.Placeholder = "filter tensor or category..."
	ti.CharLimit = 96
	ti.SetWidth(36)

	m := StudioModel{
		cfg:         cfg,
		activeTab:   0,
		width:       104,
		height:      32,
		vp:          vp,
		filterInput: ti,
	}
	m.reload()
	return m
}

func (m *StudioModel) reload() {
	m.status = ledger.InspectWorkspace(m.cfg.RepoRoot, m.cfg.PythonExe)
	if evo, p, err := ledger.LoadEvolutionBenchmark(m.cfg.RepoRoot, m.cfg.EvolutionFile); err == nil {
		m.evo = evo
		m.evoPath = p
	}
	modelTarget := m.cfg.ModelPath
	if modelTarget == "" && m.cfg.RepoRoot != "" {
		discovered := ledger.DiscoverCheckpoints(m.cfg.RepoRoot)
		if len(discovered) > 0 {
			modelTarget = filepath.Join(m.cfg.RepoRoot, filepath.FromSlash(discovered[0]))
		}
	}
	if modelTarget != "" {
		if sum, err := safetensors.InspectSafetensors(modelTarget, true); err == nil {
			m.ckptSummary = sum
		}
	}
	m.refreshViewport()
}

// Init satisfies `tea.Model`.
func (m StudioModel) Init() tea.Cmd {
	return nil
}

// Update handles keyboard navigation, tab switching, live `/` filtering, and window resize.
func (m StudioModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		vpHeight := msg.Height - 6
		if vpHeight < 6 {
			vpHeight = 6
		}
		vpWidth := msg.Width - 2
		if vpWidth < 40 {
			vpWidth = 40
		}
		m.vp.SetWidth(vpWidth)
		m.vp.SetHeight(vpHeight)
		m.refreshViewport()
		return m, nil

	case tea.KeyPressMsg:
		if m.filtering {
			switch msg.String() {
			case "esc":
				m.filtering = false
				m.filterInput.Blur()
				m.refreshViewport()
				return m, nil
			case "enter":
				m.filtering = false
				m.filterQuery = strings.TrimSpace(m.filterInput.Value())
				m.filterInput.Blur()
				m.vp.GotoTop()
				m.refreshViewport()
				return m, nil
			default:
				var cmd tea.Cmd
				m.filterInput, cmd = m.filterInput.Update(msg)
				m.filterQuery = strings.TrimSpace(m.filterInput.Value())
				m.refreshViewport()
				return m, cmd
			}
		}

		switch msg.String() {
		case "q", "ctrl+c":
			return m, tea.Quit
		case "1":
			m.setTab(0)
			return m, nil
		case "2":
			m.setTab(1)
			return m, nil
		case "3":
			m.setTab(2)
			return m, nil
		case "4":
			m.setTab(3)
			return m, nil
		case "5":
			m.setTab(4)
			return m, nil
		case "tab", "l", "right":
			m.setTab((m.activeTab + 1) % len(tabTitles))
			return m, nil
		case "shift+tab", "h", "left":
			m.setTab((m.activeTab - 1 + len(tabTitles)) % len(tabTitles))
			return m, nil
		case "/":
			m.filtering = true
			cmd := m.filterInput.Focus()
			return m, cmd
		case "c":
			m.filterQuery = ""
			m.filterInput.SetValue("")
			m.statusMessage = "filter cleared"
			m.refreshViewport()
			return m, nil
		case "r":
			m.reload()
			m.statusMessage = "reloaded workspace ledgers"
			return m, nil
		case "g":
			m.vp.GotoTop()
			return m, nil
		case "G":
			m.vp.GotoBottom()
			return m, nil
		}
	}

	var cmd tea.Cmd
	m.vp, cmd = m.vp.Update(msg)
	return m, cmd
}

func (m *StudioModel) setTab(idx int) {
	m.activeTab = idx
	m.statusMessage = ""
	m.vp.GotoTop()
	m.refreshViewport()
}

func (m *StudioModel) refreshViewport() {
	switch m.activeTab {
	case 0:
		m.vp.SetContent(render.RenderStatus(m.status))
	case 1:
		if m.evo == nil {
			m.vp.SetContent(theme.Muted.Render("No evolution ledger found. Run `ndiff evolution analyze` or pass `--file <path.json>`."))
		} else {
			m.vp.SetContent(render.RenderEvolutionDAG(m.evo))
		}
	case 2:
		m.vp.SetContent(m.renderCheckpointsTab())
	case 3:
		m.vp.SetContent(m.renderRegressionsTab())
	case 4:
		if m.ckptSummary == nil {
			m.vp.SetContent(theme.Muted.Render("No local .safetensors checkpoint detected. Pass `--model <dir>` or run `scripts/seed_demo_fixtures.py`."))
		} else {
			m.vp.SetContent(render.RenderCheckpointSummary(m.ckptSummary, m.filterQuery, 200))
		}
	}
}

func (m StudioModel) renderCheckpointsTab() string {
	if m.evo == nil || len(m.evo.Checkpoints) == 0 {
		return theme.Muted.Render("No registered evolution checkpoints loaded.")
	}
	var b strings.Builder
	b.WriteString(theme.HeaderBanner("Checkpoint Registry & 5-Component Fingerprints") + "\n\n")

	t := table.New().
		Border(lipgloss.RoundedBorder()).
		BorderStyle(lipgloss.NewStyle().Foreground(theme.Zinc700)).
		Headers("LABEL", "STAGE", "PARAMS", "SIZE", "COMPOSITE", "WEIGHTS", "CONFIG", "TOKENIZER", "METADATA").
		StyleFunc(func(row, _ int) lipgloss.Style {
			if row == table.HeaderRow {
				return theme.TableHeader
			}
			if row%2 == 0 {
				return theme.TableCell
			}
			return theme.TableCellMuted
		})

	needle := strings.ToLower(m.filterQuery)
	for _, ck := range m.evo.Checkpoints {
		if needle != "" &&
			!strings.Contains(strings.ToLower(ck.VersionLabel), needle) &&
			!strings.Contains(strings.ToLower(ck.TransformationType), needle) {
			continue
		}
		fp := ck.ComponentFingerprints
		t.Row(
			ck.VersionLabel,
			ck.TransformationType,
			theme.HumanParams(ck.ParameterCount),
			theme.HumanBytes(ck.FileSizeBytes),
			theme.ShortHash(ck.SHA256Fingerprint, 10),
			theme.ShortHash(fp.WeightsSHA256, 10),
			theme.ShortHash(fp.ArchitectureConfigSHA256, 10),
			theme.ShortHash(fp.TokenizerSHA256, 10),
			theme.ShortHash(fp.MetadataSHA256, 10),
		)
	}
	b.WriteString(t.String() + "\n")

	if len(m.evo.Representation) > 0 {
		b.WriteString("\n" + theme.SectionTitle.Render("Centered Linear CKA & Logit JS Divergence (vs Base)") + "\n")
		rt := table.New().
			Border(lipgloss.RoundedBorder()).
			BorderStyle(lipgloss.NewStyle().Foreground(theme.Zinc700)).
			Headers("CHECKPOINT", "LINEAR CKA", "ALIGNMENT", "REP DRIFT", "LOGIT JS DIV").
			StyleFunc(func(row, _ int) lipgloss.Style {
				if row == table.HeaderRow {
					return theme.TableHeader
				}
				return theme.TableCell
			})
		for _, rp := range m.evo.Representation {
			rt.Row(
				rp.VersionLabel,
				fmt.Sprintf("%.6f", rp.MeanCKAToBase),
				theme.Bar(rp.MeanCKAToBase, 14, theme.Emerald400),
				fmt.Sprintf("%.6f", rp.RepresentationDrift),
				fmt.Sprintf("%.6f", rp.MeanLogitJSToBase),
			)
		}
		b.WriteString(rt.String() + "\n")
	}
	return b.String()
}

func (m StudioModel) renderRegressionsTab() string {
	if m.evo == nil {
		return theme.Muted.Render("No evolution ledger loaded.")
	}
	var b strings.Builder
	b.WriteString(theme.HeaderBanner("Regression Origin & Epistemic Verification") + "\n\n")

	sum := m.evo.RegressionSummary
	cardText := fmt.Sprintf(
		"Total Events: %d   •   Behavioral (pass-rate): %d   •   Score (rubric): %d   •   Causal Verified: %d   •   Holm Sig: %d",
		sum.TotalEvents,
		sum.BehavioralRegressionObservedCount,
		sum.ScoreRegressionObservedCount,
		sum.CausalInterventionVerifiedCount,
		sum.StatisticallySignificantCount,
	)
	b.WriteString(theme.Card.Render(cardText) + "\n\n")

	t := table.New().
		Border(lipgloss.RoundedBorder()).
		BorderStyle(lipgloss.NewStyle().Foreground(theme.Zinc700)).
		Headers("EDGE", "CATEGORY", "KIND", "BEHAV", "SCORE", "CAUSAL", "PASS RATE", "MEAN SCORE", "P_HOLM", "TIER").
		StyleFunc(func(row, _ int) lipgloss.Style {
			if row == table.HeaderRow {
				return theme.TableHeader
			}
			if row%2 == 0 {
				return theme.TableCell
			}
			return theme.TableCellMuted
		})

	needle := strings.ToLower(m.filterQuery)
	for _, ev := range m.evo.RegressionOrigins {
		if needle != "" &&
			!strings.Contains(strings.ToLower(ev.AffectedCategory), needle) &&
			!strings.Contains(strings.ToLower(ev.EpistemicStatus), needle) &&
			!strings.Contains(strings.ToLower(ev.FirstAffectedLabel), needle) {
			continue
		}
		t.Row(
			ev.PreviousLabel+" -> "+ev.FirstAffectedLabel,
			ev.AffectedCategory,
			ev.RegressionKind,
			flagBadge(ev.BehavioralRegressionObserved),
			flagBadge(ev.ScoreRegressionObserved),
			flagBadge(ev.CausalInterventionVerified),
			fmt.Sprintf("%.1f%% -> %.1f%%", ev.PassRateBefore*100, ev.PassRateAfter*100),
			fmt.Sprintf("%.3f -> %.3f", ev.ScoreBefore, ev.ScoreAfter),
			fmt.Sprintf("%.3f", ev.HolmBonferroniPValue),
			theme.StatusBadge(ev.EpistemicStatus),
		)
	}
	b.WriteString(t.String() + "\n")
	return b.String()
}

func flagBadge(v bool) string {
	if v {
		return lipgloss.NewStyle().Bold(true).Foreground(theme.Cyan400).Render("YES")
	}
	return theme.Muted.Render("no")
}

// View renders the declarative Bubble Tea v2 view (`AltScreen = true`).
func (m StudioModel) View() tea.View {
	var tabs []string
	for i, name := range tabTitles {
		if i == m.activeTab {
			tabs = append(tabs, theme.ActiveTab.Render(name))
		} else {
			tabs = append(tabs, theme.InactiveTab.Render(name))
		}
	}
	topBar := lipgloss.JoinHorizontal(lipgloss.Top,
		theme.BrandPill.Render("NEURAL DIFF"),
		" ",
		strings.Join(tabs, " "),
	)

	divider := theme.Muted.Render(strings.Repeat("─", maxInt(40, m.width-2)))

	var footerLeft string
	if m.filtering {
		footerLeft = theme.SectionTitle.Render("Filter: ") + m.filterInput.View() + theme.Muted.Render("  (enter apply • esc cancel)")
	} else {
		keysHelp := "[1-5/tab] switch view  [j/k/pgdn] scroll  [/] filter  [c] clear  [r] reload  [q] quit"
		if m.filterQuery != "" {
			keysHelp += fmt.Sprintf("   •   active filter: %q", m.filterQuery)
		}
		if m.statusMessage != "" {
			keysHelp += "   •   " + m.statusMessage
		}
		footerLeft = theme.Muted.Render(keysHelp)
	}
	scrollPct := fmt.Sprintf(" %3.0f%% ", m.vp.ScrollPercent()*100)
	footer := footerLeft + "  " + theme.SubtlePill.Render(scrollPct)

	screen := strings.Join([]string{
		topBar,
		divider,
		m.vp.View(),
		divider,
		footer,
	}, "\n")

	v := tea.NewView(screen)
	v.AltScreen = true
	v.WindowTitle = "NEURAL DIFF · Terminal Studio"
	return v
}

// RunStudio launches the interactive Bubble Tea v2 studio.
func RunStudio(cfg StudioConfig) error {
	p := tea.NewProgram(NewStudioModel(cfg))
	_, err := p.Run()
	return err
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
