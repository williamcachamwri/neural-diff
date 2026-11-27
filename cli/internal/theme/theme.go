package theme

import (
	"fmt"
	"image/color"
	"math"
	"os"
	"strings"

	"charm.land/fang/v2"
	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/term"
)

// Hardware-instrument palette: neutral zinc surfaces with high-contrast semantic accents.
// Strictly zero emojis, zero neon gradients, and restrained rounded geometry.
var (
	Zinc950 = lipgloss.Color("#09090b")
	Zinc900 = lipgloss.Color("#18181b")
	Zinc800 = lipgloss.Color("#27272a")
	Zinc700 = lipgloss.Color("#3f3f46")
	Zinc500 = lipgloss.Color("#71717a")
	Zinc400 = lipgloss.Color("#a1a1aa")
	Zinc200 = lipgloss.Color("#e4e4e7")
	Zinc100 = lipgloss.Color("#f4f4f5")

	Cyan400    = lipgloss.Color("#22d3ee")
	Cyan500    = lipgloss.Color("#06b6d4")
	Emerald400 = lipgloss.Color("#34d399")
	Amber400   = lipgloss.Color("#fbbf24")
	Rose400    = lipgloss.Color("#fb7185")
	Violet400  = lipgloss.Color("#a78bfa")
)

// Styles bundles reusable Lip Gloss v2 styles for consistent terminal rendering.
var (
	BrandPill = lipgloss.NewStyle().
			Bold(true).
			Foreground(Zinc950).
			Background(Cyan400).
			Padding(0, 1)

	SubtlePill = lipgloss.NewStyle().
			Foreground(Zinc200).
			Background(Zinc800).
			Padding(0, 1)

	Title = lipgloss.NewStyle().
		Bold(true).
		Foreground(Zinc100)

	SectionTitle = lipgloss.NewStyle().
			Bold(true).
			Foreground(Cyan400)

	Muted = lipgloss.NewStyle().
		Foreground(Zinc500)

	Label = lipgloss.NewStyle().
		Foreground(Zinc400)

	Value = lipgloss.NewStyle().
		Foreground(Zinc100)

	MonoHash = lipgloss.NewStyle().
			Foreground(Zinc400)

	BorderColor = Zinc700

	Card = lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(Zinc700).
		Padding(0, 1)

	ActiveTab = lipgloss.NewStyle().
			Bold(true).
			Foreground(Zinc950).
			Background(Cyan400).
			Padding(0, 1)

	InactiveTab = lipgloss.NewStyle().
			Foreground(Zinc400).
			Background(Zinc900).
			Padding(0, 1)

	TableHeader = lipgloss.NewStyle().
			Bold(true).
			Foreground(Zinc200).
			Padding(0, 1)

	TableCell = lipgloss.NewStyle().
			Foreground(Zinc200).
			Padding(0, 1)

	TableCellMuted = lipgloss.NewStyle().
			Foreground(Zinc400).
			Padding(0, 1)
)

// IsStdoutTTY reports whether standard output is connected to an interactive terminal.
func IsStdoutTTY() bool {
	return term.IsTerminal(os.Stdout.Fd())
}

// IsStderrTTY reports whether standard error is connected to an interactive terminal.
func IsStderrTTY() bool {
	return term.IsTerminal(os.Stderr.Fd())
}

// NoColorEnabled follows https://no-color.org/ and explicit flag overrides.
func NoColorEnabled(explicitNoColor bool) bool {
	if explicitNoColor {
		return true
	}
	_, ok := os.LookupEnv("NO_COLOR")
	return ok
}

// FangColorScheme configures charm.land/fang/v2 help, usage, and error pages
// with a clean zinc + cyan instrument palette.
func FangColorScheme(ld lipgloss.LightDarkFunc) fang.ColorScheme {
	fg := ld(lipgloss.Color("#18181b"), Zinc100)
	muted := ld(lipgloss.Color("#52525b"), Zinc400)
	dim := ld(lipgloss.Color("#71717a"), Zinc500)
	accent := ld(lipgloss.Color("#0891b2"), Cyan400)
	emerald := ld(lipgloss.Color("#059669"), Emerald400)
	amber := ld(lipgloss.Color("#d97706"), Amber400)
	codeBg := ld(lipgloss.Color("#f4f4f5"), Zinc900)

	return fang.ColorScheme{
		Base:           fg,
		Title:          accent,
		Description:    muted,
		Codeblock:      codeBg,
		Program:        accent,
		DimmedArgument: dim,
		Comment:        dim,
		Flag:           emerald,
		FlagDefault:    dim,
		Command:        fg,
		QuotedString:   amber,
		Argument:       fg,
		Help:           muted,
		Dash:           dim,
		ErrorHeader:    [2]color.Color{Zinc950, Rose400},
		ErrorDetails:   Rose400,
	}
}

// StatusBadge formats semantic status or epistemic tier tags without emojis.
func StatusBadge(status string) string {
	norm := strings.ToUpper(strings.TrimSpace(status))
	switch {
	case norm == "PASSED" || norm == "COMPATIBLE" || norm == "HEALTHY" || norm == "CAUSALLY VERIFIED" || norm == "TRUE" || norm == "OK":
		return lipgloss.NewStyle().Bold(true).Foreground(Emerald400).Render("[" + norm + "]")
	case norm == "CAUSAL SUPPORT" || norm == "INFERRED":
		return lipgloss.NewStyle().Bold(true).Foreground(Cyan400).Render("[" + norm + "]")
	case norm == "CORRELATED" || norm == "WARNING" || norm == "PARTIAL" || norm == "EXPLORATORY":
		return lipgloss.NewStyle().Bold(true).Foreground(Amber400).Render("[" + norm + "]")
	case strings.Contains(norm, "FAIL") || strings.Contains(norm, "INCOMPATIBLE") || strings.Contains(norm, "BLOCKED") || norm == "REGRESSION":
		return lipgloss.NewStyle().Bold(true).Foreground(Rose400).Render("[" + norm + "]")
	default:
		return lipgloss.NewStyle().Foreground(Zinc400).Render("[" + norm + "]")
	}
}

// Bar renders a compact fixed-width Unicode block bar in [0, 1].
func Bar(ratio float64, width int, fg color.Color) string {
	if width <= 0 {
		return ""
	}
	if math.IsNaN(ratio) || ratio < 0 {
		ratio = 0
	}
	if ratio > 1 {
		ratio = 1
	}
	filled := int(math.Round(ratio * float64(width)))
	if filled > width {
		filled = width
	}
	fillStr := strings.Repeat("█", filled)
	emptyStr := strings.Repeat("░", width-filled)
	return lipgloss.NewStyle().Foreground(fg).Render(fillStr) +
		lipgloss.NewStyle().Foreground(Zinc700).Render(emptyStr)
}

// ShortHash truncates a hex digest cleanly for compact tabular display.
func ShortHash(h string, keep int) string {
	h = strings.TrimSpace(h)
	if h == "" {
		return "—"
	}
	if keep <= 0 || len(h) <= keep {
		return h
	}
	return h[:keep]
}

// HumanParams formats parameter counts cleanly (e.g. 134.52M, 494.03M, 7.62B).
func HumanParams(n int64) string {
	switch {
	case n >= 1_000_000_000:
		return fmt.Sprintf("%.2fB", float64(n)/1e9)
	case n >= 1_000_000:
		return fmt.Sprintf("%.2fM", float64(n)/1e6)
	case n >= 1_000:
		return fmt.Sprintf("%.2fK", float64(n)/1e3)
	default:
		return fmt.Sprintf("%d", n)
	}
}

// HumanBytes formats byte sizes cleanly using binary units.
func HumanBytes(b int64) string {
	const unit = 1024
	if b < unit {
		return fmt.Sprintf("%d B", b)
	}
	div, exp := int64(unit), 0
	for n := b / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %ciB", float64(b)/float64(div), "KMGTPE"[exp])
}

// HeaderBanner renders a crisp single-line instrument header.
func HeaderBanner(subtitle string) string {
	left := BrandPill.Render("NEURAL DIFF")
	if subtitle == "" {
		return left
	}
	return left + "  " + Title.Render(subtitle)
}
