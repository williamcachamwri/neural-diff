package engine

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"charm.land/bubbles/v2/spinner"
	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"

	"github.com/williamcachamwri/neural-diff/cli/internal/theme"
)

// ResolvePythonExecutable finds the workspace virtualenv Python or system Python 3.
func ResolvePythonExecutable(repoRoot string) string {
	if custom := os.Getenv("NEURAL_DIFF_PYTHON"); custom != "" {
		return custom
	}
	if repoRoot != "" {
		if runtime.GOOS == "windows" {
			venvWin := filepath.Join(repoRoot, ".venv", "Scripts", "python.exe")
			if fileExists(venvWin) {
				return venvWin
			}
		} else {
			venvUnix := filepath.Join(repoRoot, ".venv", "bin", "python")
			if fileExists(venvUnix) {
				return venvUnix
			}
		}
	}
	for _, name := range []string{"python3", "python"} {
		if p, err := exec.LookPath(name); err == nil {
			return p
		}
	}
	return ""
}

type execDoneMsg struct {
	stdout []byte
	stderr []byte
	err    error
	dur    time.Duration
}

type spinModel struct {
	spin   spinner.Model
	label  string
	runCmd func() execDoneMsg
	result *execDoneMsg
}

func (m spinModel) Init() tea.Cmd {
	return tea.Batch(
		m.spin.Tick,
		func() tea.Msg {
			return m.runCmd()
		},
	)
}

func (m spinModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyPressMsg:
		if msg.String() == "ctrl+c" {
			return m, tea.Quit
		}
	case execDoneMsg:
		m.result = &msg
		return m, tea.Quit
	case spinner.TickMsg:
		var cmd tea.Cmd
		m.spin, cmd = m.spin.Update(msg)
		return m, cmd
	}
	return m, nil
}

func (m spinModel) View() tea.View {
	if m.result != nil {
		return tea.NewView("")
	}
	line := fmt.Sprintf(" %s %s",
		m.spin.View(),
		theme.Muted.Render(m.label),
	)
	return tea.NewView(line)
}

// RunPythonCLI executes `python -m core.cli --json <args...>` and returns raw stdout/stderr.
// When interactiveSpinner is true and stderr is a TTY, displays a minimal Bubble Tea v2 spinner.
func RunPythonCLI(ctx context.Context, repoRoot string, cliArgs []string, interactiveSpinner bool, label string) ([]byte, time.Duration, error) {
	py := ResolvePythonExecutable(repoRoot)
	if py == "" {
		return nil, 0, errors.New("python executable not found; set NEURAL_DIFF_PYTHON or create .venv")
	}

	fullArgs := append([]string{"-m", "core.cli", "--json"}, cliArgs...)
	runner := func() execDoneMsg {
		start := time.Now()
		cmd := exec.CommandContext(ctx, py, fullArgs...)
		if repoRoot != "" {
			cmd.Dir = repoRoot
		}
		cmd.Env = append(os.Environ(), "NEURAL_DIFF_CLI_JSON=1")
		var outBuf, errBuf bytes.Buffer
		cmd.Stdout = &outBuf
		cmd.Stderr = &errBuf
		err := cmd.Run()
		return execDoneMsg{
			stdout: outBuf.Bytes(),
			stderr: errBuf.Bytes(),
			err:    err,
			dur:    time.Since(start),
		}
	}

	var done execDoneMsg
	if interactiveSpinner && theme.IsStderrTTY() && !theme.NoColorEnabled(false) {
		sp := spinner.New(
			spinner.WithSpinner(spinner.MiniDot),
			spinner.WithStyle(lipgloss.NewStyle().Foreground(theme.Cyan400)),
		)
		m := spinModel{
			spin:   sp,
			label:  label,
			runCmd: runner,
		}
		p := tea.NewProgram(m, tea.WithOutput(os.Stderr))
		finalModel, err := p.Run()
		if err != nil {
			done = runner()
		} else if sm, ok := finalModel.(spinModel); ok && sm.result != nil {
			done = *sm.result
		} else {
			return nil, 0, errors.New("operation interrupted")
		}
	} else {
		done = runner()
	}

	if done.err != nil {
		errText := strings.TrimSpace(string(done.stderr))
		if errText == "" {
			errText = done.err.Error()
		}
		return done.stdout, done.dur, fmt.Errorf("%s", errText)
	}
	return done.stdout, done.dur, nil
}

func fileExists(p string) bool {
	info, err := os.Stat(p)
	return err == nil && !info.IsDir()
}
