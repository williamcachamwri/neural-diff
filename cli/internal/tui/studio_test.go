package tui

import (
	"path/filepath"
	"strings"
	"testing"

	tea "charm.land/bubbletea/v2"
)

func TestStudioModelNavigationAndFiltering(t *testing.T) {
	repoRoot, err := filepath.Abs(filepath.Join("..", "..", ".."))
	if err != nil {
		t.Fatal(err)
	}

	m := NewStudioModel(StudioConfig{
		RepoRoot: repoRoot,
	})

	// Resize window
	updated, _ := m.Update(tea.WindowSizeMsg{Width: 120, Height: 36})
	m = updated.(StudioModel)

	v0 := m.View()
	if !v0.AltScreen {
		t.Fatal("expected Bubble Tea v2 View.AltScreen to be true")
	}
	if !strings.Contains(v0.Content, "NEURAL DIFF") {
		t.Fatal("expected header banner in rendered TUI view")
	}

	// Switch across all 5 tabs via key presses
	for idx, keyName := range []string{"2", "3", "4", "5", "1"} {
		next, _ := m.Update(tea.KeyPressMsg{Code: rune(keyName[0]), Text: keyName})
		m = next.(StudioModel)
		v := m.View()
		if v.Content == "" {
			t.Fatalf("empty view content at step %d (key %s)", idx, keyName)
		}
	}

	// Activate live filter ('/'), type 'json', press enter
	next, _ := m.Update(tea.KeyPressMsg{Code: '/', Text: "/"})
	m = next.(StudioModel)
	if !m.filtering {
		t.Fatal("expected filtering mode to activate on '/'")
	}

	for _, r := range "json" {
		next, _ = m.Update(tea.KeyPressMsg{Code: r, Text: string(r)})
		m = next.(StudioModel)
	}
	next, _ = m.Update(tea.KeyPressMsg{Code: tea.KeyEnter})
	m = next.(StudioModel)
	if m.filtering {
		t.Fatal("expected filtering mode to exit on Enter")
	}
	if m.filterQuery != "json" {
		t.Fatalf("expected filterQuery='json', got %q", m.filterQuery)
	}

	// Clear filter ('c') and quit ('q')
	next, _ = m.Update(tea.KeyPressMsg{Code: 'c', Text: "c"})
	m = next.(StudioModel)
	if m.filterQuery != "" {
		t.Fatalf("expected cleared filterQuery, got %q", m.filterQuery)
	}

	_, quitCmd := m.Update(tea.KeyPressMsg{Code: 'q', Text: "q"})
	if quitCmd == nil {
		t.Fatal("expected Quit cmd on 'q'")
	}
}
