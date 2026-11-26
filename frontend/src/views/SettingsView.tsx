import React, { useEffect, useState } from "react";
import { apiClient } from "../api/client";
import type { RuntimeSettings } from "../types/api";

interface Props {
  onError: (err: Error) => void;
  onSettingsSaved: () => void;
}

export const SettingsView: React.FC<Props> = ({ onError, onSettingsSaved }) => {
  const [settings, setSettings] = useState<RuntimeSettings | null>(null);
  const [savedNotice, setSavedNotice] = useState(false);

  useEffect(() => {
    apiClient
      .getSettings()
      .then(setSettings)
      .catch((err) => onError(err instanceof Error ? err : new Error(String(err))));
  }, [onError]);

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!settings) return;
    try {
      const updated = await apiClient.updateSettings(settings);
      setSettings(updated);
      setSavedNotice(true);
      onSettingsSaved();
      setTimeout(() => setSavedNotice(false), 2500);
    } catch (err) {
      onError(err instanceof Error ? err : new Error(String(err)));
    }
  };

  if (!settings) {
    return (
      <div className="panel" data-testid="view-settings">
        <span style={{ color: "var(--text-muted)", fontSize: 13 }}>Loading runtime settings...</span>
      </div>
    );
  }

  return (
    <div className="panel" data-testid="view-settings">
      <div className="panel-header">
        <div>
          <h2 className="panel-title">Runtime &amp; Serialization Settings</h2>
          <div className="panel-subtitle" style={{ marginTop: 2 }}>
            Default compute backend, streaming block size, and checkpoint loading rules
          </div>
        </div>
        {savedNotice && (
          <span className="status-inline">
            <span className="dot dot-ok" />
            <span>Saved</span>
          </span>
        )}
      </div>

      <form onSubmit={handleSave}>
        <div className="grid-2">
          <div className="field-group">
            <label htmlFor="set-device">Compute Backend</label>
            <select
              id="set-device"
              value={settings.preferred_device}
              onChange={(e) =>
                setSettings({
                  ...settings,
                  preferred_device: e.target.value as RuntimeSettings["preferred_device"],
                })
              }
            >
              <option value="auto">Auto (CUDA with CPU fallback)</option>
              <option value="cuda">CUDA</option>
              <option value="cpu">CPU</option>
            </select>
          </div>

          <div className="field-group">
            <label htmlFor="set-chunk">Streaming Chunk Size (Elements)</label>
            <input
              id="set-chunk"
              type="number"
              min={4096}
              max={16777216}
              value={settings.chunk_elements}
              onChange={(e) =>
                setSettings({ ...settings, chunk_elements: Number(e.target.value) })
              }
            />
          </div>

          <div className="field-group">
            <label htmlFor="set-atol">Absolute Tolerance (atol)</label>
            <input
              id="set-atol"
              type="number"
              step="any"
              value={settings.atol}
              onChange={(e) => setSettings({ ...settings, atol: Number(e.target.value) })}
            />
          </div>

          <div className="field-group">
            <label htmlFor="set-rtol">Relative Tolerance (rtol)</label>
            <input
              id="set-rtol"
              type="number"
              step="any"
              value={settings.rtol}
              onChange={(e) => setSettings({ ...settings, rtol: Number(e.target.value) })}
            />
          </div>

          <div className="field-group">
            <label htmlFor="set-tokens">Max Generated Tokens</label>
            <input
              id="set-tokens"
              type="number"
              min={8}
              max={512}
              value={settings.max_new_tokens}
              onChange={(e) =>
                setSettings({ ...settings, max_new_tokens: Number(e.target.value) })
              }
            />
          </div>

          <div className="field-group">
            <label>PyTorch Archive Policy</label>
            <label className="check-tile">
              <input
                type="checkbox"
                checked={settings.allow_pytorch_weights_only}
                onChange={(e) =>
                  setSettings({
                    ...settings,
                    allow_pytorch_weights_only: e.target.checked,
                  })
                }
              />
              <span>Allow opcode-verified PyTorch weights_only archives</span>
            </label>
          </div>
        </div>

        <button type="submit" className="btn btn-primary" data-testid="btn-save-settings">
          Save Changes
        </button>
      </form>
    </div>
  );
};
