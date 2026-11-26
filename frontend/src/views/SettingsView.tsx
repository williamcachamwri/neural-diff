import React, { useEffect, useState } from "react";
import { apiClient } from "../api/client";
import type { RuntimeSettings } from "../types/api";

interface Props {
  onError: (err: Error) => void;
  onSettingsSaved: () => void;
}

export const SettingsView: React.FC<Props> = ({ onError, onSettingsSaved }) => {
  const [settings, setSettings] = useState<RuntimeSettings | null>(null);
  const [savedBanner, setSavedBanner] = useState(false);

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
      setSavedBanner(true);
      onSettingsSaved();
      setTimeout(() => setSavedBanner(false), 3000);
    } catch (err) {
      onError(err instanceof Error ? err : new Error(String(err)));
    }
  };

  if (!settings) {
    return (
      <div className="panel" data-testid="view-settings">
        Loading persisted runtime settings...
      </div>
    );
  }

  return (
    <div className="panel" data-testid="view-settings">
      <div className="panel-header">
        <h2 className="panel-title">Engine &amp; Security Settings</h2>
        {savedBanner && <span className="badge badge-ok">Persisted to SQLite</span>}
      </div>

      <form onSubmit={handleSave}>
        <div className="grid-2">
          <div className="field-group">
            <label htmlFor="set-device">Default Compute Device</label>
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
              <option value="auto">auto (CUDA if verified, else CPU)</option>
              <option value="cuda">cuda</option>
              <option value="cpu">cpu</option>
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
            <label htmlFor="set-atol">Default Absolute Tolerance (atol)</label>
            <input
              id="set-atol"
              type="number"
              step="any"
              value={settings.atol}
              onChange={(e) => setSettings({ ...settings, atol: Number(e.target.value) })}
            />
          </div>

          <div className="field-group">
            <label htmlFor="set-rtol">Default Relative Tolerance (rtol)</label>
            <input
              id="set-rtol"
              type="number"
              step="any"
              value={settings.rtol}
              onChange={(e) => setSettings({ ...settings, rtol: Number(e.target.value) })}
            />
          </div>

          <div className="field-group">
            <label htmlFor="set-tokens">Default Behavioral Max New Tokens</label>
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
            <label>Checkpoint Serialization Policy</label>
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
              <span>
                Allow safe PyTorch <code>weights_only=True</code> + opcode-scanned archives (Safetensors always preferred)
              </span>
            </label>
          </div>
        </div>

        <button type="submit" className="btn btn-primary" data-testid="btn-save-settings">
          Save Settings
        </button>
      </form>
    </div>
  );
};
