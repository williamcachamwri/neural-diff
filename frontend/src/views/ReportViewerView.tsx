import React, { useEffect, useState } from "react";
import { Download } from "lucide-react";
import { apiClient } from "../api/client";
import type { ComparisonBundle } from "../types/api";

interface Props {
  activeBundle: ComparisonBundle | null;
  onError: (err: Error) => void;
}

export const ReportViewerView: React.FC<Props> = ({ activeBundle, onError }) => {
  const [format, setFormat] = useState<"markdown" | "html" | "json">("markdown");
  const [content, setContent] = useState<string>("");

  useEffect(() => {
    if (!activeBundle) {
      setContent("");
      return;
    }
    let cancelled = false;
    apiClient
      .fetchReportText(activeBundle.comparison_id, format)
      .then((txt) => {
        if (!cancelled) setContent(txt);
      })
      .catch((err) => {
        if (!cancelled) onError(err instanceof Error ? err : new Error(String(err)));
      });
    return () => {
      cancelled = true;
    };
  }, [activeBundle, format, onError]);

  if (!activeBundle) {
    return (
      <div className="panel" data-testid="view-reports">
        <h2 className="panel-title">Reports &amp; Export</h2>
        <p style={{ color: "var(--text-muted)", fontSize: 13 }}>
          Select a completed comparison to preview and export reports.
        </p>
      </div>
    );
  }

  const cid = encodeURIComponent(activeBundle.comparison_id);

  return (
    <div data-testid="view-reports">
      <div className="panel">
        <div className="panel-header" style={{ alignItems: "center" }}>
          <div>
            <h2 className="panel-title">Report Artifacts</h2>
            <div className="panel-subtitle mono" style={{ marginTop: 2 }}>
              {activeBundle.comparison_id}
            </div>
          </div>

          <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 10 }}>
            <div className="segmented-bar">
              <button
                type="button"
                className={`segmented-btn ${format === "markdown" ? "active" : ""}`}
                onClick={() => setFormat("markdown")}
              >
                Markdown
              </button>
              <button
                type="button"
                className={`segmented-btn ${format === "html" ? "active" : ""}`}
                onClick={() => setFormat("html")}
              >
                Preview Interactive HTML
              </button>
              <button
                type="button"
                className={`segmented-btn ${format === "json" ? "active" : ""}`}
                onClick={() => setFormat("json")}
              >
                JSON
              </button>
            </div>

            <div style={{ display: "inline-flex", gap: 6 }}>
              <a
                className="btn"
                href={`/api/comparisons/${cid}/export/html`}
                download={`${activeBundle.comparison_id}.html`}
              >
                <Download size={12} />
                <span>.html</span>
              </a>
              <a
                className="btn"
                href={`/api/comparisons/${cid}/export/markdown`}
                download={`${activeBundle.comparison_id}.md`}
              >
                <Download size={12} />
                <span>.md</span>
              </a>
              <a
                className="btn"
                href={`/api/comparisons/${cid}/export/json`}
                download={`${activeBundle.comparison_id}.json`}
              >
                <Download size={12} />
                <span>.json</span>
              </a>
            </div>
          </div>
        </div>

        {format === "html" ? (
          <iframe
            title="NEURAL DIFF HTML Report Preview"
            sandbox=""
            srcDoc={content}
            style={{
              width: "100%",
              height: 600,
              border: "1px solid var(--border-subtle)",
              borderRadius: 8,
              background: "#09090b",
            }}
          />
        ) : (
          <pre
            className="diff-box mono"
            style={{ maxHeight: 600, overflow: "auto", margin: 0 }}
          >
            {content || "Loading report..."}
          </pre>
        )}
      </div>
    </div>
  );
};
