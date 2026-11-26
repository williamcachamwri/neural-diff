import React, { useEffect, useState } from "react";
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
        <h2 className="panel-title">Report Viewer &amp; Export</h2>
        <p style={{ color: "var(--text-muted)" }}>
          Select a completed comparison to preview and export HTML, Markdown, or JSON reports.
        </p>
      </div>
    );
  }

  const cid = encodeURIComponent(activeBundle.comparison_id);

  return (
    <div data-testid="view-reports">
      <div className="panel">
        <div className="panel-header">
          <h2 className="panel-title">
            Report Viewer &amp; Export (<span className="mono">{activeBundle.comparison_id}</span>)
          </h2>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
            <button
              className={`btn ${format === "markdown" ? "btn-primary" : ""}`}
              onClick={() => setFormat("markdown")}
            >
              Preview Markdown
            </button>
            <button
              className={`btn ${format === "html" ? "btn-primary" : ""}`}
              onClick={() => setFormat("html")}
            >
              Preview Interactive HTML
            </button>
            <button
              className={`btn ${format === "json" ? "btn-primary" : ""}`}
              onClick={() => setFormat("json")}
            >
              Preview Strict JSON
            </button>
            <a
              className="btn"
              href={`/api/comparisons/${cid}/export/html`}
              download={`${activeBundle.comparison_id}.html`}
            >
              Download .html
            </a>
            <a
              className="btn"
              href={`/api/comparisons/${cid}/export/markdown`}
              download={`${activeBundle.comparison_id}.md`}
            >
              Download .md
            </a>
            <a
              className="btn"
              href={`/api/comparisons/${cid}/export/json`}
              download={`${activeBundle.comparison_id}.json`}
            >
              Download .json
            </a>
          </div>
        </div>

        {format === "html" ? (
          <iframe
            title="NEURAL DIFF HTML Report Preview"
            sandbox=""
            srcDoc={content}
            style={{
              width: "100%",
              height: 620,
              border: "1px solid var(--border)",
              borderRadius: 8,
              background: "#0b1020",
            }}
          />
        ) : (
          <pre
            className="diff-box mono"
            style={{ maxHeight: 620, overflow: "auto", margin: 0 }}
          >
            {content || "Loading report artifact..."}
          </pre>
        )}
      </div>
    </div>
  );
};
