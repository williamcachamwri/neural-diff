import { describe, expect, it } from "vitest";
import {
  formatCategory,
  formatMetricNumber,
  formatRole,
  formatSeverity,
  formatStatus,
  formatVerdict,
} from "../utils/format";

describe("Studio human-centered formatters", () => {
  it("formats raw snake_case and enum strings cleanly", () => {
    expect(formatVerdict("FAILED_REGRESSION")).toBe("Regression detected");
    expect(formatVerdict("PASSED")).toBe("Passed");
    expect(formatCategory("code_preservation")).toBe("Code Preservation");
    expect(formatCategory("json_reliability")).toBe("JSON Schema");
    expect(formatRole("attn_q")).toBe("Query (Q)");
    expect(formatSeverity("CRITICAL")).toBe("Critical");
    expect(formatStatus("both_passed")).toBe("Maintained");
  });

  it("formats metric floats cleanly without scientific noise on standard ranges", () => {
    expect(formatMetricNumber(0.8666667, 3)).toBe("0.867");
    expect(formatMetricNumber(0.0000421, 3)).toBe("4.21e-5");
    expect(formatMetricNumber(null)).toBe("—");
  });
});
