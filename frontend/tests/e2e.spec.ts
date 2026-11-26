import { expect, test } from "@playwright/test";

test.describe("NEURAL DIFF Full-Stack Live E2E Workflow", () => {
  test("verifies live backend health, compatibility validation, error handling, and all 12 views", async ({
    page,
  }) => {
    await page.goto("/");

    // 1. Dashboard loads real backend status
    await expect(page.getByTestId("view-dashboard")).toBeVisible();
    await expect(page.getByText(/BACKEND ONLINE/i)).toBeVisible({ timeout: 15_000 });

    // 2. Navigate to Compatibility Check and trigger real error handling on disallowed traversal path
    await page.getByTestId("nav-compatibility").click();
    await expect(page.getByTestId("view-compatibility")).toBeVisible();
    await page.getByTestId("compat-model-a").fill("../../outside_forbidden.safetensors");
    await page.getByTestId("compat-model-b").fill("../../outside_forbidden_b.safetensors");
    await page.getByTestId("btn-validate-compatibility").click();

    await expect(page.getByTestId("error-banner")).toBeVisible();
    await expect(page.getByTestId("error-banner")).toContainText("SECURITY_VALIDATION_ERROR");
    await page.getByTestId("btn-dismiss-error").click();
    await expect(page.getByTestId("error-banner")).toBeHidden();

    // 3. Validate compatibility on real local demo fixtures
    await page.getByTestId("compat-model-a").fill("data/demo_fixtures/synth_model_a");
    await page.getByTestId("compat-model-b").fill("data/demo_fixtures/synth_model_b");
    await page.getByTestId("btn-validate-compatibility").click();
    await expect(page.getByTestId("compatibility-result")).toBeVisible();
    await expect(page.getByTestId("compatibility-result")).toContainText("21/21");

    // 4. Execute a live comparison through the UI against real FastAPI backend
    await page.getByTestId("nav-compare").click();
    await expect(page.getByTestId("view-compare")).toBeVisible();
    await page.getByTestId("input-model-a").fill("data/demo_fixtures/synth_model_a");
    await page.getByTestId("input-model-b").fill("data/demo_fixtures/synth_model_b");
    await page.getByTestId("btn-run-comparison").click();

    // Automatically transitions to Tensor Explorer once comparison completes
    await expect(page.getByTestId("view-tensors")).toBeVisible({ timeout: 25_000 });
    await expect(page.getByTestId("tensor-detail-inspector")).toBeVisible();

    // 5. Verify remaining views render live bundle measurements
    for (const navId of [
      "structural",
      "layers",
      "activations",
      "behavioral",
      "regression",
      "history",
      "reports",
      "settings",
    ]) {
      await page.getByTestId(`nav-${navId}`).click();
      await expect(page.getByTestId(`view-${navId}`)).toBeVisible();
    }
  });
});
