import { expect, test } from "@playwright/test";

test.describe("NEURAL DIFF Neural Evolution Tracking Live E2E", () => {
  test("creates an evolution project, registers checkpoints, runs analysis, inspects DAG, and previews export", async ({
    page,
  }) => {
    await page.goto("/");
    await expect(page.getByText(/Backend online/i)).toBeVisible({ timeout: 15_000 });

    // 1. Open Neural Evolution Studio
    await page.getByTestId("nav-evolution").click();
    await expect(page.getByTestId("evolution-studio-view")).toBeVisible();

    // 2. Create a new Evolution Project
    const uniqueProj = `E2E Evolution ${Date.now()}`;
    await page.getByTestId("input-new-project-name").fill(uniqueProj);
    await page.getByTestId("input-new-project-desc").fill("Playwright full-stack DAG verification");
    await page.getByTestId("btn-create-project").click();
    await expect(page.getByTestId("evo-project-selector")).toContainText(uniqueProj);

    // 3. Register Base checkpoint (synth_model_a) and Child SFT checkpoint (synth_model_b)
    await page.getByTestId("evo-tab-registry").click();
    await expect(page.getByTestId("evo-section-registry")).toBeVisible();

    await page.getByTestId("input-ckpt-source").fill("data/demo_fixtures/synth_model_a");
    await page.getByTestId("input-ckpt-label").fill("v0-base");
    await page.getByTestId("select-ckpt-transform").selectOption("Base Model");
    await page.getByTestId("btn-register-checkpoint").click();
    await expect(page.getByTestId("evo-checkpoint-details")).toContainText("v0-base");

    // Select v0-base as parent for v1-sft
    const parentSelect = page.getByTestId("select-ckpt-parents");
    const firstOptVal = await parentSelect.locator("option").first().getAttribute("value");
    if (firstOptVal) {
      await parentSelect.selectOption([firstOptVal]);
    }
    await page.getByTestId("input-ckpt-source").fill("data/demo_fixtures/synth_model_b");
    await page.getByTestId("input-ckpt-label").fill("v1-sft");
    await page.getByTestId("select-ckpt-transform").selectOption("SFT");
    await page.getByTestId("btn-register-checkpoint").click();
    await expect(page.getByTestId("evo-checkpoint-details")).toContainText("v1-sft");

    // 4. Execute live multi-generation Evolution Analysis
    await page.getByTestId("btn-run-evolution-analysis").click();
    await expect(page.getByTestId("btn-run-evolution-analysis")).toContainText(
      "Run Evolution Analysis",
      { timeout: 30_000 }
    );

    // 5. Inspect Interactive DAG, Weights, Layers, Representations & Export Preview
    await page.getByTestId("evo-tab-graph").click();
    await expect(page.getByTestId("evolution-dag-canvas")).toBeVisible();
    await page.getByTestId("btn-evo-compare-pair").click();
    await expect(page.getByTestId("evo-arbitrary-compare-panel")).toContainText("Direct L2");

    await page.getByTestId("evo-tab-weights").click();
    await expect(page.getByTestId("evo-section-weights")).toContainText("Direct Cumulative Weight");

    await page.getByTestId("evo-tab-layers").click();
    await expect(page.getByTestId("evo-section-layers")).toBeVisible();

    await page.getByTestId("evo-tab-representations").click();
    await expect(page.getByTestId("evo-section-representations")).toContainText("Linear CKA");

    await page.getByTestId("evo-tab-experiments").click();
    await expect(page.getByTestId("btn-evo-download-pdf")).toBeVisible();
    await page.getByTestId("btn-evo-preview-csv").click();
    await expect(page.getByTestId("evo-export-preview-content")).toContainText(
      "[CHECKPOINT_REGISTRY]"
    );
  });
});
