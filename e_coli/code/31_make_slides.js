/*
 * GFP variants project — results deck generator.
 *
 * Builds a PowerPoint summarising steps 1-4: data cleaning, EDA, split
 * design, baselines, the 8-model comparison, the full-data MLP, and the
 * extrapolation analysis. Every figure slide carries the same three
 * sections: Objective / What it shows / Conclusion.
 *
 * Run with: node make_slides.js
 * Output  : ../output/GFP_ML_results.pptx
 */

const pptxgen = require("pptxgenjs");
const path = require("path");

const OUT = path.join(__dirname, "..", "output");
const fig = (name) => path.join(OUT, name);

// Palette built around GFP's own emission colour rather than a generic blue.
const DARK = "0B2818";   // deep forest, near-black
const FOREST = "2C5F2D"; // primary green
const GLOW = "6FDC3C";   // fluorescent GFP green — the accent
const INK = "1A2620";    // body text
const MUTED = "5F6F62";  // captions
const PANEL = "F1F5F0";  // subtle card tint
const WHITE = "FFFFFF";

const HEAD = "Cambria";  // safe-list serif for headers
const BODY = "Calibri";  // safe-list sans for body

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";            // 13.3 x 7.5 in — set before adding slides
pres.author = "ML for GFP";
pres.title = "Predicting GFP brightness from sequence";

/* ---------- helpers ---------------------------------------------------- */

function slideTitle(slide, text, colour = INK) {
  slide.addText(text, {
    x: 0.5, y: 0.3, w: 12.3, h: 0.7, isTextBox: true,
    fontFace: HEAD, fontSize: 32, bold: true, color: colour, margin: 0,
  });
}

// The repeated motif: a small glowing dot before every section header.
function sectionBlock(slide, { x, y, w, label, lines, labelColour = FOREST }) {
  slide.addShape(pres.ShapeType.ellipse, {
    x: x, y: y + 0.07, w: 0.13, h: 0.13, fill: { color: GLOW },
  });
  slide.addText(label.toUpperCase(), {
    x: x + 0.24, y: y - 0.02, w: w - 0.24, h: 0.28, isTextBox: true,
    fontFace: BODY, fontSize: 12, bold: true, color: labelColour,
    charSpacing: 1.2, margin: 0,
  });
  slide.addText(
    lines.map((t, i) => ({
      text: t,
      options: { bullet: lines.length > 1, breakLine: i < lines.length - 1 },
    })),
    {
      x: x, y: y + 0.32, w: w, h: 1.5, isTextBox: true,
      fontFace: BODY, fontSize: 13, color: INK, margin: 0,
      paraSpaceAfter: 4, valign: "top",
    }
  );
}

function figureSlide(slide, { title, image, imgX, imgY, imgW, imgH, cols, notes }) {
  slideTitle(slide, title);
  slide.addImage({ path: image, x: imgX, y: imgY, w: imgW, h: imgH });
  cols.forEach((c) => sectionBlock(slide, c));
  if (notes) slide.addNotes(notes);
}

function statCard(slide, { x, y, w, h, value, label, tint = PANEL }) {
  slide.addShape(pres.ShapeType.roundRect, {
    x, y, w, h, rectRadius: 0.08, fill: { color: tint },
  });
  slide.addText(value, {
    x: x + 0.15, y: y + 0.12, w: w - 0.3, h: 0.55, isTextBox: true,
    fontFace: HEAD, fontSize: 28, bold: true, color: FOREST, margin: 0,
  });
  slide.addText(label, {
    x: x + 0.15, y: y + 0.68, w: w - 0.3, h: 0.45, isTextBox: true,
    fontFace: BODY, fontSize: 11, color: MUTED, margin: 0,
  });
}

/* ---------- 1. title --------------------------------------------------- */
{
  const s = pres.addSlide();
  s.background = { color: DARK };
  s.addText("Predicting GFP brightness\nfrom protein sequence", {
    x: 0.8, y: 1.6, w: 9.5, h: 2.0, isTextBox: true,
    fontFace: HEAD, fontSize: 40, bold: true, color: WHITE,
    lineSpacing: 46, margin: 0,
  });
  s.addText("A machine-learning pipeline over 141,150 deep-mutational-scan variants", {
    x: 0.8, y: 3.75, w: 10.5, h: 0.4, isTextBox: true,
    fontFace: BODY, fontSize: 17, color: GLOW, margin: 0,
  });
  s.addText(
    "E. coli FACS-seq libraries — avGFP · amacGFP · ppluGFP2 · cgreGFP\n" +
    "Sarkisyan et al. 2016 · Gonzalez Somermeyer et al. 2022",
    {
      x: 0.8, y: 4.4, w: 10.5, h: 0.9, isTextBox: true,
      fontFace: BODY, fontSize: 13, color: "AFC3B2", lineSpacing: 20, margin: 0,
    }
  );
  s.addText("Steps 1–4: cleaning · splits · baselines · model comparison", {
    x: 0.8, y: 6.4, w: 10.5, h: 0.4, isTextBox: true,
    fontFace: BODY, fontSize: 12, italic: true, color: "7E927F", margin: 0,
  });
  s.addNotes("Goal: predict fold-wild-type brightness from amino-acid sequence, " +
    "so candidate variants can be ranked before wet-lab validation.");
}

/* ---------- 2. objective & dataset ------------------------------------- */
{
  const s = pres.addSlide();
  slideTitle(s, "Objective and dataset");

  s.addText(
    "Predict fold-wild-type brightness from an amino-acid sequence, so that candidate " +
    "variants can be ranked before wet-lab validation. All four libraries are " +
    "random-mutagenesis deep mutational scans read out by FACS-seq in E. coli.",
    {
      x: 0.5, y: 1.15, w: 12.3, h: 0.8, isTextBox: true,
      fontFace: BODY, fontSize: 14, color: INK, margin: 0,
    }
  );

  const cards = [
    { value: "141,150", label: "variants after cleaning" },
    { value: "4", label: "protein backbones" },
    { value: "6,625", label: "model features" },
    { value: "2–3", label: "median mutations per variant" },
  ];
  cards.forEach((c, i) =>
    statCard(s, { x: 0.5 + i * 3.14, y: 2.2, w: 2.9, h: 1.25, ...c })
  );

  sectionBlock(s, {
    x: 0.5, y: 3.95, w: 5.9, label: "Label",
    lines: [
      "Brightness = variant MFI ÷ wild-type MFI of the same backbone",
      "Raw MFI is not comparable across libraries — see slide 5",
    ],
  });
  sectionBlock(s, {
    x: 6.9, y: 3.95, w: 5.9, label: "Rows per backbone",
    lines: [
      "avGFP 51,715   ·   amacGFP 33,511",
      "ppluGFP2 31,402   ·   cgreGFP 24,516",
    ],
  });
  s.addNotes("Every backbone has a single fixed sequence length — no indels — " +
    "so position-wise encoding needs no alignment.");
}

/* ---------- 3. cleaning ------------------------------------------------ */
{
  const s = pres.addSlide();
  slideTitle(s, "Step 1 — building a modelling-ready table");

  const steps = [
    ["141,469", "rows merged from 15 source files"],
    ["−3", "sequences with unresolved X residues"],
    ["−316", "rows with no usable brightness label"],
    ["141,150", "final modelling table"],
  ];
  steps.forEach(([v, l], i) => {
    const x = 0.5 + i * 3.14;
    statCard(s, {
      x, y: 1.2, w: 2.9, h: 1.45, value: v, label: l,
      tint: i === 3 ? "E2F3D9" : PANEL,
    });
  });

  sectionBlock(s, {
    x: 0.5, y: 3.25, w: 5.9, label: "What was dropped, and why",
    lines: [
      "49 natural wild-types — no same-family WT to normalise against",
      "264 ML-designed variants — colony assay, no WT reference",
      "“% positive cells” column — 100% empty after cleaning",
    ],
  });
  sectionBlock(s, {
    x: 6.9, y: 3.25, w: 5.9, label: "A tagging bug worth flagging",
    lines: [
      "Grouping rows by their classification label mis-filed 4 wild-type rows " +
      "as a different assay",
      "Fixed by grouping on the analysing method instead — brightness is only " +
      "comparable within one assay",
    ],
  });
  sectionBlock(s, {
    x: 0.5, y: 5.15, w: 12.3, label: "Conclusion",
    lines: [
      "141,150 rows on a single consistent FACS-seq scale; 6 classic engineered " +
      "variants (EGFP, sfGFP…) are held aside because they were measured by " +
      "molecular brightness (EC×QY) and sit on a completely different scale.",
    ],
  });
}

/* ---------- 4. EDA figure ---------------------------------------------- */
{
  const s = pres.addSlide();
  figureSlide(s, {
    title: "Step 1 — label distribution",
    image: fig("eda_distributions.png"),
    imgX: 1.15, imgY: 1.1, imgW: 11.0, imgH: 11.0 / 2.67,
    cols: [
      {
        x: 0.5, y: 5.4, w: 3.9, label: "Objective",
        lines: ["Characterise the label before choosing a model class."],
      },
      {
        x: 4.7, y: 5.4, w: 3.9, label: "What it shows",
        lines: [
          "Brightness is strongly bimodal — a spike at ~0 and a peak at ~1.0",
          "Same shape in all four backbones; most variants carry 2–3 mutations",
        ],
      },
      {
        x: 8.9, y: 5.4, w: 3.9, label: "Conclusion",
        lines: [
          "~34% of variants are non-functional. A purely additive regressor " +
          "cannot express “one lethal mutation”, so expect it to struggle here.",
        ],
      },
    ],
    notes: "The dark spike is loss-of-function variants. This motivates both the " +
      "two-stage idea and the non-linear models tested later.",
  });
}

/* ---------- 5. cross-library scale finding ----------------------------- */
{
  const s = pres.addSlide();
  slideTitle(s, "Step 1 — brightness is not comparable across libraries");

  s.addText(
    "Two identical protein sequences appear in two different DMS libraries — a natural " +
    "experiment for cross-library calibration. An amacGFP variant with 41 mutations " +
    "converged to exactly the avGFP F64L sequence.",
    {
      x: 0.5, y: 1.1, w: 12.3, h: 0.7, isTextBox: true,
      fontFace: BODY, fontSize: 14, color: INK, margin: 0,
    }
  );

  const rows = [
    ["", "measured in avGFP library", "measured in amacGFP library"],
    ["Same protein — raw MFI", "5,238.56", "2,291.56"],
    ["Same protein — fold-WT brightness", "1.000", "0.245"],
    ["Effect of H169L on that background", "0.865×", "1.540×"],
  ];
  s.addTable(
    rows.map((r, ri) =>
      r.map((c, ci) => ({
        text: c,
        options: {
          bold: ri === 0 || ci === 0,
          color: ri === 0 ? WHITE : INK,
          fill: { color: ri === 0 ? FOREST : ri % 2 ? WHITE : PANEL },
          fontSize: 14, fontFace: BODY,
          align: ci === 0 ? "left" : "center",
        },
      }))
    ),
    { x: 0.5, y: 2.05, w: 12.3, colW: [5.1, 3.6, 3.6], rowH: 0.52, border: { pt: 0 } }
  );

  sectionBlock(s, {
    x: 0.5, y: 4.5, w: 5.9, label: "What it shows",
    lines: [
      "Identical proteins read 4.1× apart after normalisation",
      "Normalising made the gap worse, not better (2.3× raw → 4.1×)",
      "The same mutation moves in opposite directions in the two libraries",
    ],
  });
  sectionBlock(s, {
    x: 6.9, y: 4.5, w: 5.9, label: "Conclusion",
    lines: [
      "Fold-WT normalisation does not put libraries on one scale.",
      "Keep backbone as an explicit model feature, and score anything " +
      "cross-backbone with rank metrics only — never RMSE.",
    ],
  });
  s.addNotes("Because the background sequence is literally identical, the H169L " +
    "disagreement is assay noise rather than biology. It puts a rough floor on " +
    "achievable accuracy.");
}

/* ---------- 6. split design -------------------------------------------- */
{
  const s = pres.addSlide();
  slideTitle(s, "Step 2 — split design, and why it matters");

  s.addText(
    "DMS variants differ from wild-type by only 2–3 mutations, so near-identical " +
    "sequences are everywhere. A plain random split scatters them across train and " +
    "test, letting a model score well purely by memorisation.",
    {
      x: 0.5, y: 1.1, w: 12.3, h: 0.7, isTextBox: true,
      fontFace: BODY, fontSize: 14, color: INK, margin: 0,
    }
  );

  const schemes = [
    ["Random", "Stratified by backbone and dark/bright, duplicate-aware",
      "Optimistic upper bound — memorisation allowed"],
    ["Position-holdout", "Whole mutated positions withheld from training",
      "Realistic: can the model score an unseen site?"],
    ["Backbone-holdout", "Train on 3 backbones, test on the 4th",
      "Hardest: cross-protein transfer (not yet run)"],
  ];
  schemes.forEach(([name, how, why], i) => {
    const x = 0.5 + i * 4.18;
    s.addShape(pres.ShapeType.roundRect, {
      x, y: 2.0, w: 3.95, h: 2.35, rectRadius: 0.08, fill: { color: PANEL },
    });
    s.addText(name, {
      x: x + 0.2, y: 2.18, w: 3.55, h: 0.4, isTextBox: true,
      fontFace: HEAD, fontSize: 18, bold: true, color: FOREST, margin: 0,
    });
    s.addText(how, {
      x: x + 0.2, y: 2.62, w: 3.55, h: 0.85, isTextBox: true,
      fontFace: BODY, fontSize: 12, color: INK, margin: 0,
    });
    s.addText(why, {
      x: x + 0.2, y: 3.52, w: 3.55, h: 0.7, isTextBox: true,
      fontFace: BODY, fontSize: 12, italic: true, color: MUTED, margin: 0,
    });
  });

  sectionBlock(s, {
    x: 0.5, y: 4.65, w: 5.9, label: "Verification",
    lines: [
      "Zero training rows touch a held-out position, in every backbone",
      "Zero duplicate-sequence leakage across the random split",
    ],
  });
  sectionBlock(s, {
    x: 6.9, y: 4.65, w: 5.9, label: "Conclusion",
    lines: [
      "Report random and position-holdout side by side. The gap between them " +
      "is the honest measure of how much a model memorises.",
    ],
  });
}

/* ---------- 7. baselines figure ---------------------------------------- */
{
  const s = pres.addSlide();
  figureSlide(s, {
    title: "Step 3 — baselines and their failure modes",
    image: fig("baseline_results.png"),
    imgX: 0.65, imgY: 1.1, imgW: 12.0, imgH: 12.0 / 3.48,
    cols: [
      {
        x: 0.5, y: 4.85, w: 3.9, label: "Objective",
        lines: ["Establish a baseline and diagnose where it breaks."],
      },
      {
        x: 4.7, y: 4.85, w: 3.9, label: "What it shows",
        lines: [
          "Ridge on combined features leads: ρ = 0.880 random, 0.814 position",
          "Middle: truly dead variants are predicted at 0.5–0.75, not 0",
          "Right: accuracy falls steadily as more of the variant is unseen",
        ],
      },
      {
        x: 8.9, y: 4.85, w: 3.9, label: "Conclusion",
        lines: [
          "Strong overall, but the additive model cannot represent a single " +
          "lethal mutation — and its position features say nothing at all " +
          "about an unseen site.",
        ],
      },
    ],
    notes: "On pure extrapolation the sparse-feature ridge scores exactly the " +
      "mean-predictor floor (0.268 both), because a never-seen position has no " +
      "fitted coefficient.",
  });
}

/* ---------- 8. model comparison figure --------------------------------- */
{
  const s = pres.addSlide();
  figureSlide(s, {
    title: "Step 4 — eight model families, 104 configurations",
    image: fig("model_comparison.png"),
    imgX: 1.0, imgY: 1.05, imgW: 11.3, imgH: 11.3 / 2.8,
    cols: [
      {
        x: 0.5, y: 5.3, w: 3.9, label: "Objective",
        lines: [
          "Compare families on identical features and test sets, tuning each on " +
          "the validation split only.",
        ],
      },
      {
        x: 4.7, y: 5.3, w: 3.9, label: "What it shows",
        lines: [
          "Linear models lead the full tier (ρ ≈ 0.88); XGBoost trails at 0.813",
          "RandomForest and kernel SVR are weakest (ρ ≈ 0.66–0.70)",
          "MLP leads at equal data in the 12k tier",
        ],
      },
      {
        x: 8.9, y: 5.3, w: 3.9, label: "Conclusion",
        lines: [
          "The landscape is largely additive, which is exactly what linear " +
          "models fit best. L1 actively hurts — both Lasso and ElasticNet chose " +
          "the smallest penalty offered.",
        ],
      },
    ],
    notes: "Two tiers were needed because RandomForest and kernel SVR cannot train " +
      "on 113k rows. The sub12k tier trains on 12,000 rows but scores the same full " +
      "test set, so the tiers are directly comparable.",
  });
}

/* ---------- 9. full-data MLP ------------------------------------------- */
{
  const s = pres.addSlide();
  slideTitle(s, "Step 4c — the MLP at full scale");

  s.addText(
    "The 12k tier suggested the MLP would win given all the data. Trained on the full " +
    "set via epoch-level checkpointing, it does — and the R² gap is far wider than the " +
    "ranking gap.",
    {
      x: 0.5, y: 1.1, w: 12.3, h: 0.7, isTextBox: true,
      fontFace: BODY, fontSize: 14, color: INK, margin: 0,
    }
  );

  const rows = [
    ["Split / metric", "ridge", "MLP (full data)"],
    ["Random — Spearman ρ", "0.880", "0.911"],
    ["Random — R²", "0.705", "0.870"],
    ["Position-holdout — Spearman ρ", "0.814", "0.823"],
    ["Position-holdout — R²", "0.552", "0.621"],
  ];
  s.addTable(
    rows.map((r, ri) =>
      r.map((c, ci) => ({
        text: c,
        options: {
          bold: ri === 0 || ci === 0 || ci === 2,
          color: ri === 0 ? WHITE : INK,
          fill: { color: ri === 0 ? FOREST : ri % 2 ? WHITE : PANEL },
          fontSize: 14, fontFace: BODY,
          align: ci === 0 ? "left" : "center",
        },
      }))
    ),
    { x: 0.5, y: 2.05, w: 7.4, colW: [3.6, 1.9, 1.9], rowH: 0.5, border: { pt: 0 } }
  );

  statCard(s, {
    x: 8.3, y: 2.05, w: 4.5, h: 1.3,
    value: "0.705 → 0.870", label: "test R², random split — ridge → MLP",
    tint: "E2F3D9",
  });
  sectionBlock(s, {
    x: 8.3, y: 3.6, w: 4.5, label: "Why that matters",
    lines: [
      "The MLP is not just ranking better — it predicts actual brightness far " +
      "more accurately, which is where the additive models failed.",
    ],
  });

  sectionBlock(s, {
    x: 0.5, y: 4.75, w: 7.4, label: "Method note",
    lines: [
      "A single full-data fit needs ~9 minutes, longer than one run allows, so " +
      "training drives epochs manually: score the real validation split each " +
      "epoch, keep the best epoch's weights, checkpoint to disk, resume.",
      "Both runs early-stopped at epoch 6 of 10 — so this is a lower bound.",
    ],
  });
  sectionBlock(s, {
    x: 0.5, y: 6.35, w: 12.3, label: "Conclusion", labelColour: FOREST,
    lines: [
      "The MLP is the best overall model, and the R² jump says it is capturing " +
      "the non-additive dark mode the linear models could not.",
    ],
  });
}

/* ---------- 10. extrapolation figure ----------------------------------- */
{
  const s = pres.addSlide();
  slideTitle(s, "Step 4d — does that advantage survive real extrapolation?");
  s.addImage({ path: fig("extrapolation_mlp.png"), x: 0.5, y: 1.15, w: 7.6, h: 7.6 / 2.0 });

  sectionBlock(s, {
    x: 8.35, y: 1.15, w: 4.45, label: "Objective",
    lines: [
      "The headline numbers lean on multi-mutants whose other mutations were " +
      "already seen. Strip that away.",
    ],
  });
  sectionBlock(s, {
    x: 8.35, y: 2.65, w: 4.45, label: "What it shows",
    lines: [
      "All rows: MLP leads at ρ = 0.823",
      "Single mutants at unseen sites: MLP 0.372, ridge 0.377, XGBoost 0.443",
      "Every model collapses toward the floor as novelty rises",
    ],
  });
  sectionBlock(s, {
    x: 8.35, y: 4.75, w: 4.45, label: "Conclusion",
    lines: [
      "The MLP's gain is entirely recombination of known mutations.",
      "XGBoost — last on the headline metric — is the best extrapolator.",
    ],
  });

  s.addShape(pres.ShapeType.roundRect, {
    x: 0.5, y: 5.5, w: 7.6, h: 1.35, rectRadius: 0.08, fill: { color: "FBF2DC" },
  });
  s.addText(
    [
      { text: "Suggestive, not established.  ", options: { bold: true } },
      {
        text: "Paired bootstrap over the 293 single-mutant rows: XGBoost beats the " +
          "MLP by Δρ = +0.070, 95% CI [−0.003, +0.144], P(Δ>0) = 0.97. The interval " +
          "just crosses zero, and 293 rows is thin — treat it as a lead to confirm.",
      },
    ],
    {
      x: 0.75, y: 5.68, w: 7.1, h: 1.0, isTextBox: true,
      fontFace: BODY, fontSize: 12, color: "6B4E12", margin: 0,
    }
  );
  s.addNotes("This is the most decision-relevant slice in the analysis and also the " +
    "thinnest. A position-holdout split weighted toward single mutants would sharpen it.");
}

/* ---------- 11. cross-validated extrapolation benchmark ---------------- */
{
  const s = pres.addSlide();
  slideTitle(s, "Step 5 — the extrapolation result, settled");
  s.addImage({ path: fig("extrapolation_cv.png"), x: 0.5, y: 1.1, w: 8.1, h: 8.1 / 2.71 });

  sectionBlock(s, {
    x: 8.85, y: 1.1, w: 3.95, label: "Objective",
    lines: [
      "Step 4d rested on 293 single mutants and a CI that crossed zero. " +
      "The dataset holds 4,596.",
    ],
  });
  sectionBlock(s, {
    x: 8.85, y: 2.5, w: 3.95, label: "Method",
    lines: [
      "5-fold cross-validation over positions — every variant tested once by " +
      "a model that never saw any of its sites. 15.7× more evidence, same data.",
    ],
  });
  sectionBlock(s, {
    x: 8.85, y: 4.2, w: 3.95, label: "Conclusion",
    lines: [
      "XGBoost is decisively the best extrapolator, and wins in all four backbones.",
      "The MLP is significantly worse than plain ridge.",
    ],
  });

  const boot = [
    ["xgboost − mlp", "+0.112", "[+0.091, +0.133]"],
    ["xgboost − ridge", "+0.072", "[+0.054, +0.091]"],
    ["ridge − mlp", "+0.039", "[+0.017, +0.061]"],
  ];
  s.addShape(pres.ShapeType.roundRect, {
    x: 0.5, y: 4.55, w: 8.1, h: 2.25, rectRadius: 0.08, fill: { color: PANEL },
  });
  s.addText("Paired bootstrap, 4,596 single mutants — all three now significant", {
    x: 0.75, y: 4.72, w: 7.6, h: 0.3, isTextBox: true,
    fontFace: BODY, fontSize: 12, bold: true, color: FOREST, margin: 0,
  });
  boot.forEach(([c, d, ci], i) => {
    const y = 5.15 + i * 0.45;
    s.addText(c, {
      x: 0.75, y, w: 2.9, h: 0.35, isTextBox: true, valign: "middle",
      fontFace: BODY, fontSize: 13, color: INK, margin: 0,
    });
    s.addText(d, {
      x: 3.7, y, w: 1.4, h: 0.35, isTextBox: true, valign: "middle",
      fontFace: BODY, fontSize: 13, bold: true, color: FOREST, margin: 0,
    });
    s.addText("95% CI " + ci, {
      x: 5.2, y, w: 3.2, h: 0.35, isTextBox: true, valign: "middle",
      fontFace: BODY, fontSize: 13, color: MUTED, margin: 0,
    });
  });
  s.addNotes("The MLP's extra capacity is spent learning the specific mutation " +
    "landscape it was shown, which is exactly what fails to transfer. ppluGFP2 is " +
    "the one backbone where the models tie, and also where all of them do worst.");
}

/* ---------- 12. leave-one-backbone-out --------------------------------- */
{
  const s = pres.addSlide();
  figureSlide(s, {
    title: "Step 7 — leave-one-backbone-out: predicting a new protein",
    image: fig("lobo.png"),
    imgX: 1.25, imgY: 1.05, imgW: 10.8, imgH: 10.8 / 2.23,
    cols: [
      {
        x: 0.5, y: 6.0, w: 3.9, label: "Objective",
        lines: ["Train on three backbones, predict the fourth — the hardest " +
                "transfer test, and the last one outstanding."],
      },
      {
        x: 4.7, y: 6.0, w: 3.9, label: "What it shows",
        lines: [
          "XGBoost best again (mean ρ 0.483) — a third independent confirmation",
          "The 6,559 sparse columns add nothing: 0.481 dense vs 0.483 combined",
        ],
      },
      {
        x: 8.9, y: 6.0, w: 3.9, label: "Conclusion",
        lines: [
          "Predicting a new GFP (ρ≈0.48) is about as hard as predicting a new " +
          "site in a known one (ρ≈0.44) — not dramatically harder.",
        ],
      },
    ],
    notes: "For a held-out protein every one of its position columns is zero " +
      "throughout training, so 99% of the feature width is dead weight here and " +
      "the 66 transferable descriptors carry all the signal. avGFP is easiest to " +
      "predict from the others, consistent with the step-1 finding that an " +
      "amacGFP variant converged to an exactly avGFP sequence.",
  });
}

/* ---------- 13. XGBoost retuning attempt (negative result) ------------- */
{
  const s = pres.addSlide();
  figureSlide(s, {
    title: "Step 8 — retuning XGBoost for extrapolation: no gain found",
    image: fig("xgb_tune.png"),
    imgX: 0.55, imgY: 1.05, imgW: 12.2, imgH: 12.2 / 3.12,
    cols: [
      {
        x: 0.5, y: 5.2, w: 3.9, label: "Objective",
        lines: [
          "Every XGBoost config so far was picked on the random split — the " +
          "regime it's worst at. Retune against the extrapolation objective " +
          "instead, dense-only per step 7.",
        ],
      },
      {
        x: 4.7, y: 5.2, w: 3.9, label: "What it shows",
        lines: [
          "A cheap 2-fold screen favoured shallower trees (ρ 0.461 vs ≈0.42)",
          "That gain vanished when checked on all 5 folds: best candidate " +
          "0.4393 vs default 0.4377, Δ=+0.0015, 95% CI [−0.015, +0.018]",
        ],
      },
      {
        x: 8.9, y: 5.2, w: 3.9, label: "Conclusion",
        lines: [
          "The default config is already about as good as anything nearby — " +
          "no free lunch in these hyperparameters.",
          "Confirms (again) the sparse block can be dropped: dense-only ρ " +
          "matches step 5's combined-feature result exactly (0.438 vs 0.441).",
        ],
      },
    ],
    notes: "A 2-fold screen is noisy enough to manufacture an apparent win; the " +
      "full 5-fold, bootstrapped check is what actually decides it. This is the " +
      "same lesson as step 4d to step 5 — small-sample extrapolation checks " +
      "overstate differences. Worth remembering before trusting the next quick result.",
  });
}

/* ---------- 14. step 9: the alignment ---------------------------------- */
{
  const s = pres.addSlide();
  slideTitle(s, "Step 9 — importing information from outside the assay");

  s.addText(
    "Every model plateaus at ρ ≈ 0.44 on unseen sites because nothing in the features " +
    "describes a specific position. BLOSUM62 knows how often Leu replaces Ile in general; " +
    "it does not know position 66 is the chromophore tyrosine. ESM-2 was the intended " +
    "source and is unreachable offline — so instead: the four backbones are homologous " +
    "GFPs from four organisms, and their differences are themselves an evolutionary record.",
    {
      x: 0.5, y: 1.1, w: 12.3, h: 1.0, isTextBox: true,
      fontFace: BODY, fontSize: 14, color: INK, margin: 0,
    }
  );

  const cards = [
    { value: "255", label: "alignment columns (Needleman–Wunsch, affine gaps)" },
    { value: "11", label: "new features per variant" },
    { value: "0", label: "network access required" },
  ];
  cards.forEach((c, i) =>
    statCard(s, { x: 0.5 + i * 4.18, y: 2.4, w: 3.95, h: 1.45, ...c })
  );

  sectionBlock(s, {
    x: 0.5, y: 4.25, w: 5.9, label: "Two independent validations",
    lines: [
      "The chromophore aligns — avGFP Tyr66-Gly67 lands in one column in all four",
      "Identities match the known phylogeny: amacGFP–avGFP 83%, cgreGFP ~45%, " +
      "ppluGFP2 the outlier at 18–25%",
    ],
  });
  sectionBlock(s, {
    x: 6.9, y: 4.25, w: 5.9, label: "What the features ask",
    lines: [
      "Is this site fixed across the homologs?",
      "Has nature already made this exact substitution somewhere?",
      "evo_score = mean BLOSUM(mut, homologs) − mean BLOSUM(wt, homologs)",
    ],
  });
  sectionBlock(s, {
    x: 0.5, y: 6.05, w: 12.3, label: "Leak-free by construction",
    lines: [
      "The features derive only from the four wild-type sequences — no brightness " +
      "measurement enters them, so they are constant per (backbone, position, mutant) " +
      "and cannot leak a label across a cross-validation fold.",
    ],
  });
  s.addNotes("Position 65 reads S, not T — correctly reflecting that this dataset's " +
    "reference is the F64L parent, not the S65T-containing EGFP lineage. A small " +
    "detail that independently confirms the wild-type derivation from step 6.");
}

/* ---------- 15. step 9 result ------------------------------------------ */
{
  const s = pres.addSlide();
  figureSlide(s, {
    title: "Step 9 — the plateau breaks",
    image: fig("evo.png"),
    imgX: 0.5, imgY: 1.05, imgW: 12.3, imgH: 12.3 / 3.02,
    cols: [
      {
        x: 0.5, y: 5.35, w: 3.9, label: "Objective",
        lines: [
          "Does site-specific evolutionary information beat ρ ≈ 0.44, where " +
          "eight model families and a hyperparameter search could not?",
        ],
      },
      {
        x: 4.7, y: 5.35, w: 3.9, label: "What it shows",
        lines: [
          "Single mutants 0.438 → 0.482; Δρ = +0.044, 95% CI [+0.028, +0.060], " +
          "P(Δ>0) = 1.000",
          "evo_score_sum is the highest-gain feature in the whole model",
          "ppluGFP2 alone gets worse — the 18–25% identity outlier",
        ],
      },
      {
        x: 8.9, y: 5.35, w: 3.9, label: "Conclusion",
        lines: [
          "The first change in the project to move the extrapolation number — " +
          "and it came from new information, not a better model.",
        ],
      },
    ],
    notes: "Steps 4 to 8 reshuffled models and hyperparameters within one fixed view " +
      "of a mutation and never beat 0.44. Adding genuine site-specific information " +
      "beat it immediately, from eleven features computed off four sequences.",
  });
}

/* ---------- 16. step 9 on a new protein -------------------------------- */
{
  const s = pres.addSlide();
  slideTitle(s, "Step 9d — and on an entirely new protein");

  s.addText(
    "Leave-one-backbone-out, same protocol as step 7. Fair by construction: for a " +
    "held-out backbone the evolutionary features come from the other three wild-type " +
    "sequences — if you are engineering a new FP you know its sequence, and no " +
    "brightness measurement from the held-out protein is used.",
    {
      x: 0.5, y: 1.1, w: 12.3, h: 0.8, isTextBox: true,
      fontFace: BODY, fontSize: 14, color: INK, margin: 0,
    }
  );

  const rows = [
    ["Features", "amacGFP", "avGFP", "cgreGFP", "ppluGFP2", "mean"],
    ["dense (66)", "0.421", "0.623", "0.539", "0.339", "0.481"],
    ["dense + evo (77)", "0.485", "0.675", "0.618", "0.317", "0.524"],
    ["Δ", "+0.064", "+0.052", "+0.079", "−0.022", "+0.043"],
  ];
  s.addTable(
    rows.map((r, ri) =>
      r.map((c, ci) => ({
        text: c,
        options: {
          bold: ri === 0 || ci === 0 || ri === 3 || ci === 5,
          color: ri === 0 ? WHITE : (ri === 3 && ci === 4 ? "A03030" : INK),
          fill: { color: ri === 0 ? FOREST : ri === 3 ? "E2F3D9" : WHITE },
          fontSize: 14, fontFace: BODY,
          align: ci === 0 ? "left" : "center",
        },
      }))
    ),
    { x: 0.5, y: 2.1, w: 12.3, colW: [3.3, 1.8, 1.8, 1.8, 1.8, 1.8],
      rowH: 0.5, border: { pt: 0 } }
  );

  sectionBlock(s, {
    x: 0.5, y: 4.3, w: 5.9, label: "The exception is the informative part",
    lines: [
      "ppluGFP2 degrades in both experiments — and it is the phylogenetic outlier " +
      "at 18–25% identity, where aligned columns carry little real constraint",
      "cgreGFP gains most, and it is the one with two relatives at ~45%",
    ],
  });
  sectionBlock(s, {
    x: 6.9, y: 4.3, w: 5.9, label: "Why this argues FOR ESM-2, not against it",
    lines: [
      "The benefit tracks having relatives at intermediate evolutionary distance",
      "If three sequences are worth +0.044, the millions behind a language model " +
      "should be worth more — and it would have no ppluGFP2 blind spot",
    ],
  });
  s.addNotes("cgreGFP sits at the sweet spot: close enough to align confidently, " +
    "distant enough that a shared residue means evolution actually held it fixed.");
}


/* ---------- 17. step 10 — why maximising the model fails ---------------- */
{
  const s = pres.addSlide();
  slideTitle(s, "Step 10 — the obvious design loop does not work");

  s.addText(
    "The project's second half is to propose bright sequences. Before maximising the " +
    "step-9 model over candidates, its out-of-fold predictions were decomposed by " +
    "brightness stratum — and the ρ ≈ 0.45 turns out to be almost entirely " +
    "dead-vs-alive separation, not brightness ranking.",
    { x: 0.5, y: 1.1, w: 12.3, h: 0.8, isTextBox: true,
      fontFace: BODY, fontSize: 14, color: INK, margin: 0 }
  );

  const rows = [
    ["Subset (unseen positions)", "n", "Spearman ρ"],
    ["all novel variants", "16,017", "0.455"],
    ["functional  (y ≥ 0.5)", "12,141", "0.247"],
    ["y ≥ 0.8", "8,734", "0.130"],
    ["y ≥ 1.0", "2,698", "−0.146"],
  ];
  s.addTable(
    rows.map((r, ri) =>
      r.map((c, ci) => ({
        text: c,
        options: {
          bold: ri === 0 || ci === 0 || ri === 4,
          color: ri === 0 ? WHITE : (ri === 4 && ci === 2 ? "A03030" : INK),
          fill: { color: ri === 0 ? FOREST : ri === 4 ? "F7E4E4" : WHITE },
          fontSize: 14, fontFace: BODY,
          align: ci === 0 ? "left" : "center",
        },
      }))
    ),
    { x: 0.5, y: 2.1, w: 7.0, colW: [3.6, 1.7, 1.7], rowH: 0.48, border: { pt: 0 } }
  );

  statCard(s, { x: 7.9, y: 2.1, w: 2.35, h: 1.15, value: "0.778",
    label: "AUC for 'is this variant alive' — what it really predicts" });
  statCard(s, { x: 10.45, y: 2.1, w: 2.35, h: 1.15, value: "< 1×",
    label: "enrichment at top 20 for y ≥ 1.2× WT", tint: "F7E4E4" });

  sectionBlock(s, {
    x: 7.9, y: 3.6, w: 4.9, label: "Compressed, too",
    lines: [
      "Predictions cap at 1.29 while truth reaches 2.48",
      "Variants at y ≥ 1.2 sit at median predicted rank 6,452 of 16,017",
    ],
  });

  sectionBlock(s, {
    x: 0.5, y: 4.45, w: 7.0, label: "Step 10a — and it is not fixable by splitting it",
    lines: [
      "Dedicated classifier: AUC 0.778 → 0.791",
      "Regressor on functional rows only: ρ 0.247 → 0.262",
      "Precision for y ≥ 1.2 stays at base rate — open item #3 answered",
    ],
  });

  s.addNotes("At never-assayed positions the model is a foldability filter. " +
    "Maximising it in the bright regime optimises a quantity uncorrelated with truth.");
}

/* ---------- 18. step 10 — design by recombination ----------------------- */
{
  const s = pres.addSlide();
  slideTitle(s, "Step 10 — so design by recombination, not by invention");

  s.addText(
    "It is not a data ceiling. On the random split the bright end is sharply " +
    "predictable (precision@20 for y ≥ 1.2 = 1.00 against a 3.0% base rate). The signal " +
    "is real; what fails is transfer. So propose novel COMBINATIONS of substitutions " +
    "each already characterised in that library — the regime where the model works.",
    { x: 0.5, y: 1.1, w: 12.3, h: 0.9, isTextBox: true,
      fontFace: BODY, fontSize: 14, color: INK, margin: 0 }
  );

  const rows = [
    ["Precision @ top 20", "base rate", "dense+evo (77)", "dense+evo+sparse (6,636)"],
    ["y ≥ 1.0× WT", "0.111", "1.00", "0.95"],
    ["y ≥ 1.2× WT", "0.030", "0.95", "0.80"],
    ["y ≥ 1.5× WT", "0.005", "0.15", "0.45"],
  ];
  s.addTable(
    rows.map((r, ri) =>
      r.map((c, ci) => ({
        text: c,
        options: {
          bold: ri === 0 || ci === 0 || (ri === 3 && ci === 3),
          color: ri === 0 ? WHITE : INK,
          fill: { color: ri === 0 ? FOREST : (ri === 3 ? "E2F3D9" : WHITE) },
          fontSize: 14, fontFace: BODY,
          align: ci === 0 ? "left" : "center",
        },
      }))
    ),
    { x: 0.5, y: 2.2, w: 12.3, colW: [3.3, 2.2, 3.2, 3.6], rowH: 0.5,
      border: { pt: 0 } }
  );

  sectionBlock(s, {
    x: 0.5, y: 4.3, w: 5.9, label: "A reversal worth recording",
    lines: [
      "The sparse block was retired in steps 4/7/8 as dead weight — correctly, for EXTRAPOLATION",
      "Recombination is the interpolation regime, and there it is exactly the memorised per-substitution knowledge the task exploits",
    ],
  });
  sectionBlock(s, {
    x: 6.9, y: 4.3, w: 5.9, label: "Tested to this project's standard",
    lines: [
      "Paired bootstrap, precision@20 at y ≥ 1.5: sparse − dense = +0.372, 95% CI [+0.100, +0.650], P(>0) = 0.998",
      "Evaluated on 13,581 held-out novel combinations",
    ],
  });
  s.addNotes("State the regime before stating which features matter. Finding 4 " +
    "(sparse is dead weight) and this result are both true, of different regimes.");
}

/* ---------- 19. step 10 — the component constraint ---------------------- */
{
  const s = pres.addSlide();
  slideTitle(s, "Step 10 — screen the components, then stack");

  s.addText(
    "An unconstrained beam search returned 6-mutation designs whose predictions ROSE " +
    "with depth while the empirical base rate falls ~28× — the search exploiting model " +
    "optimism. The matched population resolves it: avGFP multi-mutants where every " +
    "component was assayed alone.",
    { x: 0.5, y: 1.1, w: 12.3, h: 0.9, isTextBox: true,
      fontFace: BODY, fontSize: 14, color: INK, margin: 0 }
  );

  const rows = [
    ["Mean brightness (fold WT)", "2 mut", "3 mut", "4 mut", "5 mut", "6 mut"],
    ["every component ≥ 1.0× WT", "1.078", "1.103", "1.074", "—", "—"],
    ["every component ≥ 0.9× WT", "0.991", "0.982", "0.944", "0.937", "0.914"],
    ["unfiltered", "0.701", "0.545", "0.365", "0.217", "0.118"],
  ];
  s.addTable(
    rows.map((r, ri) =>
      r.map((c, ci) => ({
        text: c,
        options: {
          bold: ri === 0 || ci === 0,
          color: ri === 0 ? WHITE : (ri === 3 ? "A03030" : INK),
          fill: { color: ri === 0 ? FOREST : (ri === 3 ? "F7E4E4" : "E2F3D9") },
          fontSize: 14, fontFace: BODY,
          align: ci === 0 ? "left" : "center",
        },
      }))
    ),
    { x: 0.5, y: 2.2, w: 12.3, colW: [4.3, 1.6, 1.6, 1.6, 1.6, 1.6], rowH: 0.5,
      border: { pt: 0 } }
  );

  sectionBlock(s, {
    x: 0.5, y: 4.3, w: 5.9, label: "The finding",
    lines: [
      "Brightness collapses with mutation count because deleterious mutations accumulate — not because of stacking as such",
      "Conditioned on good components it is roughly preserved to 5–6 mutations",
    ],
  });
  sectionBlock(s, {
    x: 6.9, y: 4.3, w: 5.9, label: "So the loop is constrained",
    lines: [
      "Extension pool: substitutions individually MEASURED at ≥ 0.95× WT — 80 of 1,778 characterised for avGFP",
      "Where a measurement exists it beats the oracle for a component: K158G measures 2.48, is predicted 0.917",
    ],
  });
}

/* ---------- 20. step 10 — the panel ------------------------------------ */
{
  const s = pres.addSlide();
  slideTitle(s, "Step 10 — the deliverable, and its positive control");

  sectionBlock(s, {
    x: 0.5, y: 1.15, w: 5.9, label: "It rediscovers superfolder GFP",
    lines: [
      "No literature input enters the pipeline at any point",
      "Ranking all 1,778 characterised avGFP substitutions puts V163A 1st (cycle-3), Y39N 2nd, I171V 3rd, Y145F 50th — all superfolder mutations",
      "The top designs are dominated by that same set plus position 105",
    ],
  });

  sectionBlock(s, {
    x: 6.9, y: 1.15, w: 5.9, label: "What is on the plate",
    lines: [
      "23 avGFP variants: 5 designs at each of 2, 3, 4, 5 mutations, plus 3 single-mutant controls",
      "Every component individually measured ≥ 0.95× WT",
      "None of the top 50 designs exist in the library — all genuinely novel combinations",
    ],
  });

  statCard(s, { x: 0.5, y: 4.25, w: 2.9, h: 1.3, value: "~19 / 20",
    label: "expected at ≥ 1.2× WT, from precision@20 in this regime", tint: "E2F3D9" });
  statCard(s, { x: 3.65, y: 4.25, w: 2.9, h: 1.3, value: "~9 / 20",
    label: "expected at ≥ 1.5× WT (base rate would give 0.1)", tint: "E2F3D9" });
  statCard(s, { x: 6.8, y: 4.25, w: 2.9, h: 1.3, value: "31×",
    label: "enrichment over picking at random from the same pool" });
  statCard(s, { x: 9.95, y: 4.25, w: 2.85, h: 1.3, value: "0.998",
    label: "P(sparse beats dense) — the oracle choice, bootstrapped" });

  s.addText(
    "Measured results feed straight back as new characterised substitutions, widening " +
    "the recombination pool. That is the loop that compounds.",
    { x: 0.5, y: 5.75, w: 12.3, h: 0.5, isTextBox: true,
      fontFace: BODY, fontSize: 13, italic: true, color: MUTED, margin: 0 }
  );
}

/* ---------- 21. step 10 figure ----------------------------------------- */
{
  const s = pres.addSlide();
  figureSlide(s, {
    title: "Step 10 — the whole argument in four panels",
    image: fig("design.png"), imgX: 0.5, imgY: 1.05, imgW: 8.3, imgH: 5.84,
    cols: [
      { x: 9.1, y: 1.15, w: 3.7, label: "(a) and (b)",
        lines: [
          "At unseen positions the model ranks functionality, not brightness",
          "In the recombination regime it ranks brightness well — and the retired sparse block finds the brightest",
        ] },
      { x: 9.1, y: 3.75, w: 3.7, label: "(c) and (d)",
        lines: [
          "Stacking only destroys brightness when components are unscreened",
          "The proposed panel, with the empirical hit rate for each depth along the top",
        ] },
    ],
    notes: "Generated by make_design_report.py from design_shortlist.csv and design_regime_benchmark.csv.",
  });
}


/* ---------- 22. step 11 — ESM-2 unblocked --------------------------------- */
{
  const s = pres.addSlide();
  slideTitle(s, "Step 11 — ESM-2, unblocked");

  s.addText(
    "huggingface.co turned out reachable from a later session even though the original " +
    "blocked hosts stayed blocked. esm_scores.py gained a HuggingFace backend; the " +
    "masked-marginals formulation and wild-type checks are unchanged. Two different " +
    "outputs of the same weights were tested, and they gave two different answers.",
    { x: 0.5, y: 1.1, w: 12.3, h: 0.8, isTextBox: true,
      fontFace: BODY, fontSize: 14, color: INK, margin: 0 }
  );

  const rows = [
    ["Feature set (single mutants)", "ρ, unseen positions"],
    ["dense (66)", "0.438"],
    ["dense + evo (77) — step 9 winner", "0.478"],
    ["dense + ESM-2 650M zero-shot score (81)", "0.487"],
    ["dense + evo + ESM-2 650M embedding (109)", "0.519"],
  ];
  s.addTable(
    rows.map((r, ri) =>
      r.map((c, ci) => ({
        text: c,
        options: {
          bold: ri === 0 || ri === 4,
          color: ri === 0 ? WHITE : INK,
          fill: { color: ri === 0 ? FOREST : (ri === 4 ? "E2F3D9" : WHITE) },
          fontSize: 14, fontFace: BODY,
          align: ci === 0 ? "left" : "center",
        },
      }))
    ),
    { x: 0.5, y: 2.1, w: 9.2, colW: [6.6, 2.6], rowH: 0.48, border: { pt: 0 } }
  );

  statCard(s, { x: 10.0, y: 2.1, w: 2.8, h: 1.0, value: "+0.040",
    label: "embed vs +evo, 95% CI [+0.025, +0.055]", tint: "E2F3D9" });

  sectionBlock(s, {
    x: 0.5, y: 4.3, w: 5.9, label: "A log-odds score is not an embedding",
    lines: [
      "Zero-shot: how likely is this substitution — significantly WORSE than the evo features alone (−0.035)",
      "Embedding: what kind of site is this — structural context a scalar collapses away",
      "Stacking the score on top of the embedding is slightly worse: the embedding subsumes it",
    ],
  });
  sectionBlock(s, {
    x: 6.9, y: 4.3, w: 5.9, label: "Cheap and fold-safe, like the evo features",
    lines: [
      "4 forward passes — one per wild-type, not per variant",
      "PCA to 16 dims, fitted on the 4 wild-type residues only — no labels",
      "Width swept 8/16/32/64: flat, every CI vs 16 spans zero",
    ],
  });
}

/* ---------- 23. step 11 — the ppluGFP2 question resolved ------------------ */
{
  const s = pres.addSlide();
  slideTitle(s, "Step 11 — and it fixes exactly the blind spot step 9 predicted");

  s.addText(
    "Step 9d found ppluGFP2 — the phylogenetic outlier at 18–25% identity — was the one " +
    "backbone the evo features actively hurt, and argued a language model would not " +
    "share that blind spot since it has seen sequences the other three backbones haven't. " +
    "Leave-one-backbone-out confirms it.",
    { x: 0.5, y: 1.1, w: 12.3, h: 0.85, isTextBox: true,
      fontFace: BODY, fontSize: 14, color: INK, margin: 0 }
  );

  const rows = [
    ["Features", "amacGFP", "avGFP", "cgreGFP", "ppluGFP2", "mean"],
    ["dense + evo (step 9 winner)", "0.486", "0.666", "0.613", "0.314", "0.520"],
    ["dense + evo + emb", "0.487", "0.675", "0.657", "0.393", "0.553"],
    ["Δ", "+0.001", "+0.009", "+0.044", "+0.079", "+0.033"],
  ];
  s.addTable(
    rows.map((r, ri) =>
      r.map((c, ci) => ({
        text: c,
        options: {
          bold: ri === 0 || ci === 0 || ri === 3 || ci === 5,
          color: ri === 0 ? WHITE : (ri === 3 && ci === 4 ? "1F7A3F" : INK),
          fill: { color: ri === 0 ? FOREST : ri === 3 ? "E2F3D9" : WHITE },
          fontSize: 14, fontFace: BODY,
          align: ci === 0 ? "left" : "center",
        },
      }))
    ),
    { x: 0.5, y: 2.15, w: 12.3, colW: [3.3, 1.8, 1.8, 1.8, 1.8, 1.8],
      rowH: 0.5, border: { pt: 0 } }
  );

  sectionBlock(s, {
    x: 0.5, y: 4.3, w: 11.3, label: "The headline model changes",
    lines: [
      "dense+evo+emb beats dense+evo on BOTH extrapolation benchmarks (position CV and LOBO), each confirmed by bootstrap",
      "It is now the recommended feature set for any never-assayed-position or new-protein task, replacing step 9's dense+evo",
      "Step 9's prediction was right about the mechanism (no ppluGFP2 blind spot) and wrong about the formulation — embeddings, not log-odds scores",
    ],
  });
}

/* ---------- 24. step 12 — re-tuned and stress-tested ----------------------- */
{
  const s = pres.addSlide();
  slideTitle(s, "Step 12 — re-tuning the design oracle, and a harder check");

  sectionBlock(s, {
    x: 0.5, y: 1.15, w: 5.9, label: "Re-tuning — negative, correctly",
    lines: [
      "The design oracle's config was inherited from step 8, tuned for a different regime",
      "10-config grid × with/without ESM embeddings, selected on validation, confirmed once on test",
      "tuned − incumbent, mean brightness of top 20: +0.060, 95% CI [−0.074, +0.223] — fails the project's own bar, kept unchanged",
    ],
  });
  sectionBlock(s, {
    x: 6.9, y: 1.15, w: 5.9, label: "Near-neighbour stress test",
    lines: [
      "Design-regime rows split by whether any training variant shares all but one of their substitutions",
      "52% have such a neighbour to copy from; 48% do not, and those give the honest lower bound",
    ],
  });

  const rows = [
    ["Precision @ top 20", "all rows", "near neighbour", "no neighbour"],
    ["y ≥ 1.0× WT", "0.95", "1.00", "0.80"],
    ["y ≥ 1.2× WT", "0.80", "0.90", "0.65"],
    ["y ≥ 1.5× WT", "0.45", "0.35", "0.40"],
  ];
  s.addTable(
    rows.map((r, ri) =>
      r.map((c, ci) => ({
        text: c,
        options: {
          bold: ri === 0 || ci === 0 || ri === 3,
          color: ri === 0 ? WHITE : INK,
          fill: { color: ri === 0 ? FOREST : (ri === 3 ? "E2F3D9" : WHITE) },
          fontSize: 14, fontFace: BODY,
          align: ci === 0 ? "left" : "center",
        },
      }))
    ),
    { x: 0.5, y: 3.75, w: 8.5, colW: [3.1, 1.8, 1.8, 1.8], rowH: 0.42,
      border: { pt: 0 } }
  );

  sectionBlock(s, {
    x: 9.3, y: 3.8, w: 3.5, label: "Quote the range",
    lines: [
      "0.65–0.90 at ≥1.2×, ~0.35–0.45 at ≥1.5×. Even the low end is ~21× over base rate",
    ],
  });

  sectionBlock(s, {
    x: 0.5, y: 5.65, w: 11.3, label: "Two small items closed",
    lines: [
      "Log-transform: tested, not assumed — refitting on log1p(y) changes the trees, and it is worse (ρ 0.818 vs 0.823; 0.437 vs 0.467 among functional). Keep the raw target",
      "Mammalian dataset (13 rows, 8 engineered FPs, 7 assay methods): excluded — not DMS variants of the four backbones, so it cannot enter training",
    ],
  });
}

/* ---------- 25. step 11-12 figure ------------------------------------------ */
{
  const s = pres.addSlide();
  figureSlide(s, {
    title: "Steps 11–13 in four panels",
    image: fig("esm.png"), imgX: 0.5, imgY: 1.05, imgW: 8.3, imgH: 5.84,
    cols: [
      { x: 9.1, y: 1.15, w: 3.7, label: "(a) and (b)",
        lines: [
          "Zero-shot scores add little; embeddings add what the evo features added, again",
          "ppluGFP2 — the outlier the evo features hurt — is the one embeddings fix",
        ] },
      { x: 9.1, y: 3.75, w: 3.7, label: "(c) and (d)",
        lines: [
          "Embedding width doesn't matter — 8 through 64 dims are statistically tied",
          "Half the designs have no near neighbour to copy — those set the honest lower bound",
        ] },
    ],
    notes: "Generated by make_esm_report.py from esm_benchmark, lobo_esm, esm_dims_sweep and design_stress_test outputs.",
  });
}

/* ---------- 26. conclusions -------------------------------------------- */
{
  const s = pres.addSlide();
  s.background = { color: DARK };
  slideTitle(s, "Where this leaves us", WHITE);

  const picks = [
    ["MLP", "Ranking recombinations of characterised mutations — best overall (ρ 0.911 / R² 0.870)"],
    ["Ridge", "Fast interpretable workhorse — 6 s vs 9 min, and each coefficient is a mutation's effect"],
    ["XGBoost", "Any transfer task — best at unseen positions and unseen proteins. Confirmed three ways"],
    ["+ evo + ESM-2 emb", "Current best for extrapolation: 0.438 → 0.519 unseen sites, 0.481 → 0.553 unseen proteins"],
    ["Design", "Recombination only — 23-variant panel, 13–18 of 20 expected at ≥1.2× WT depending on near-neighbour support. Rediscovers superfolder GFP unprompted"],
    ["Dropped", "Lasso, ElasticNet, RandomForest, kernel SVR, ESM-2 zero-shot scores — all dominated"],
  ];
  picks.forEach(([name, why], i) => {
    const y = 1.05 + i * 0.63;
    const muted = i === picks.length - 1;      // the "Dropped" row
    // name and description share one y, height and vertical centring, so the
    // two different font sizes still sit on a common line
    s.addShape(pres.ShapeType.ellipse, {
      x: 0.55, y: y + 0.15, w: 0.16, h: 0.16,
      fill: { color: muted ? "6B7F6D" : GLOW },
    });
    s.addText(name, {
      x: 0.95, y: y, w: 1.9, h: 0.46, isTextBox: true, valign: "middle",
      fontFace: HEAD, fontSize: 17, bold: true,
      color: muted ? "9FB3A1" : GLOW, margin: 0,
    });
    s.addText(why, {
      x: 2.95, y: y, w: 9.8, h: 0.46, isTextBox: true, valign: "middle",
      fontFace: BODY, fontSize: 13.5, color: muted ? "9FB3A1" : "E8F0E8", margin: 0,
    });
  });

  s.addText("NEXT", {
    x: 0.55, y: 4.85, w: 12.3, h: 0.3, isTextBox: true,
    fontFace: BODY, fontSize: 12, bold: true, color: GLOW, charSpacing: 1.2, margin: 0,
  });
  s.addText(
    [
      { text: "Every item on this slide in earlier versions of the deck is now done: ESM-2 ran (step 11), the design oracle was re-tuned and stress-tested (step 12), the two small open items closed", options: { bullet: true, breakLine: true } },
      { text: "Keep using the panel. Measured results return as new characterised substitutions, which widens the recombination pool — the one loop here that compounds", options: { bullet: true, breakLine: true } },
      { text: "Do NOT extend the design loop to never-assayed positions: step 10 shows enrichment there is below 1. Design by recombination is validated; design by invention is not", options: { bullet: true, breakLine: true } },
      { text: "Genuinely optional and left that way: a full-scale MLP tune (interpolation ceiling only) and an FPbase harvest", options: { bullet: true } },
    ],
    {
      x: 0.55, y: 5.2, w: 12.2, h: 1.9, isTextBox: true,
      fontFace: BODY, fontSize: 13.5, color: "E8F0E8", margin: 0, paraSpaceAfter: 5,
    }
  );
}

pres.writeFile({ fileName: path.join(OUT, "GFP_ML_results.pptx") })
  .then((f) => console.log("wrote " + f));
