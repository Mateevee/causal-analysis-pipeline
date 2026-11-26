# Protocol — R1-4e: pipeline effects against real interventions in the light tunnel (2026-11-21)

## 1. Request and question
- **R1, point 4:** "the validation is on simulated data from the development suite". Up to now, no effect has been
  checked on real data.
- **Question:** on the observational data of the light tunnel (random walks `lt_walks_v1`), does the effect θ̂ that the
  pipeline estimates for "colour actuator → sensor" have the **sign** and the **order of magnitude** of the effect measured by
  **randomized intervention** on the same apparatus (`lt_interventions_standard_v1`)?
- **Scope:** it is an offline analysis of already recorded outputs. No method runs, and nothing in the pipeline changes.

## 2. What was already seen before freezing (declared)
- The structure of the files: columns, number of rows and the **design** of the interventions, i.e. the levels of the
  manipulated columns. In the reference, `red`, `green` and `blue` are uniform on 0–85, `pol_1` and `pol_2` on −15–24, and
  the `l_*` on 0–170; the other actuators are constant. In `*_mid` the actuator is in 86–170, in `*_strong` in
  171–255, and the other colours stay in 0–85.
- In the `original.csv` of `patch_tau10`, only the structure: one row with lag ≥ 1 per pair and, in some scenes, one more
  contemporaneous row (lag 0); the list of entities of each scene; and the `results.csv` of `lt_color_mix` has 4 links.
- **Not seen:** no sensor value in the intervention files and no θ̂, standard error or interval of the
  pipeline in these scenes.

## 3. Data
- **θ̂ of the pipeline** (`patch_tau10`, the submitted method; τ_max = 10):
  - `real_causal_chamber_lt_color_mix`: R1-2c, execution on the runner
    (`_outra_maquina/r1_2c_completo_2026-11-18/.../run/`);
  - `real_causal_chamber_lt_actuators_white` and `real_causal_chamber_lt_smooth_polarizers`: runs of the paper
    (`common.folder`, currently `Results_Extra/Run_20261009_causal_chamber/`).
- **Pairs evaluated:** source ∈ {red, green, blue} with variance in the scene, and target any other variable of the scene that
  is a **measured sensor** (not an actuator). Per pair, the row of `original.csv` with lag ≥ 1 (Stage 6); the
  contemporaneous rows (lag 0) are left out because they have no Stage 6 θ.
- **Units:** the pipeline estimates θ̂ on the z scale (Stage 2). The conversion to the raw unit is
  θ̂_raw = θ̂ · sd(Y) / sd(X), with the standard deviations from the `source.csv` of the scene itself (the same 500 observations).
  The interval is converted the same way.
- **Reference effect (interventional truth), per colour c:** the data of `uniform_reference.csv` (c on 0–85),
  `uniform_c_mid.csv` (86–170) and `uniform_c_strong.csv` (171–255) are stacked, and the sensor Y is regressed by OLS
  on all actuators that vary in the stacked set (red, green, blue, pol_1, pol_2, l_11 to l_32). Since the
  actuators are drawn independently, the coefficient β_c is the average linear causal effect of c on Y over
  0–255. HC3 standard error. "Significant β" = |β/SE| > 1.96.
- **Secondary check of the truth:** the contrast between levels, (mean of Y in c_strong − mean in c_mid) /
  (mean of c in c_strong − mean in c_mid). Reported alongside β, with the sign agreement between the two.

## 4. Outcomes (fixed beforehand)
- **E1, primary: sign agreement.** Among the pairs with significant β, the fraction with sign(θ̂) = sign(β). Per scene
  and overall.
- **E2: ordering.** Spearman between θ̂_raw and β (signed), on the pairs with significant β.
- **E3: magnitude (descriptive).** Median and quartiles of θ̂_raw / β on the pairs with significant β. **Expectation
  declared beforehand:** attenuation (ratio < 1). The physical effect is instantaneous; θ̂ is a lagged effect (lag ≥ 1) with
  controls that include the lags of the target itself, and the paper already shows |θ̂| below the true value in the
  internal generator.
- **E4: interval (descriptive).** Fraction of the pairs with significant β in which the converted 95% CI of θ̂ contains β.
- **E5: false effect.** Among the pairs with non-significant β, the fraction in which the 95% CI of θ̂ excludes zero.
- **E6: reported.** The same E1 and E3 restricted to the pairs that the pipeline reported (`results.csv`).

## 5. Reading (fixed beforehand)
- **E1 ≥ 90%** (overall): "the sign of the effect estimated by the pipeline agrees with that of the real intervention in X of N
  pairs" supports "directional predictive influence" on real data.
- **70% ≤ E1 < 90%:** partial agreement; report the discordant pairs.
- **E1 < 70%:** the direction does not hold on real data; report it as a limitation, with the same prominence.
- **Magnitude:** whatever E3/E4, the text does not claim quantitative validity. The intervals remain
  "exploratory".
- **Caveats that always go along:**
  - a single apparatus, three scenes;
  - effects almost linear by construction (additive light);
  - lagged θ̂ against an instantaneous effect;
  - the walks and the interventions are different sessions of the same apparatus;
  - the truth is the average linear effect over 0–255.
- **Polarizer pairs** (pol_1, pol_2 → sensors) are left out: Malus's law (cos²) makes the average linear slope
  uninformative. If they appear, only descriptive.

## 6. Output
`Results_Revisao/r1_4e_intervencoes/` (`relatorio.md`, `por_par.csv`, `dados.json`). Script:
`revision/offline/r1_4e_interventions.py`, written after this freeze, which follows this document. Any
deviation is noted in `NOTAS.md`.
