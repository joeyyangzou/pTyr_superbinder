# Manuscript text draft

## Methods: model evaluation

The classification dataset was divided once into an 80% development set and
an untouched 20% test set using seed 2026. The regression dataset was sorted
by descending log(ratio), partitioned into consecutive blocks of five, and one
sequence per block was assigned to the independent test set; the other four
were assigned to the development set. Ten-fold outer cross-validation was conducted only
within the development set. In each outer fold, an inner validation subset was
drawn from the outer training partition and used for early stopping. After the
optimal epoch was selected, a newly initialized CNN was refitted on the full
outer training partition and evaluated on the held-out outer fold. For final
testing, epoch selection was performed within the development set, after which
a newly initialized model was trained on the complete development set and
evaluated once on the independent test set. Regression target transformations
were fitted using training data only. Test-set confidence intervals were
estimated using 2,000 stratified bootstrap replicates.

## Results: classification

The classifier achieved a mean AUROC of 0.971 +/- 0.004 across the ten
outer cross-validation folds conducted within the 80% development set. On the
untouched 20% test set, AUROC was 0.974 (95% CI, 0.968-0.978), AUPRC was 0.979 (95% CI, 0.975-0.982), F1 was
0.925 (95% CI, 0.915-0.934), and MCC was 0.853 (95% CI, 0.835-0.870). The Brier score was 0.057 (95% CI, 0.051-0.063), and ECE was
0.020 (95% CI, 0.016-0.030). Calibration was assessed using the Brier score, ECE, and a
reliability diagram. Final probabilities were Platt calibrated using only
development-set out-of-fold predictions. Ten independently initialized runs
and the corresponding deep-ensemble model-disagreement SD were also reported.

## Results: regression

Across the ten outer development-set folds, mean Pearson r was 0.868 +/-
0.017 and mean Spearman rho was 0.829 +/- 0.014.
On the untouched test set, Pearson r was 0.870 (95% CI, 0.849-0.889), Spearman rho was
0.825 (95% CI, 0.808-0.842), MAE was 0.206 (95% CI, 0.194-0.219), and RMSE was 0.303 (95% CI, 0.277-0.332). Residuals were
examined graphically as a function of the predicted value. Ten independently
initialized regression runs were summarized as mean +/- SD, and prediction SD
across the ten models was reported as deep-ensemble epistemic disagreement.

## Interpretation of split designs

The random 80:20 analysis evaluates interpolation within the mutational
library. Because related variants can occur across a random split, these
results should not be interpreted as direct evidence of extrapolation to
dissimilar sequence regions.
