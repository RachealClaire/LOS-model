# Model Development Report - Length of Hospital Stay
**Date:** 2026-03-23 14:53

## Methods
- **Models**: GradientBoosting, XGBoost, LightGBM, RandomForest, ElasticNet
- **Tuning**: RandomizedSearchCV (n_iter=40, cv=5-fold)
- **Target transform**: log1p (train only); expm1 on predictions
- **Pooling**: Rubin's Rules via 50 bootstrap datasets
- **df adjustment**: Barnard-Rubin (1999)
- **Internal validation**: Bootstrap optimism correction (n=200)

## Model Comparison
| Model | CV RMSE | Test RMSE | MAE | R2 |
|-------|---------|-----------|-----|----|
| XGBoost | 100.738 | 38.885 | 15.758 | 0.0389 |
| RandomForest | 100.943 | 39.150 | 16.022 | 0.0258 |
| LightGBM | 100.891 | 39.190 | 15.359 | 0.0238 |
| GradientBoosting | 100.715 | 39.260 | 15.875 | 0.0203 |
| ElasticNet | 101.083 | 39.468 | 15.534 | 0.0098 |
| Naive baseline | - | 41.186 | - | - |

## Best Model: XGBoost
| Set | RMSE | MAE | R2 |
|-----|------|-----|-----|
| Rubin OOB | 104.277 (42.928-165.625) | 26.255 | -0.0236 |
| Apparent  | 99.725 | 21.740 | 0.0726 |
| Corrected | 91.891 | - | -0.0397 |
| Test set  | 38.885 | 15.758 | 0.0389 |

## Top 10 Features
| Feature | Importance | 95% CI |
|---------|------------|--------|
| Resistance | 0.1832 | (-1.8275-2.1938) |
| ward | 0.1699 | (-1.8390-2.1788) |
| age | 0.1271 | (-1.8813-2.1356) |
| name_of_rrh | 0.1170 | (-1.8914-2.1254) |
| cormobid_condition | 0.0879 | (-1.9205-2.0963) |
| sex | 0.0852 | (-1.9232-2.0937) |
| antibiotics_prescribed | 0.0792 | (-1.9292-2.0875) |
| num_infection_sites | 0.0790 | (-1.9293-2.0874) |
| referral | 0.0714 | (-1.9370-2.0799) |

## References
1. Rubin DB (1987). Multiple Imputation for Nonresponse in Surveys.
2. Barnard J, Rubin DB (1999). Biometrika 86(4):948-955.
3. Harrell FE (2015). Regression Modeling Strategies. Springer.
4. Steyerberg EW (2019). Clinical Prediction Models. Springer.
