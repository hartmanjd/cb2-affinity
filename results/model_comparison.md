# Model comparison

Saved baseline and tuned validation results. Each row identifies its dataset and whether it matches the newest preparation. Results remain visible until new runs complete. Reserved tests are unevaluated. MLP entries are means ± sample SD across seeds.

The [provenance workbook](../provenance/CB2_data_provenance.xlsx) is the supplied presentation snapshot. Current exact scores and predictions are under `provenance/models/`; this comparison refreshes independently.

| Model | Variant | Split | N | MAE | RMSE | R² | Dataset | Status |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| Dummy mean | baseline | random | 358 | 0.9725 | 1.1890 | -0.0015 | chembl_cb2_20260922T165756862384Z | current preparation |
| Dummy median | baseline | random | 358 | 0.9710 | 1.1883 | -0.0003 | chembl_cb2_20260922T165756862384Z | current preparation |
| Dummy mean | baseline | scaffold | 461 | 0.9560 | 1.1459 | -0.0016 | chembl_cb2_20260922T165756862384Z | current preparation |
| Dummy median | baseline | scaffold | 461 | 0.9558 | 1.1473 | -0.0041 | chembl_cb2_20260922T165756862384Z | current preparation |
| Dummy | tuned | random | 358 | 0.9725 | 1.1890 | -0.0015 | chembl_cb2_20260922T165756862384Z | current preparation |
| Dummy | tuned | scaffold | 461 | 0.9558 | 1.1473 | -0.0041 | chembl_cb2_20260922T165756862384Z | current preparation |
| Random forest | baseline | random | 358 | 0.5643 | 0.7215 | 0.6313 | chembl_cb2_20260922T165756862384Z | current preparation |
| Random forest | baseline | scaffold | 461 | 0.6927 | 0.8746 | 0.4166 | chembl_cb2_20260922T165756862384Z | current preparation |
| Random forest | tuned | random | 358 | 0.5605 | 0.7257 | 0.6269 | chembl_cb2_20260922T165756862384Z | current preparation |
| Random forest | tuned | scaffold | 461 | 0.6931 | 0.8808 | 0.4082 | chembl_cb2_20260922T165756862384Z | current preparation |
| XGBoost | baseline | random | 358 | 0.6449 | 0.8197 | 0.5240 | chembl_cb2_20260922T165756862384Z | current preparation |
| XGBoost | baseline | scaffold | 461 | 0.7349 | 0.9002 | 0.3819 | chembl_cb2_20260922T165756862384Z | current preparation |
| XGBoost | tuned | random | 358 | 0.6097 | 0.7833 | 0.5653 | chembl_cb2_20260922T165756862384Z | current preparation |
| XGBoost | tuned | scaffold | 461 | 0.7033 | 0.8681 | 0.4251 | chembl_cb2_20260922T165756862384Z | current preparation |
| Support vector regression | baseline | random | 358 | 0.5545 | 0.7237 | 0.6290 | chembl_cb2_20260922T165756862384Z | current preparation |
| Support vector regression | baseline | scaffold | 461 | 0.6596 | 0.8240 | 0.4821 | chembl_cb2_20260922T165756862384Z | current preparation |
| Support vector regression | tuned | random | 358 | 0.5225 | 0.6850 | 0.6677 | chembl_cb2_20260922T165756862384Z | current preparation |
| Support vector regression | tuned | scaffold | 461 | 0.6007 | 0.7641 | 0.5547 | chembl_cb2_20260922T165756862384Z | current preparation |
| Multilayer perceptron | baseline | random | 358 | 0.6310 ± 0.0128 | 0.8250 ± 0.0209 | 0.5176 ± 0.0245 | chembl_cb2_20260922T165756862384Z | current preparation |
| Multilayer perceptron | baseline | scaffold | 461 | 0.7312 ± 0.0246 | 0.9134 ± 0.0325 | 0.3629 ± 0.0447 | chembl_cb2_20260922T165756862384Z | current preparation |
| Multilayer perceptron | tuned | random | 358 | 0.6066 ± 0.0204 | 0.8017 ± 0.0242 | 0.5444 ± 0.0277 | chembl_cb2_20260922T165756862384Z | current preparation |
| Multilayer perceptron | tuned | scaffold | 461 | 0.7255 ± 0.0223 | 0.9081 ± 0.0295 | 0.3705 ± 0.0401 | chembl_cb2_20260922T165756862384Z | current preparation |
