## 1.2.0 / 2026-10-12 (fork: elenacandellone/GraphRole)

* [Fixed] `node_role_factor` / `role_feature_factor` (and hence `roles`, `role_percentage`) are the NMF factors of the selected model, not their MDL-quantized encoding; the encoded factors are available as `node_role_factor_encoded` / `role_feature_factor_encoded`, and the selected number of bits as `n_bits`
* [Added] `random_state` argument to `RoleExtractor` for a reproducible NMF initialization
* [Changed] model selection computes one NMF per number of roles and reuses it across bit depths
* [Fixed] default ReFeX aggregations work on pandas >= 2 (aggregations passed by name); dropped the `pandas < 2.0` pin

## 1.1.1 / 2023-10-08

* [Changed] updated dependency version requirements

## 1.1.0 / 2023-04-08

* [Added] optionally include node attributes as local features

## 1.0.3 / 2023-03-18

* [Changed] added Python 3.10 and Python 3.11 support and updated necessary dependencies

## 1.0.2 / 2021-06-26

* [Fixed] eliminated runtime warning when computing log of array containing zeros
* [Changed] added Python 3.9 support and updated necessary dependencies

## 1.0.1 / 2020-08-25

* [Changed] updated dependencies to latest versions
