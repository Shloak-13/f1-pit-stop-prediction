param(
  [string]$Python = "python",
  [string]$Models = "lgbm,xgb,hist_gbdt,extra_trees,logistic"
)

$ErrorActionPreference = "Stop"

& $Python src/diagnostics.py --skip-adv
& $Python src/features.py
& $Python src/train.py --cv stratified --models $Models --tag s0
& $Python src/train.py --cv race --models $Models --tag race
