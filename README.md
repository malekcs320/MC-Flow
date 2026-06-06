# MC-Flow District-Heating Forecasting

This repository contains the code, configurations, and evaluation scripts for reproducing the MC-Flow district-heating forecasting experiments.

## Visualization of the training architecture

<img width="911" height="368" alt="image" src="https://github.com/user-attachments/assets/3e03dd86-a7f0-4358-9d15-03bba3fd8694" />


## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD/src"
```

Weights & Biases is optional. Training runs without it by default; add `--wandb` to enable logging.

## Checkpoints

Pretrained paper checkpoints are available in the repository `checkpoints/` folder. Use those checkpoints directly for evaluation and figure reproduction, or retrain with the commands below to regenerate them.

For the horizon-specific scripts, arrange or symlink the checkpoints as:

```text
checkpoints/
├── h3/
│   └── checkpoint_best.pth
├── h5/
│   └── checkpoint_best.pth
└── h7/
    └── checkpoint_best.pth
```


## Data

The smart heat meter data should be downloaded from Zenodo:

- Dataset: [Three years of hourly data from 3021 smart heat meters installed in Danish residential buildings](https://zenodo.org/records/6563114)
- DOI: `10.5281/zenodo.6563114`
- Download file: `3_years_3021_smart_heat_meters_residential_denmark.zip`

We provide the time-aligned weather file for the same recordings period at data/weather.csv. 

The Zenodo archive already contains the preprocessed per-building files in the form `{building_id}.csv` and the metadata file `contextual_data.csv`. After downloading and extracting the archive, copy those files into the repository layout below, and add the weather file.

```text
MCFLOW/
├── contextual_data.csv
├── aalborg_2018_2020_merged.csv
├── buildings_all/
│   ├── 1.csv
│   ├── 2.csv
│   └── ...
├── checkpoints/
│   ├── h3/checkpoint_best.pth
│   ├── h5/checkpoint_best.pth
│   └── h7/checkpoint_best.pth
└── configs/
    ├── mcflow_h3.yaml
    ├── mcflow_h5.yaml
    └── mcflow_h7.yaml
```


The loader converts `time_rounded` to datetimes, computes daily demand as the non-negative difference of cumulative `energy_heat_kwh`, joins normalized temperature from `aalborg_2018_2020_merged.csv`, and applies the chronological 70/10/20 split.

## Training

```bash
python scripts/train_mcflow.py --config configs/mcflow_h3.yaml --output-dir checkpoints/h3 --seed 42
python scripts/train_mcflow.py --config configs/mcflow_h5.yaml --output-dir checkpoints/h5 --seed 42
python scripts/train_mcflow.py --config configs/mcflow_h7.yaml --output-dir checkpoints/h7 --seed 42
```

Each run writes:

- `checkpoint_latest.pth`
- `checkpoint_best.pth`

The best checkpoint is selected by validation  weighted CRPS.

## Evaluation

```bash
python scripts/evaluate_mcflow.py \
  --config configs/mcflow_h7.yaml \
  --checkpoint checkpoints/h7/checkpoint_best.pth \
  --split test \
  --samples-per-mode 10 \
  --seed 42 \
  --output outputs/metrics/mcflow_h7.json
```

The JSON includes the run-identifying hyperparameters and both methods:

- `rmse_mcflow`, `mae_mcflow`, `crps_mcflow`
- `rmse_stage1`, `mae_stage1`, `crps_stage1`

For MC-Flow, CRPS uses exact weights `pi_k / S` over `K * S` refined samples. For MC-Flow without CFM, CRPS uses the K macro anchors with weights `pi_k`. RMSE/MAE use the top-probability refined-mode empirical mean for MC-Flow and the top-probability anchor for the stage-1 ablation.

## Reproduce Tables

After the three best checkpoints exist at `checkpoints/h3`, `checkpoints/h5`, and `checkpoints/h7`:

```bash
bash scripts/reproduce_tables.sh
```

This writes:

```text
outputs/metrics/mcflow_h3.json
outputs/metrics/mcflow_h5.json
outputs/metrics/mcflow_h7.json
outputs/metrics/all_mcflow_results.csv
```

## Reproduce Figures

Calibration diagnostics:

```bash
python scripts/make_calibration.py \
  --config configs/mcflow_h7.yaml \
  --checkpoint checkpoints/h7/checkpoint_best.pth \
  --samples-per-mode 10 \
  --seed 42 \
  --output outputs/figures/calibration_metrics.pdf
```

Rolling qualitative forecast figure:

```bash
python scripts/make_rolling_forecast.py \
  --config configs/mcflow_h7.yaml \
  --checkpoint checkpoints/h7/checkpoint_best.pth \
  --num-windows 20 \
  --top-k 4 \
  --samples-per-mode 50 \
  --past-context-days 14 \
  --seed 42 \
  --output outputs/figures/forecast_figure_kwh.pdf
```
