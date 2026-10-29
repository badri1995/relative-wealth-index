# Relative Wealth Index research

Selected notebooks from Badrinath Padmanabhan's work at IIT Delhi on socioeconomic measurement using census-derived targets and satellite-related predictors.

The released code covers creation of a PCA-based relative index, random forest regression against that index, and a separate historical prediction experiment using a six-neighbour feature model. It is a research source snapshot, not a complete standalone data product.

## Notebooks

| Notebook                                   | Purpose                                                                                                             |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------- |
| `wealth_index_ground_truth_creation.ipynb` | Aggregate census asset and housing fields, standardise features, derive a first-component index and wealth rankings |
| `rwi_rf_regression.ipynb`                  | Align index targets with supplied training/test village indices and fit a selected-feature random forest            |
| `rwi_prediction_2019_6neighbors.ipynb`     | Apply a separately supplied historical model and feature list to 2019 predictors                                    |

The prediction notebook expects its own six-neighbour model. It is **not** automatically the model produced by the regression notebook. Model files and feature inputs are intentionally absent.

Related public work on satellite indicator preparation is in [ADI_complete](https://github.com/badri1995/ADI_complete). That repository and these notebooks are separate workflow snapshots; they should not be assumed to share an identical feature schema.

## Setup and inputs

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

The first code cell of each notebook resolves `data/` and `outputs/` relative to the repository. Set `RESEARCH_DATA_DIR` and `RESEARCH_OUTPUT_DIR` before starting Jupyter to use other locations. Only the output directory is created automatically.

Required local inputs include:

| Stage                 | Files and expected structure                                                                                                                     |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| Target construction   | `India.csv` with the original numbered census columns, and `gt_2011.csv` with village identifiers                                                |
| Regression            | `dataset.pickle`, a mapping containing `X_train` and `X_test` DataFrames, plus `top_selected_feature_using_mean.pickle`                          |
| Target alignment      | `outputs/new_relative_wealth_index.pickle`, produced by the target notebook, with village identifiers as the index and an `RWI` column           |
| Historical prediction | `rwi_6neighbor_random_forest_model.pickle`, `new_top_40_feature_names.pickle`, and `household_input_2019_combined_new_latest_final_clean.pickle` |

Obtain datasets through authorised channels and match their schemas and identifiers before running. Pickle can execute code: only load trusted files. Historical pickles may require the original pandas and scikit-learn versions; the dependency list is not a recovered experiment lockfile.

## Interpretation and limitations

PCA sign and feature construction affect the direction and interpretation of the index. Wealth ranks should not be interpreted before checking those choices. This is a relative research measure, not a household financial assessment or an individual eligibility tool.

The notebooks assume upstream feature construction and train/test partitioning. They do not establish spatially independent evaluation or cross-year comparability. The normalised error diagnostics also need care when a target mean or range approaches zero. No original data, trained models, outputs, maps or performance claims are included.

## Provenance and release changes

These are existing Drive research notebooks with outputs, execution metadata, Colab mounting and inline package installation removed. Local paths replace private Drive paths. Regression reload prediction now uses the same selected columns as fitting, correcting the original full-feature/selected-feature mismatch. Other exploratory logic is retained.

Notebook structure and code-cell syntax have been checked. A small artificial-data regression check verifies the feature-selection correction; it is not a research result. The original dataset workflow has not been reproduced end to end.

This work formed part of a collaborative IIT Delhi research effort. This repository does not imply sole authorship of the surrounding datasets or broader project. No project-wide licence has been added; third-party data and dependencies retain their own terms.
