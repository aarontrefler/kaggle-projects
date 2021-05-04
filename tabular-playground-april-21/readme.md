# Tabular Playground Series - Apr 2021
* [Kaggle Competition Link](https://www.kaggle.com/c/tabular-playground-series-apr-2021/overview)  
* Submission Date: 03-30-2021
* Score: 334/1244 (Top 27%)


## Overview
---
The aim of this project was to compare the effectiveness of the following tabular modelling approaches: XGBoost, Fastai Neural Network, TPOT AutoML.
* **Goal**: Create a model that predicts the survival outcome of a passenger on the Synthanic (i.e., simulated Titanic)
* **Preprocessing**: Feature engineering based on several publicy available Kaggle kernels
* **Modelling**
  * **XGBoost**: Test/Validation split was used to find the optimal tree limit. Final model was fit with the optimal tree limit and the entire labeled dataset.
  * **Fastai**: Model fit using default settings over 5 epochs. Custom imputation followed by Fastai processing (i.e., Catigorify and Normalization) was utilized
  * **TPOT**: AutoML process launched with a time limit of 2 hours and 20 minute max per pipeline. Custom imputation was utilized.

## Project
---
```
├── input                                                  <- Source data
├── working
│   ├── 1.0-preprocess.ipynb                               <- EDA and feature engineering
│   ├── 2.0-modelling-xgboost.ipynb                        <- XGBoost modelling and submission
│   ├── 2.1-modelling-fastai.ipynb                         <- Fastai modelling and submission
│   ├── 2.2-modelling-tpot.ipynb                           <- TPOT modelling and submission
│   ├── 3.0-prediction-voting.ipynb                        <- Prediction blending across modelling approaches
│   ├── data                                               <- Processed data
│   ├── models                                             <- Modelling output
│   ├── submissions                                        <- Kaggle submission files
├── readme.md
```

## Setup
---
### Download Competition Data to `input`
```
pip install kaggle --upgrade
kaggle competitions download -c tabular-playground-series-apr-2021
```
### Unzip Competition Data
```
sudo yum install epel-release
sudo yum --enablerepo=epel install p7zip
7za x tabular-playground-series-apr-2021.zip
```
