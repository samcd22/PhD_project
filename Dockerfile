FROM python:3.8

# Installing numpy, matplotlib, pandas, os, json, numpyencoder, imagio, fractions, labellines, warnings, statsmodels, jax, numpyro, sklearn, optuna, shutil
RUN pip install numpy pandas matplotlib os json 
RUN pip install numpyencoder imagio fractions labellines warnings statsmodels jax numpyro sklearn optuna shutil