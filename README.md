# Multi-task Learning Based Attentive Quantile Regression Temporal Convolutional Network for Multi-energy Probabilistic Load Forecasting

Authors: Han Guo, Bin Huang, Jianhui Wang (Southern Methodist University)

Motivation: The burgeoning proliferation of integrated energy systems has fostered an unprecedented degree of coupling among various energy streams, thereby elevating the necessity for unified multi-energy forecasting (MEF). Prior approaches predominantly relied on independent predictions for heterogeneous load demands, overlooking the synergy embedded within the dataset. The two principal challenges in MEF are extracting the intricate coupling correlations among diverse loads and accurately capturing the inherent uncertainties associated with each type of load. This study proposes an attentive quantile regression temporal convolutional network (QTCN) as a probabilistic framework for MEF, featuring an end-to-end predictor for the probabilistic intervals of electrical, thermal, and cooling loads. This study leverages an attention layer to extract correlations between diverse loads. Subsequently, a QTCN is implemented to retain the temporal characteristics of load data and gauge the uncertainties and temporal correlations of each load type. The multi-task learning framework is deployed to facilitate simultaneous regression of various quantiles, thereby expediting the training progression of the forecasting model.

Implementation of this code package:<br />
1) data preprocessing and plot function: utils.py
2) attention tcn network: attention_layer.py, tcn.py and attention_tcn.py
3) training and inference: main.ipynb
4) heatmap of attention score: heatmap_attentionscore.ipynb
5) evaluation function: evaluation.py
6) experimental configuration: options.py

Questions? Contact Han Guo at guoh@smu.edu.
