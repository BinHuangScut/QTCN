import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

import numpy as np
import pandas as pd
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def generate_data(data, lagged_hour):
    three_loads_hour, three_loads_y = list(), list()
    
    for i in range(data.size(-1)):
        # end_i = i + lagged_hour
        if i-lagged_hour < 0:
            continue
        seq_hour, seq_y = data[:, i-lagged_hour: i], data[:, i]
        three_loads_hour.append(seq_hour)
        three_loads_y.append(seq_y)
    
    three_loads_hour = torch.stack(three_loads_hour).type(torch.float32)
    three_loads_y = torch.stack(three_loads_y).type(torch.float32)
    
    return three_loads_hour, three_loads_y


def pinball_loss(Quantile, Predicted, Label):
    Error = Label - Predicted 
    Loss = torch.maximum(Error*Quantile, Error*(Quantile-1)).sum()
    return Loss/Predicted.size(0)


def get_seperate_results(all_outputs, three_loads_y_test, quantiles):
    electricity_results = torch.zeros(len(quantiles), three_loads_y_test.size(0))
    cold_results = torch.zeros(len(quantiles), three_loads_y_test.size(0))
    heat_results = torch.zeros(len(quantiles), three_loads_y_test.size(0))
    
    for i in range(len(quantiles)):
        electricity_results[i] = all_outputs[i][:, 0]
        cold_results[i] = all_outputs[i][:, 1]
        heat_results[i] = all_outputs[i][:, 2]
    
    electricity_results = electricity_results.cpu().detach()
    cold_results = cold_results.cpu().detach()
    heat_results = heat_results.cpu().detach()
    
    return electricity_results, cold_results, heat_results


def plot(y_test, forecased_result, num_result, start, flag):
    font = FontProperties()
    font.set_family('serif')
    font.set_name('Times New Roman')
    font.set_size(12)
    
    # electricity: flag = 0
    # cold: flag = 1
    # heat: flag = 2

    plt.figure(figsize = [17,5.2])
    plt.plot(y_test.cpu()[start:num_result+start, flag], color = 'red', label = 'Real Value')

    x = np.array([i for i in range(num_result)])
    plt.fill_between(x, forecased_result[2, start:num_result+start], forecased_result[-2,start:num_result+start], color = 'olive', label = '95%')
    plt.fill_between(x, forecased_result[4, start:num_result+start], forecased_result[-5,start:num_result+start], color = 'gold', label = '90%')
    plt.fill_between(x, forecased_result[9, start:num_result+start], forecased_result[-10,start:num_result+start], color = 'beige', label = '80%')
    # plt.plot(Model_output_2.detach().cpu()[:100], color = 'red', label = '1_input')

    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontproperties(font)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontproperties(font)

    plt.xlabel('t/h', fontproperties=font)
    plt.ylabel('Load/KW', fontproperties=font)

    ax.xaxis.label.set_fontsize(14)  
    ax.yaxis.label.set_fontsize(14)  

    ax.tick_params(axis='both', labelsize=12)  

    plt.legend(prop=font)
    plt.xlim(x[0], x[-1])
    # plt.savefig('electricity.pdf', format='pdf',bbox_inches='tight')
    plt.show()