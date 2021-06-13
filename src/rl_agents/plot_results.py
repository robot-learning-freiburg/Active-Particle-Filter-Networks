#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.python.framework import tensor_util
from tensorflow.core.util import event_pb2
from tensorflow.python.lib.io import tf_record

def my_summary_iterator(path):
    for r in tf_record.tf_record_iterator(path):
        yield event_pb2.Event.FromString(r)

def read_events(summary_dir):
    events_data = {}
    for filename in os.listdir(summary_dir):
        path = os.path.join(summary_dir, filename)
        for event in my_summary_iterator(path):
            for value in event.summary.value:
                t = tensor_util.MakeNdarray(value.tensor)
                if value.tag in events_data:
                    events_data[value.tag].append(t)
                else:
                    events_data[value.tag] = [t]
    for key in events_data.keys():
        events_data[key] = np.stack(events_data[key], axis=0)

    return events_data

def boxplot(metric, title, filename):

    summary_dirs = {
        "gauss_500_less_noise_0.8": "./results/2021-06-13_22-00-00/rnd_agent/igibson_pfnet/gauss_500_0.15,0.5236_0.02,0.0873_0.8/log_dir/",
        "gauss_500_no_noise_0.8": "./results/2021-06-13_22-00-00/rnd_agent/igibson_pfnet/gauss_500_0.30,0.5236_0.0,0.0_0.8/log_dir/",
        "gauss_500_with_noise_0.8": "./results/2021-06-13_22-00-00/rnd_agent/igibson_pfnet/gauss_500_0.30,0.5236_0.04,0.0873_0.8/log_dir/",
        "gauss_500_with_noise_1.0": "./results/2021-06-13_22-00-00/rnd_agent/igibson_pfnet/gauss_500_0.30,0.5236_0.04,0.0873_1.0/log_dir/",
        "gauss_1500_with_noise_0.8": "./results/2021-06-13_22-00-00/rnd_agent/igibson_pfnet/gauss_1500_0.30,0.5236_0.04,0.0873_0.8/log_dir/",
        "uniform_1500_with_noise_0.8": "./results/2021-06-13_22-00-00/rnd_agent/igibson_pfnet/uniform_1500_0.30,0.5236_0.04,0.0873_0.8/log_dir/",
    }
    events_data = []
    for key, value in summary_dirs.items():
        events_data.append(read_events(value))

    fig = plt.figure(figsize =(16, 7))

    # create boxplot
    plt.boxplot([
            events_data[0][metric],
            events_data[1][metric],
            events_data[2][metric],
            events_data[3][metric],
            events_data[4][metric],
            events_data[5][metric]
    ])
    events = list(summary_dirs.keys())
    plt.xlabel('Experiment', fontweight='bold')
    plt.ylabel('RMSE (in meters)', fontweight='bold')
    plt.yticks(np.arange(0, 5, 0.5))
    plt.xticks(np.arange(1, len(events)+1), events)
    x1,x2,y1,y2 = plt.axis()
    plt.axis([x1, x2, 0, 5])
    plt.title(title, fontsize=16, fontweight='bold')

    # save figure
    plt.savefig(filename)

if __name__ == '__main__':
    boxplot('eps_final_rmse', 'Random Agent Performance w.r.t Finetuned PFNet (Episode End RMSE)', 'rnd_agent_igibson_pfnet_eps_final_rmse.png')
