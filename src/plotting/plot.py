#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.summary.summary_iterator import summary_iterator
from tensorflow.python.framework import tensor_util

def getEventFileData(path):
    data = {}
    for event in summary_iterator(path):
        for value in event.summary.value:
            t = tensor_util.MakeNdarray(value.tensor)
            if value.tag not in data:
                data[value.tag] = []
            data[value.tag].append([event.step, t.item()])
    return data

def main():
    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111)
    plot = 'eval'

    if plot == 'train':

        # igibson apartments generalize
        batches = 8000
        generalize_aprts_path = "/media/neo/robotics/final_report/Aprts_400_8.0/Aprts_rgbd/train/events.out.tfevents.1630605625.rlgpu2.17887.0.v2"
        generalize_aprts_loss = np.array(getEventFileData(generalize_aprts_path)["loss"])
        generalize_aprts = ax.plot(generalize_aprts_loss[:, 0]*batches, generalize_aprts_loss[:, 1])

        # igibson apartments
        # batches = 8000
        # aprts_path = "/media/neo/robotics/final_report/Aprts_400_8.0/Aprts_overfit_rgbd/train/events.out.tfevents.1630920914.rlgpu2.12306.0.v2"
        # aprts_loss = np.array(getEventFileData(aprts_path)["loss"])
        # aprts = ax.plot(aprts_loss[:, 0]*batches, aprts_loss[:, 1])

        # igibson 10 apartments
        batches = 1250
        ten_aprts_path = "/media/neo/robotics/final_report/Aprts_400_8.0/10_Aprts_rgbd/train/events.out.tfevents.1630920486.rlgpu2.39725.0.v2"
        ten_aprts_loss = np.array(getEventFileData(ten_aprts_path)["loss"])
        ten_aprts = ax.plot(ten_aprts_loss[:, 0]*batches, ten_aprts_loss[:, 1])

        # igibson 1 apartments
        batches = 330
        one_aprts_path = "/media/neo/robotics/final_report/Rs_400_8.0/Rs_rgb_depth/train/events.out.tfevents.1631779563.rlgpu2.40822.0.v2"
        one_aprts_loss = np.array(getEventFileData(one_aprts_path)["loss"])
        one_aprts = ax.plot(one_aprts_loss[:, 0]*batches, one_aprts_loss[:, 1])

        ax.set_title('Training loss for iGibson environment', fontsize=18, weight='bold')
        ax.set_xlabel("number of training batches", fontsize=16)
        ax.set_ylabel("mean square error (cm)", fontsize=16)
        ax.legend([
                    "115 Apartments",
                    # "115 Floors",
                    "17 Apartments",
                    "1 Apartment"
                ], loc='upper right', fontsize=12)

        plt.show()
        fig.savefig("igibson_train_loss.png")
    else:

        # igibson apartments generalize
        batches = 1000
        generalize_aprts_path = "/media/neo/robotics/final_report/Aprts_400_8.0/Aprts_rgbd/eval/events.out.tfevents.1630605625.rlgpu2.17887.1.v2"
        generalize_aprts_loss = np.array(getEventFileData(generalize_aprts_path)["loss"])
        generalize_aprts = ax.plot(generalize_aprts_loss[:, 0]*batches, generalize_aprts_loss[:, 1])

        # igibson apartments
        # aprts_path = "/media/neo/robotics/final_report/Aprts_400_8.0/Aprts_overfit_rgbd/eval/events.out.tfevents.1630920914.rlgpu2.12306.1.v2"
        # aprts_loss = np.array(getEventFileData(aprts_path)["loss"])
        # aprts = ax.plot(aprts_loss[:, 0], aprts_loss[:, 1])

        # igibson 10 apartments
        batches = 300
        ten_aprts_path = "/media/neo/robotics/final_report/Aprts_400_8.0/10_Aprts_rgbd/eval/events.out.tfevents.1630920486.rlgpu2.39725.1.v2"
        ten_aprts_loss = np.array(getEventFileData(ten_aprts_path)["loss"])
        ten_aprts = ax.plot(ten_aprts_loss[:, 0]*batches, ten_aprts_loss[:, 1])

        # igibson 1 apartments
        batches = 40
        one_aprts_path = "/media/neo/robotics/final_report/Rs_400_8.0/Rs_rgb_depth/eval/events.out.tfevents.1631779563.rlgpu2.40822.1.v2"
        one_aprts_loss = np.array(getEventFileData(one_aprts_path)["loss"])
        one_aprts = ax.plot(one_aprts_loss[:, 0]*batches, one_aprts_loss[:, 1])

        ax.set_title('Evaluation loss for iGibson environment', fontsize=18, weight='bold')
        ax.set_xlabel("number of evaluation batches", fontsize=16)
        ax.set_ylabel("mean square error (cm)", fontsize=16)
        ax.legend([
                    "15 Apartments (unseen)",
                    # "15 Floors",
                    "4 Apartments",
                    "1 Apartment"
                ], loc='upper right', fontsize=12)

        plt.show()
        fig.savefig("igibson_eval_loss.png")

if __name__ == '__main__':
    main()
