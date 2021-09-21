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
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)

    # igibson apartments generalize
    generalize_aprts_path = "/media/neo/robotics/final_report/Aprts_400_8.0/Aprts_rgbd/eval/events.out.tfevents.1630605625.rlgpu2.17887.1.v2"
    generalize_aprts_loss = np.array(getEventFileData(generalize_aprts_path)["loss"])
    generalize_aprts = ax.plot(generalize_aprts_loss[:, 0], generalize_aprts_loss[:, 1])
    # generalize_aprts_path = "/media/neo/robotics/final_report/Aprts_400_8.0/Aprts_rgbd/train/events.out.tfevents.1630605625.rlgpu2.17887.0.v2"
    # generalize_aprts_loss = np.array(getEventFileData(generalize_aprts_path)["loss"])
    # generalize_aprts = ax.plot(generalize_aprts_loss[:, 0], generalize_aprts_loss[:, 1])

    # igibson apartments
    aprts_path = "/media/neo/robotics/final_report/Aprts_400_8.0/Aprts_overfit_rgbd/eval/events.out.tfevents.1630920914.rlgpu2.12306.1.v2"
    aprts_loss = np.array(getEventFileData(aprts_path)["loss"])
    aprts = ax.plot(aprts_loss[:, 0], aprts_loss[:, 1])
    # aprts_path = "/media/neo/robotics/final_report/Aprts_400_8.0/Aprts_overfit_rgbd/train/events.out.tfevents.1630920914.rlgpu2.12306.0.v2"
    # aprts_loss = np.array(getEventFileData(aprts_path)["loss"])
    # aprts = ax.plot(aprts_loss[:, 0], aprts_loss[:, 1])

    # igibson 10 apartments
    ten_aprts_path = "/media/neo/robotics/final_report/Aprts_400_8.0/10_Aprts_rgbd/eval/events.out.tfevents.1630920486.rlgpu2.39725.1.v2"
    ten_aprts_loss = np.array(getEventFileData(ten_aprts_path)["loss"])
    ten_aprts = ax.plot(ten_aprts_loss[:, 0], ten_aprts_loss[:, 1])
    # ten_aprts_path = "/media/neo/robotics/final_report/Aprts_400_8.0/10_Aprts_rgbd/train/events.out.tfevents.1630920486.rlgpu2.39725.0.v2"
    # ten_aprts_loss = np.array(getEventFileData(ten_aprts_path)["loss"])
    # ten_aprts = ax.plot(ten_aprts_loss[:, 0], ten_aprts_loss[:, 1])

    # igibson 1 apartments
    one_aprts_path = "/media/neo/robotics/final_report/Rs_400_8.0/Rs_rgb_depth/eval/events.out.tfevents.1631779563.rlgpu2.40822.1.v2"
    one_aprts_loss = np.array(getEventFileData(one_aprts_path)["loss"])
    one_aprts = ax.plot(one_aprts_loss[:, 0], one_aprts_loss[:, 1])
    # one_aprts_path = "/media/neo/robotics/final_report/Rs_400_8.0/Rs_rgb_depth/train/events.out.tfevents.1631779563.rlgpu2.40822.0.v2"
    # one_aprts_loss = np.array(getEventFileData(one_aprts_path)["loss"])
    # one_aprts = ax.plot(one_aprts_loss[:, 0], one_aprts_loss[:, 1])

    ax.set_title('Evaluation loss for iGibson environment')
    ax.set_xlabel("# of train epochs")
    ax.set_ylabel("RMSE (cm)")
    ax.legend([
                "115 Floors (Generalize)",
                # "115 Floors Train (Generalize)",
                "115 Floors",
                # "115 Floors Train",
                "  15 Floors",
                # "  15 Floors Train",
                "    1 Floor",
                # "    1 Floor Train"
            ], loc='upper right', fontsize=10)

    plt.show()
    fig.savefig("igibson_eval_loss.png")

if __name__ == '__main__':
    main()
