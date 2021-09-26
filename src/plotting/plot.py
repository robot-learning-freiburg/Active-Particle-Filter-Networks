#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.summary.summary_iterator import summary_iterator
from tensorflow.python.framework import tensor_util

def getEventFileData(path):
    data = {}
    for event in summary_iterator(path):
        for value in event.summary.value:
            if value.simple_value == 0.0:
                t = tensor_util.MakeNdarray(value.tensor)
            else:
                t = np.array([value.simple_value])
            if value.tag not in data:
                data[value.tag] = []
            data[value.tag].append([event.step, t.item()])
    return data

def generalization_plts():
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

def house3d_plts():
    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111)

    # pfnet house3d apartment
    pfnet_train_path = "/media/neo/robotics/final_report/House3D_4000_8.0/house3d_rgb_depth/train/events.out.tfevents.1631127344.rlgpu2.10177.0.v2"
    pfnet_train_loss = np.array(getEventFileData(pfnet_train_path)["loss"])
    pfnet_eval_path = "/media/neo/robotics/final_report/House3D_4000_8.0/house3d_rgb_depth/eval/events.out.tfevents.1631127344.rlgpu2.10177.1.v2"
    pfnet_eval_loss = np.array(getEventFileData(pfnet_eval_path)["loss"])
    pfnet_train = ax.plot(pfnet_train_loss[:, 0], pfnet_train_loss[:, 1])
    pfnet_eval = ax.plot(pfnet_eval_loss[:, 0], pfnet_eval_loss[:, 1])

    # dpf house3d apartment
    dpf_train_path = "/media/neo/robotics/deep-activate-localization/bckp/jan/jan_22/runs/Jan23_00-10-06_pearl8/train_stats_mean_total_loss/events.out.tfevents.1611357667.pearl8.6887.3"
    dpf_train_loss = np.array(getEventFileData(dpf_train_path)["train_stats"])
    dpf_eval_path = "/media/neo/robotics/deep-activate-localization/bckp/jan/jan_27_1/runs/Jan27_10-55-58_pearl8/eval_stats_mean_loss/events.out.tfevents.1611741820.pearl8.17432.3"
    dpf_eval_loss = np.array(getEventFileData(dpf_eval_path)["eval_stats"])
    dpf_train = ax.plot(dpf_train_loss[:, 0], dpf_train_loss[:, 1])
    dpf_eval = ax.plot(dpf_eval_loss[:, 0]*3, dpf_eval_loss[:, 1])

    ax.set_title('Training/Evaluation loss for House3D environment', fontsize=18, weight='bold')
    ax.set_xlabel("number of train epochs", fontsize=16)
    ax.set_ylabel("mean square error (cm)", fontsize=16)
    ax.legend([
                "PFNet Train",
                "PFNet Eval",
                "DPF Train",
                "DPF Eval"
            ], loc='upper right', fontsize=12)

    plt.show()
    fig.savefig("house3d_loss.png")

def igibson_plts():
    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111)

    N = np.array([500, 500, 1000])
    T = np.array([50, 100, 100])
    accuracy = np.array([90.0, 90.625, 91.875])
    area = np.array([100, 200, 300])
    colors = np.random.rand(len(N+1))

    ax.scatter(x=N, y=T, s=area, c=colors, alpha=0.5)
    for i, txt in enumerate(accuracy):
        ax.annotate(f"    {txt}", (N[i], T[i]))

    ax.set_xticks(np.array([0, 250, 500, 1000]))
    ax.set_yticks(np.array([0, 10, 50, 100]))
    ax.set_title('iGibson PFNet global localization RGB-D accuracy ', fontsize=18, weight='bold')
    ax.set_xlabel("number of particles (N)", fontsize=16)
    ax.set_ylabel("episode steps (t)", fontsize=16)

    plt.show()
    fig.savefig("igibson_rgbd_accuracy.png")

if __name__ == '__main__':
    # generalization_plts()
    # house3d_plts()
    igibson_plts()
