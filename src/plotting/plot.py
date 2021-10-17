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
                ], loc='upper right', fontsize=14)

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
                ], loc='upper right', fontsize=14)

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
    ax.set_ylabel("pose error (meters)", fontsize=16)
    ax.legend([
                "PFNet Train",
                "PFNet Eval",
                "DPF Train",
                "DPF Eval"
            ], loc='upper right', fontsize=14)

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
        ax.annotate(f"    {txt}", (N[i], T[i]), fontsize=16)

    ax.set_xticks(np.array([0, 250, 500, 1000]))
    ax.set_yticks(np.array([0, 10, 50, 100]))
    ax.set_title('iGibson PFNet global localization RGB-D success (%) ', fontsize=18, weight='bold')
    ax.set_xlabel("number of particles (N)", fontsize=16)
    ax.set_ylabel("episode steps (t)", fontsize=16)

    plt.show()
    fig.savefig("igibson_rgbd_accuracy.png")

def belief_plts():
    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111)

    # kmeans representation
    pfnet_train_path = "/media/neo/robotics/August/20-07-2021/train_rl_uniform_kmeans/train/events.out.tfevents.1629406907.pearl9.4239.0.v2"
    pfnet_train_return = np.array(getEventFileData(pfnet_train_path)["Metrics/AverageReturn"])
    pfnet_eval_path = "/media/neo/robotics/August/20-07-2021/train_rl_uniform_kmeans/eval/events.out.tfevents.1629406907.pearl9.4239.1.v2"
    pfnet_eval_return = np.array(getEventFileData(pfnet_eval_path)["Metrics/AverageReturn"])
    pfnet_train = ax.plot(pfnet_train_return[:, 0], pfnet_train_return[:, 1])
    pfnet_eval = ax.plot(pfnet_eval_return[:, 0], pfnet_eval_return[:, 1])

    # belief map representation
    dpf_train_path = "/media/neo/robotics/August/20-07-2021/train_rl_uniform_likelihood/train/events.out.tfevents.1629406377.pearl8.20947.0.v2"
    dpf_train_return = np.array(getEventFileData(dpf_train_path)["Metrics/AverageReturn"])
    dpf_eval_path = "/media/neo/robotics/August/20-07-2021/train_rl_uniform_likelihood/eval/events.out.tfevents.1629406377.pearl8.20947.1.v2"
    dpf_eval_return = np.array(getEventFileData(dpf_eval_path)["Metrics/AverageReturn"])
    dpf_train = ax.plot(dpf_train_return[:, 0], dpf_train_return[:, 1])
    dpf_eval = ax.plot(dpf_eval_return[:, 0], dpf_eval_return[:, 1])

    ax.set_title('Training/Evaluation episode return for SAC agent', fontsize=18, weight='bold')
    ax.set_xlabel("number of train epochs", fontsize=16)
    ax.set_ylabel("average episode return", fontsize=16)
    ax.legend([
                "KMeans (k=10) Train",
                "KMeans (k=10) Eval",
                "Belief Map Train",
                "Belief Map Eval"
            ], loc='upper right', fontsize=14)

    plt.show()
    fig.savefig("particle_rep_sac_return.png")


def rl_train_eval_plts():
    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111)
    plot = 'train'

    if plot == 'train':

        # 1.0 box + 25 steps rl agent
        box_path_1_0 = "/media/neo/robotics/August/17-09-2021/train_rl_uniform_0.5_box_25/train/events.out.tfevents.1631716010.pearl8.18818.0.v2"
        box_return_1_0 = np.array(getEventFileData(box_path_1_0)["Metrics/AverageReturn"])
        box_1_0 = ax.plot(box_return_1_0[:, 0], box_return_1_0[:, 1])

        # 2.0 box + 25 steps rl agent
        box_path_2_0 = "/media/neo/robotics/August/17-09-2021/train_rl_uniform_1.0_box_25/train/events.out.tfevents.1631867346.pearl8.18370.0.v2"
        box_return_2_0 = np.array(getEventFileData(box_path_2_0)["Metrics/AverageReturn"])
        box_2_0 = ax.plot(box_return_2_0[:, 0], box_return_2_0[:, 1])

        # 4.0 box + 50 steps rl agent
        box_path_4_0 = "/media/neo/robotics/August/17-09-2021/train_rl_uniform_2.0_box_50/train/events.out.tfevents.1632144090.pearl2.5531.0.v2"
        box_return_4_0 = np.array(getEventFileData(box_path_4_0)["Metrics/AverageReturn"])
        box_4_0 = ax.plot(box_return_4_0[:, 0], box_return_4_0[:, 1])

        # 5.0 box + 50 steps rl agent
        box_path_5_0 = "/media/neo/robotics/August/17-09-2021/train_rl_uniform_2.5_box_50/train/events.out.tfevents.1632349779.pearl8.24775.0.v2"
        box_return_5_0 = np.array(getEventFileData(box_path_5_0)["Metrics/AverageReturn"])
        box_5_0 = ax.plot(box_return_5_0[:, 0], box_return_5_0[:, 1])

        ax.set_title('Training episode return for SAC agent with Belief Map', fontsize=18, weight='bold')
        ax.set_xlabel("number of train epochs", fontsize=16)
        ax.set_ylabel("average episode return", fontsize=16)
        ax.legend([
                    "1.0 sampling box",
                    "2.0 sampling box",
                    "4.0 sampling box",
                    "5.0 sampling box"
                ], loc='upper right', fontsize=14)

        plt.show()
        fig.savefig("rl_belief_train_returns.png")
    else:

        # 1.0 box + 25 steps rl agent
        box_path_1_0 = "/media/neo/robotics/August/17-09-2021/train_rl_uniform_0.5_box_25/eval/events.out.tfevents.1631716010.pearl8.18818.1.v2"
        box_return_1_0 = np.array(getEventFileData(box_path_1_0)["Metrics/AverageReturn"])
        box_1_0 = ax.plot(box_return_1_0[:, 0], box_return_1_0[:, 1])

        # 2.0 box + 25 steps rl agent
        box_path_2_0 = "/media/neo/robotics/August/17-09-2021/train_rl_uniform_1.0_box_25/eval/events.out.tfevents.1631867347.pearl8.18370.1.v2"
        box_return_2_0 = np.array(getEventFileData(box_path_2_0)["Metrics/AverageReturn"])
        box_2_0 = ax.plot(box_return_2_0[:, 0], box_return_2_0[:, 1])

        # 4.0 box + 50 steps rl agent
        box_path_4_0 = "/media/neo/robotics/August/17-09-2021/train_rl_uniform_2.0_box_50/eval/events.out.tfevents.1632144090.pearl2.5531.1.v2"
        box_return_4_0 = np.array(getEventFileData(box_path_4_0)["Metrics/AverageReturn"])
        box_4_0 = ax.plot(box_return_4_0[:, 0], box_return_4_0[:, 1])

        # 5.0 box + 50 steps rl agent
        box_path_5_0 = "/media/neo/robotics/August/17-09-2021/train_rl_uniform_2.5_box_50/eval/events.out.tfevents.1632349779.pearl8.24775.1.v2"
        box_return_5_0 = np.array(getEventFileData(box_path_5_0)["Metrics/AverageReturn"])
        box_5_0 = ax.plot(box_return_5_0[:, 0], box_return_5_0[:, 1])

        ax.set_title('Evaluation episode return for SAC agent with belief map', fontsize=18, weight='bold')
        ax.set_xlabel("number of train epochs", fontsize=16)
        ax.set_ylabel("average episode return", fontsize=16)
        ax.legend([
                    "1.0 sampling box",
                    "2.0 sampling box",
                    "4.0 sampling box",
                    "5.0 sampling box"
                ], loc='upper right', fontsize=14)

        plt.show()
        fig.savefig("rl_belief_eval_returns.png")


def rl_test_plts():
    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111)
    plot = 'collision_penalty'

    if plot == 'collision_penalty':
        # obstacle avoid agent
        avoid_path = "/media/neo/robotics/August/17-09-2021/train_rl_uniform_2.0_box_50/avoid_agent/events.out.tfevents.1632427985.pearl2.18690.0.v2"
        avoid_eps_mcp = np.array(getEventFileData(avoid_path)["per_eps_mcp"])

        # random agent
        rnd_path = "/media/neo/robotics/August/17-09-2021/train_rl_uniform_2.0_box_50/rnd_agent/events.out.tfevents.1632428087.pearl2.18689.0.v2"
        rnd_eps_mcp = np.array(getEventFileData(rnd_path)["per_eps_mcp"])

        # trained sac agent
        sac_path = "/media/neo/robotics/August/17-09-2021/train_rl_uniform_2.0_box_50/sac_agent/events.out.tfevents.1632427849.pearl5.11213.0.v2"
        sac_eps_mcp = np.array(getEventFileData(sac_path)["per_eps_mcp"])

        data = np.concatenate([
                    -avoid_eps_mcp[:, 1:2],
                    -rnd_eps_mcp[:, 1:2],
                    -sac_eps_mcp[:, 1:2]
                ], axis=1)
        ax.boxplot(data)

        ax.set_title('Episode mean collision penalty for 4.0 sampling box', fontsize=18, weight='bold')
        # ax.set_xlabel("agent behavior", fontsize=16)
        ax.set_ylabel("mean collision penalty (%)", fontsize=16)
        # ax.set_ylim(-1.05, 0.05)
        ax.set_xticklabels([
                    "Obstacle Avoidance Agent",
                    "Random Action Agent",
                    "Trained SAC Agent"
                ], fontsize=16)

        plt.show()
        fig.savefig("rl_belief_test_mcp.png")

    elif plot == 'orientation_error':
        # obstacle avoid agent
        avoid_path = "/media/neo/robotics/August/17-09-2021/train_rl_uniform_2.0_box_50/avoid_agent/events.out.tfevents.1632427985.pearl2.18690.0.v2"
        avoid_eps_mso = np.array(getEventFileData(avoid_path)["per_eps_mso"])

        # random agent
        rnd_path = "/media/neo/robotics/August/17-09-2021/train_rl_uniform_2.0_box_50/rnd_agent/events.out.tfevents.1632428087.pearl2.18689.0.v2"
        rnd_eps_mso = np.array(getEventFileData(rnd_path)["per_eps_mso"])

        # trained sac agent
        sac_path = "/media/neo/robotics/August/17-09-2021/train_rl_uniform_2.0_box_50/sac_agent/events.out.tfevents.1632427849.pearl5.11213.0.v2"
        sac_eps_mso = np.array(getEventFileData(sac_path)["per_eps_mso"])

        data = np.concatenate([
                    avoid_eps_mso[:, 1:2],
                    rnd_eps_mso[:, 1:2],
                    sac_eps_mso[:, 1:2]
                ], axis=1)
        ax.boxplot(data)

        ax.set_title('Episode mean orientation error for 4.0 sampling box', fontsize=18, weight='bold')
        # ax.set_xlabel("agent behavior", fontsize=16)
        ax.set_ylabel("mean orientation error (radians)", fontsize=16)
        ax.set_ylim(-0.05, 0.15)
        ax.set_xticklabels([
                    "Obstacle Avoidance Agent",
                    "Random Action Agent",
                    "Trained SAC Agent"
                ], fontsize=16)

        plt.show()
        fig.savefig("rl_belief_test_mso.png")

    elif plot == 'position_error':
        # obstacle avoid agent
        avoid_path = "/media/neo/robotics/August/17-09-2021/train_rl_uniform_2.0_box_50/avoid_agent/events.out.tfevents.1632427985.pearl2.18690.0.v2"
        avoid_eps_msp = np.array(getEventFileData(avoid_path)["per_eps_msp"])

        # random agent
        rnd_path = "/media/neo/robotics/August/17-09-2021/train_rl_uniform_2.0_box_50/rnd_agent/events.out.tfevents.1632428087.pearl2.18689.0.v2"
        rnd_eps_msp = np.array(getEventFileData(rnd_path)["per_eps_msp"])

        # trained sac agent
        sac_path = "/media/neo/robotics/August/17-09-2021/train_rl_uniform_2.0_box_50/sac_agent/events.out.tfevents.1632427849.pearl5.11213.0.v2"
        sac_eps_msp = np.array(getEventFileData(sac_path)["per_eps_msp"])

        data = np.concatenate([
                    avoid_eps_msp[:, 1:2],
                    rnd_eps_msp[:, 1:2],
                    sac_eps_msp[:, 1:2]
                ], axis=1)
        ax.boxplot(data)

        ax.set_title('Episode mean position error for 4.0 sampling box', fontsize=18, weight='bold')
        # ax.set_xlabel("agent behavior", fontsize=16)
        ax.set_ylabel("mean position error (meters)", fontsize=16)
        ax.set_ylim(-0.05, 2.0)
        ax.set_xticklabels([
                    "Obstacle Avoidance Agent",
                    "Random Action Agent",
                    "Trained SAC Agent"
                ], fontsize=16)

        plt.show()
        fig.savefig("rl_belief_test_msp.png")

    else:
        # obstacle avoid agent
        avoid_path = "/media/neo/robotics/August/17-09-2021/train_rl_uniform_2.0_box_50/avoid_agent/events.out.tfevents.1632427985.pearl2.18690.0.v2"
        avoid_eps_end_mse = np.array(getEventFileData(avoid_path)["per_eps_end_reward"])

        # random agent
        rnd_path = "/media/neo/robotics/August/17-09-2021/train_rl_uniform_2.0_box_50/rnd_agent/events.out.tfevents.1632428087.pearl2.18689.0.v2"
        rnd_eps_end_mse = np.array(getEventFileData(rnd_path)["per_eps_end_reward"])

        # trained sac agent
        sac_path = "/media/neo/robotics/August/17-09-2021/train_rl_uniform_2.0_box_50/sac_agent/events.out.tfevents.1632427849.pearl5.11213.0.v2"
        sac_eps_end_mse = np.array(getEventFileData(sac_path)["per_eps_end_reward"])

        data = np.concatenate([
                    -avoid_eps_end_mse[:, 1:2],
                    -rnd_eps_end_mse[:, 1:2],
                    -sac_eps_end_mse[:, 1:2]
                ], axis=1)
        ax.boxplot(data)

        ax.set_title('Episode end pose error for 4.0 sampling box', fontsize=18, weight='bold')
        # ax.set_xlabel("agent behavior", fontsize=16)
        ax.set_ylabel("mean squared error (meters)", fontsize=16)
        ax.set_ylim(-0.05, 0.3)
        ax.set_xticklabels([
                    "Obstacle Avoidance Agent",
                    "Random Action Agent",
                    "Trained SAC Agent"
                ], fontsize=16)

        plt.show()
        fig.savefig("rl_belief_test_end_mse.png")


def diff_steps_plts():
    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111)
    plot = 'average_return'

    # # 1.0 box + 25 steps rl agent
    # box_path_1_0_25 = "/media/neo/robotics/August/17-09-2021/train_rl_uniform_1.0_box_25/train/events.out.tfevents.1631867346.pearl8.18370.0.v2"
    # box_return_1_0_25 = np.array(getEventFileData(box_path_1_0_25)["Metrics/AverageReturn"])
    # box_1_0_25 = ax.plot(box_return_1_0_25[:, 0], box_return_1_0_25[:, 1])
    #
    # # 1.0 box + 50 steps rl agent
    # box_path_1_0_50 = "/media/neo/robotics/August/17-09-2021/train_rl_uniform_1.0_box_50/train/events.out.tfevents.1631961881.pearl2.22000.0.v2"
    # box_return_1_0_50 = np.array(getEventFileData(box_path_1_0_50)["Metrics/AverageReturn"])
    # box_1_0_50 = ax.plot(box_return_1_0_50[:, 0], box_return_1_0_50[:, 1])

    if plot == 'average_return':
        # 1.0 box + 25 steps rl agent
        box_path_1_0_25 = "/media/neo/robotics/August/17-09-2021/train_rl_uniform_1.0_box_25/eval/events.out.tfevents.1631867347.pearl8.18370.1.v2"
        box_return_1_0_25 = np.array(getEventFileData(box_path_1_0_25)["Metrics/AverageReturn"])
        box_1_0_25 = ax.plot(box_return_1_0_25[:, 0], box_return_1_0_25[:, 1])

        # 1.0 box + 50 steps rl agent
        box_path_1_0_50 = "/media/neo/robotics/August/17-09-2021/train_rl_uniform_1.0_box_50/eval/events.out.tfevents.1631961881.pearl2.22000.1.v2"
        box_return_1_0_50 = np.array(getEventFileData(box_path_1_0_50)["Metrics/AverageReturn"])
        box_1_0_50 = ax.plot(box_return_1_0_50[:, 0], box_return_1_0_50[:, 1])

        ax.set_title('Average episode return for SAC agent with belief map (2.0 sampling box)', fontsize=18, weight='bold')
        ax.set_xlabel("number of eval epochs", fontsize=16)
        ax.set_ylabel("average episode return", fontsize=16)
        ax.legend([
                    "25 particle filter steps",
                    "50 particle filter steps"
                ], loc='upper right', fontsize=14)

        plt.show()
        fig.savefig("diff_steps_eval_avg_eps_return.png")

    elif plot == 'collision_penalty':
        # 1.0 box + 25 steps rl agent
        box_path_1_0_25 = "/media/neo/robotics/August/17-09-2021/train_rl_uniform_1.0_box_25/eval/events.out.tfevents.1631867347.pearl8.18370.1.v2"
        box_return_1_0_25 = np.array(getEventFileData(box_path_1_0_25)["Metrics/AverageStepCollisionPenality"])
        box_1_0_25 = ax.plot(box_return_1_0_25[:, 0], box_return_1_0_25[:, 1])

        # 1.0 box + 50 steps rl agent
        box_path_1_0_50 = "/media/neo/robotics/August/17-09-2021/train_rl_uniform_1.0_box_50/eval/events.out.tfevents.1631961881.pearl2.22000.1.v2"
        box_return_1_0_50 = np.array(getEventFileData(box_path_1_0_50)["Metrics/AverageStepCollisionPenality"])
        box_1_0_50 = ax.plot(box_return_1_0_50[:, 0], box_return_1_0_50[:, 1])

        ax.set_title('Average step collision penalty for SAC agent with belief map (2.0 sampling box)', fontsize=18, weight='bold')
        ax.set_xlabel("number of eval epochs", fontsize=16)
        ax.set_ylabel("average collision penalty (%)", fontsize=16)
        ax.legend([
                    "25 particle filter steps",
                    "50 particle filter steps"
                ], loc='upper right', fontsize=14)

        plt.show()
        fig.savefig("diff_steps_eval_avg_step_collision.png")

    elif plot == 'orientation_error':
        # 1.0 box + 25 steps rl agent
        box_path_1_0_25 = "/media/neo/robotics/August/17-09-2021/train_rl_uniform_1.0_box_25/eval/events.out.tfevents.1631867347.pearl8.18370.1.v2"
        box_return_1_0_25 = np.array(getEventFileData(box_path_1_0_25)["Metrics/AverageStepOrientationError"])
        box_1_0_25 = ax.plot(box_return_1_0_25[:, 0], box_return_1_0_25[:, 1])
    # box_return_1_0_25 = np.array(getEventFileData(box_path_1_0_25)["Metrics/AverageStepPositionError"])

        # 1.0 box + 50 steps rl agent
        box_path_1_0_50 = "/media/neo/robotics/August/17-09-2021/train_rl_uniform_1.0_box_50/eval/events.out.tfevents.1631961881.pearl2.22000.1.v2"
        box_return_1_0_50 = np.array(getEventFileData(box_path_1_0_50)["Metrics/AverageStepOrientationError"])
        box_1_0_50 = ax.plot(box_return_1_0_50[:, 0], box_return_1_0_50[:, 1])
    # box_return_1_0_50 = np.array(getEventFileData(box_path_1_0_50)["Metrics/AverageStepPositionError"])


        ax.set_title('Average step orientation error for SAC agent with belief map (2.0 sampling box)', fontsize=18, weight='bold')
        ax.set_xlabel("number of eval epochs", fontsize=16)
        ax.set_ylabel("average orientation error (radians)", fontsize=16)
        ax.legend([
                    "25 particle filter steps",
                    "50 particle filter steps"
                ], loc='upper right', fontsize=14)

        plt.show()
        fig.savefig("diff_steps_eval_avg_step_orientation.png")

    else:
        # 1.0 box + 25 steps rl agent
        box_path_1_0_25 = "/media/neo/robotics/August/17-09-2021/train_rl_uniform_1.0_box_25/eval/events.out.tfevents.1631867347.pearl8.18370.1.v2"
        box_return_1_0_25 = np.array(getEventFileData(box_path_1_0_25)["Metrics/AverageStepPositionError"])
        box_1_0_25 = ax.plot(box_return_1_0_25[:, 0], box_return_1_0_25[:, 1])

        # 1.0 box + 50 steps rl agent
        box_path_1_0_50 = "/media/neo/robotics/August/17-09-2021/train_rl_uniform_1.0_box_50/eval/events.out.tfevents.1631961881.pearl2.22000.1.v2"
        box_return_1_0_50 = np.array(getEventFileData(box_path_1_0_50)["Metrics/AverageStepPositionError"])
        box_1_0_50 = ax.plot(box_return_1_0_50[:, 0], box_return_1_0_50[:, 1])


        ax.set_title('Average step position error for SAC agent with belief map (2.0 sampling box)', fontsize=18, weight='bold')
        ax.set_xlabel("number of eval epochs", fontsize=16)
        ax.set_ylabel("average position error (meters)", fontsize=16)
        ax.legend([
                    "25 particle filter steps",
                    "50 particle filter steps"
                ], loc='upper right', fontsize=14)

        plt.show()
        fig.savefig("diff_steps_eval_avg_step_position.png")

def diff_resample_plts():
    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111)
    plot = 'average_return'

    # # 1.0 box + 25 steps rl agent
    # box_path_1_0_25 = "/media/neo/robotics/August/17-09-2021/train_rl_uniform_1.0_box_25/train/events.out.tfevents.1631867346.pearl8.18370.0.v2"
    # box_return_1_0_25 = np.array(getEventFileData(box_path_1_0_25)["Metrics/AverageReturn"])
    # box_1_0_25 = ax.plot(box_return_1_0_25[:, 0], box_return_1_0_25[:, 1])
    #
    # # 1.0 box + 50 steps rl agent
    # box_path_1_0_50 = "/media/neo/robotics/August/17-09-2021/train_rl_uniform_1.0_box_50/train/events.out.tfevents.1631961881.pearl2.22000.0.v2"
    # box_return_1_0_50 = np.array(getEventFileData(box_path_1_0_50)["Metrics/AverageReturn"])
    # box_1_0_50 = ax.plot(box_return_1_0_50[:, 0], box_return_1_0_50[:, 1])

    if plot == 'average_return':
        # 0.5 box + 25 steps + 1.0 resample rl agent
        box_path_1_0 = "/media/neo/robotics/August/17-09-2021/train_rl_uniform_0.5_box_25/eval/events.out.tfevents.1631716010.pearl8.18818.1.v2"
        box_return_1_0 = np.array(getEventFileData(box_path_1_0)["Metrics/AverageReturn"])
        box_1_0 = ax.plot(box_return_1_0[:, 0], box_return_1_0[:, 1])

        # 0.5 box + 25 steps + 0.5 resample rl agent
        box_path_0_5 = "/media/neo/robotics/August/17-09-2021/train_rl_uniform_0.5_box_0.5/eval/events.out.tfevents.1633342690.pearl8.24069.1.v2"
        box_return_0_5 = np.array(getEventFileData(box_path_0_5)["Metrics/AverageReturn"])
        box_0_5 = ax.plot(box_return_0_5[:, 0], box_return_0_5[:, 1])

        ax.set_title('Average episode return for SAC agent with belief map (0.5 sampling box)', fontsize=18, weight='bold')
        ax.set_xlabel("number of eval epochs", fontsize=16)
        ax.set_ylabel("average episode return", fontsize=16)
        ax.legend([
                    "1.0 soft resample rate",
                    "0.5 soft resample rate"
                ], loc='upper right', fontsize=14)

        plt.show()
        fig.savefig("diff_resample_eval_avg_eps_return.png")

    elif plot == 'collision_penalty':
        # 0.5 box + 25 steps + 1.0 resample rl agent
        box_path_1_0 = "/media/neo/robotics/August/17-09-2021/train_rl_uniform_0.5_box_25/eval/events.out.tfevents.1631716010.pearl8.18818.1.v2"
        box_return_1_0 = np.array(getEventFileData(box_path_1_0)["Metrics/AverageStepCollisionPenality"])
        box_1_0 = ax.plot(box_return_1_0[:, 0], box_return_1_0[:, 1])

        # 0.5 box + 25 steps + 0.5 resample rl agent
        box_path_0_5 = "/media/neo/robotics/August/17-09-2021/train_rl_uniform_0.5_box_0.5/eval/events.out.tfevents.1633342690.pearl8.24069.1.v2"
        box_return_0_5 = np.array(getEventFileData(box_path_0_5)["Metrics/AverageStepCollisionPenality"])
        box_0_5 = ax.plot(box_return_0_5[:, 0], box_return_0_5[:, 1])

        ax.set_title('Average step collision penalty for SAC agent with belief map (0.5 sampling box)', fontsize=18, weight='bold')
        ax.set_xlabel("number of eval epochs", fontsize=16)
        ax.set_ylabel("average collision penalty (%)", fontsize=16)
        ax.legend([
                    "1.0 soft resample rate",
                    "0.5 soft resample rate"
                ], loc='upper right', fontsize=14)

        plt.show()
        fig.savefig("diff_resample_eval_avg_step_collision.png")

    elif plot == 'orientation_error':
        # 0.5 box + 25 steps + 1.0 resample rl agent
        box_path_1_0 = "/media/neo/robotics/August/17-09-2021/train_rl_uniform_0.5_box_25/eval/events.out.tfevents.1631716010.pearl8.18818.1.v2"
        box_return_1_0 = np.array(getEventFileData(box_path_1_0)["Metrics/AverageStepOrientationError"])
        box_1_0 = ax.plot(box_return_1_0[:, 0], box_return_1_0[:, 1])

        # 0.5 box + 25 steps + 0.5 resample rl agent
        box_path_0_5 = "/media/neo/robotics/August/17-09-2021/train_rl_uniform_0.5_box_0.5/eval/events.out.tfevents.1633342690.pearl8.24069.1.v2"
        box_return_0_5 = np.array(getEventFileData(box_path_0_5)["Metrics/AverageStepOrientationError"])
        box_0_5 = ax.plot(box_return_0_5[:, 0], box_return_0_5[:, 1])


        ax.set_title('Average step orientation error for SAC agent with belief map (0.5 sampling box)', fontsize=18, weight='bold')
        ax.set_xlabel("number of eval epochs", fontsize=16)
        ax.set_ylabel("average orientation error (radians)", fontsize=16)
        ax.legend([
                    "1.0 soft resample rate",
                    "0.5 soft resample rate"
                ], loc='upper right', fontsize=14)

        plt.show()
        fig.savefig("diff_resample_eval_avg_step_orientation.png")

    else:
        # 0.5 box + 25 steps + 1.0 resample rl agent
        box_path_1_0 = "/media/neo/robotics/August/17-09-2021/train_rl_uniform_0.5_box_25/eval/events.out.tfevents.1631716010.pearl8.18818.1.v2"
        box_return_1_0 = np.array(getEventFileData(box_path_1_0)["Metrics/AverageStepPositionError"])
        box_1_0 = ax.plot(box_return_1_0[:, 0], box_return_1_0[:, 1])

        # 0.5 box + 25 steps + 0.5 resample rl agent
        box_path_0_5 = "/media/neo/robotics/August/17-09-2021/train_rl_uniform_0.5_box_0.5/eval/events.out.tfevents.1633342690.pearl8.24069.1.v2"
        box_return_0_5 = np.array(getEventFileData(box_path_0_5)["Metrics/AverageStepPositionError"])
        box_0_5 = ax.plot(box_return_0_5[:, 0], box_return_0_5[:, 1])


        ax.set_title('Average step position error for SAC agent with belief map (0.5 sampling box)', fontsize=18, weight='bold')
        ax.set_xlabel("number of eval epochs", fontsize=16)
        ax.set_ylabel("average position error (meters)", fontsize=16)
        ax.legend([
                    "1.0 soft resample rate",
                    "0.5 soft resample rate"
                ], loc='upper right', fontsize=14)

        plt.show()
        fig.savefig("diff_resample_eval_avg_step_position.png")

def all_rl_eval_plts():
    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111)
    plot = 'collision_penalty'
    limit = 15

    if plot == 'collision_penalty':
        # 0.5 box + 25 steps rl agent
        box_path_0_5_25 = "/media/neo/robotics/August/17-09-2021/train_rl_uniform_0.5_box_25/eval/events.out.tfevents.1631716010.pearl8.18818.1.v2"
        box_0_5_mcp = np.array(getEventFileData(box_path_0_5_25)["Metrics/AverageStepCollisionPenality"])

        # 1.0 box + 50 steps rl agent
        box_path_1_0_50 = "/media/neo/robotics/August/17-09-2021/train_rl_uniform_1.0_box_50/eval/events.out.tfevents.1631961881.pearl2.22000.1.v2"
        box_1_0_mcp = np.array(getEventFileData(box_path_1_0_50)["Metrics/AverageStepCollisionPenality"])

        # 2.0 box + 50 steps rl agent
        box_path_2_0_50 = "/media/neo/robotics/August/17-09-2021/train_rl_uniform_2.0_box_50/eval/events.out.tfevents.1632144090.pearl2.5531.1.v2"
        box_2_0_mcp = np.array(getEventFileData(box_path_2_0_50)["Metrics/AverageStepCollisionPenality"])

        # 2.5 box + 50 steps rl agent
        box_path_2_5_50 = "/media/neo/robotics/August/17-09-2021/train_rl_uniform_2.5_box_50/eval/events.out.tfevents.1632349779.pearl8.24775.1.v2"
        box_2_5_mcp = np.array(getEventFileData(box_path_2_5_50)["Metrics/AverageStepCollisionPenality"])

        # full aprt + 50 steps rl agent
        box_path_50 = "/media/neo/robotics/August/17-09-2021/train_rl_uniform_likelihood_rnd_50_new/eval/events.out.tfevents.1633162880.pearl8.3776.1.v2"
        box_mcp = np.array(getEventFileData(box_path_50)["Metrics/AverageStepCollisionPenality"])

        data = np.concatenate([
                    box_0_5_mcp[:limit, 1:2],
                    box_1_0_mcp[:limit, 1:2],
                    box_2_0_mcp[:limit, 1:2],
                    box_2_5_mcp[:limit, 1:2],
                    box_mcp[:limit, 1:2]
                ], axis=1)
        ax.boxplot(data)

        ax.set_title('Episode mean collision penalty for different start pose sampling area', fontsize=22, weight='bold')
        # ax.set_xlabel("agent behavior", fontsize=16)
        ax.set_ylabel("mean collision penalty (%)", fontsize=18)
        # ax.set_ylim(-1.05, 0.05)
        ax.set_xticklabels([
                    "1.0 Sampling Box",
                    "2.0 Sampling Box",
                    "4.0 Sampling Box",
                    "5.0 Sampling Box",
                    "Full Apartment",
                ], fontsize=18)

        plt.show()
        fig.savefig("rl_belief_eval_mcp.png")

    elif plot == 'orientation_error':
        # 0.5 box + 25 steps rl agent
        box_path_0_5_25 = "/media/neo/robotics/August/17-09-2021/train_rl_uniform_0.5_box_25/eval/events.out.tfevents.1631716010.pearl8.18818.1.v2"
        box_0_5_mso = np.array(getEventFileData(box_path_0_5_25)["Metrics/AverageStepOrientationError"])

        # 1.0 box + 50 steps rl agent
        box_path_1_0_50 = "/media/neo/robotics/August/17-09-2021/train_rl_uniform_1.0_box_50/eval/events.out.tfevents.1631961881.pearl2.22000.1.v2"
        box_1_0_mso = np.array(getEventFileData(box_path_1_0_50)["Metrics/AverageStepOrientationError"])

        # 2.0 box + 50 steps rl agent
        box_path_2_0_50 = "/media/neo/robotics/August/17-09-2021/train_rl_uniform_2.0_box_50/eval/events.out.tfevents.1632144090.pearl2.5531.1.v2"
        box_2_0_mso = np.array(getEventFileData(box_path_2_0_50)["Metrics/AverageStepOrientationError"])

        # 2.5 box + 50 steps rl agent
        box_path_2_5_50 = "/media/neo/robotics/August/17-09-2021/train_rl_uniform_2.5_box_50/eval/events.out.tfevents.1632349779.pearl8.24775.1.v2"
        box_2_5_mso = np.array(getEventFileData(box_path_2_5_50)["Metrics/AverageStepOrientationError"])

        # full aprt + 50 steps rl agent
        box_path_50 = "/media/neo/robotics/August/17-09-2021/train_rl_uniform_likelihood_rnd_50_new/eval/events.out.tfevents.1633162880.pearl8.3776.1.v2"
        box_mso = np.array(getEventFileData(box_path_50)["Metrics/AverageStepOrientationError"])

        data = np.concatenate([
                    box_0_5_mso[:limit, 1:2],
                    box_1_0_mso[:limit, 1:2],
                    box_2_0_mso[:limit, 1:2],
                    box_2_5_mso[:limit, 1:2],
                    box_mso[:limit, 1:2]
                ], axis=1)
        ax.boxplot(data)

        ax.set_title('Episode mean orientation error for different start pose sampling area', fontsize=22, weight='bold')
        # ax.set_xlabel("agent behavior", fontsize=16)
        ax.set_ylabel("mean orientation error (radians)", fontsize=18)
        # ax.set_ylim(-0.05, 0.15)
        ax.set_xticklabels([
                    "1.0 Sampling Box",
                    "2.0 Sampling Box",
                    "4.0 Sampling Box",
                    "5.0 Sampling Box",
                    "Full Apartment",
                ], fontsize=18)

        plt.show()
        fig.savefig("rl_belief_eval_mso.png")

    elif plot == 'position_error':
        # 0.5 box + 25 steps rl agent
        box_path_0_5_25 = "/media/neo/robotics/August/17-09-2021/train_rl_uniform_0.5_box_25/eval/events.out.tfevents.1631716010.pearl8.18818.1.v2"
        box_0_5_msp = np.array(getEventFileData(box_path_0_5_25)["Metrics/AverageStepPositionError"])

        # 1.0 box + 50 steps rl agent
        box_path_1_0_50 = "/media/neo/robotics/August/17-09-2021/train_rl_uniform_1.0_box_50/eval/events.out.tfevents.1631961881.pearl2.22000.1.v2"
        box_1_0_msp = np.array(getEventFileData(box_path_1_0_50)["Metrics/AverageStepPositionError"])

        # 2.0 box + 50 steps rl agent
        box_path_2_0_50 = "/media/neo/robotics/August/17-09-2021/train_rl_uniform_2.0_box_50/eval/events.out.tfevents.1632144090.pearl2.5531.1.v2"
        box_2_0_msp = np.array(getEventFileData(box_path_2_0_50)["Metrics/AverageStepPositionError"])

        # 2.5 box + 50 steps rl agent
        box_path_2_5_50 = "/media/neo/robotics/August/17-09-2021/train_rl_uniform_2.5_box_50/eval/events.out.tfevents.1632349779.pearl8.24775.1.v2"
        box_2_5_msp = np.array(getEventFileData(box_path_2_5_50)["Metrics/AverageStepPositionError"])

        # full aprt + 50 steps rl agent
        box_path_50 = "/media/neo/robotics/August/17-09-2021/train_rl_uniform_likelihood_rnd_50_new/eval/events.out.tfevents.1633162880.pearl8.3776.1.v2"
        box_msp = np.array(getEventFileData(box_path_50)["Metrics/AverageStepPositionError"])

        data = np.concatenate([
                    box_0_5_msp[:limit, 1:2],
                    box_1_0_msp[:limit, 1:2],
                    box_2_0_msp[:limit, 1:2],
                    box_2_5_msp[:limit, 1:2],
                    box_msp[:limit, 1:2]
                ], axis=1)
        ax.boxplot(data)

        ax.set_title('Episode mean position error for different start pose sampling area', fontsize=22, weight='bold')
        # ax.set_xlabel("agent behavior", fontsize=16)
        ax.set_ylabel("mean position error (meters)", fontsize=18)
        # ax.set_ylim(-1.05, 0.05)
        ax.set_xticklabels([
                    "1.0 Sampling Box",
                    "2.0 Sampling Box",
                    "4.0 Sampling Box",
                    "5.0 Sampling Box",
                    "Full Apartment",
                ], fontsize=18)

        plt.show()
        fig.savefig("rl_belief_eval_msp.png")

    else:
        # 0.5 box + 25 steps rl agent
        box_path_0_5_25 = "/media/neo/robotics/August/17-09-2021/train_rl_uniform_0.5_box_25/eval/events.out.tfevents.1631716010.pearl8.18818.1.v2"
        box_0_5_mer = np.array(getEventFileData(box_path_0_5_25)["Metrics/AverageReturn"])

        # 1.0 box + 50 steps rl agent
        box_path_1_0_50 = "/media/neo/robotics/August/17-09-2021/train_rl_uniform_1.0_box_50/eval/events.out.tfevents.1631961881.pearl2.22000.1.v2"
        box_1_0_mer = np.array(getEventFileData(box_path_1_0_50)["Metrics/AverageReturn"])

        # 2.0 box + 50 steps rl agent
        box_path_2_0_50 = "/media/neo/robotics/August/17-09-2021/train_rl_uniform_2.0_box_50/eval/events.out.tfevents.1632144090.pearl2.5531.1.v2"
        box_2_0_mer = np.array(getEventFileData(box_path_2_0_50)["Metrics/AverageReturn"])

        # 2.5 box + 50 steps rl agent
        box_path_2_5_50 = "/media/neo/robotics/August/17-09-2021/train_rl_uniform_2.5_box_50/eval/events.out.tfevents.1632349779.pearl8.24775.1.v2"
        box_2_5_mer = np.array(getEventFileData(box_path_2_5_50)["Metrics/AverageReturn"])

        # full aprt + 50 steps rl agent
        box_path_50 = "/media/neo/robotics/August/17-09-2021/train_rl_uniform_likelihood_rnd_50_new/eval/events.out.tfevents.1633162880.pearl8.3776.1.v2"
        box_mer = np.array(getEventFileData(box_path_50)["Metrics/AverageReturn"])

        data = np.concatenate([
                    box_0_5_mer[:limit, 1:2],
                    box_1_0_mer[:limit, 1:2],
                    box_2_0_mer[:limit, 1:2],
                    box_2_5_mer[:limit, 1:2],
                    box_mer[:limit, 1:2]
                ], axis=1)
        ax.boxplot(data)

        ax.set_title('Episode mean return for different start pose sampling area', fontsize=22, weight='bold')
        # ax.set_xlabel("agent behavior", fontsize=16)
        ax.set_ylabel("return", fontsize=18)
        # ax.set_ylim(-1.05, 0.05)
        ax.set_xticklabels([
                    "1.0 Sampling Box",
                    "2.0 Sampling Box",
                    "4.0 Sampling Box",
                    "5.0 Sampling Box",
                    "Full Apartment",
                ], fontsize=18)

        plt.show()
        fig.savefig("rl_belief_eval_mer.png")

if __name__ == '__main__':
    # generalization_plts()
    # house3d_plts()
    # igibson_plts()
    # belief_plts()
    # rl_train_eval_plts()
    # rl_test_plts()
    # diff_steps_plts()
    diff_resample_plts()
    # all_rl_eval_plts()
