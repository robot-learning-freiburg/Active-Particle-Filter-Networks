import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import cv2

from environments.env_utils import render
from pfnetwork import pfnet
from environments.envs.localize_env import LocalizeGibsonEnv


def _draw_arrow(ax, robot_pose, color, alpha):
    y, x, heading = np.squeeze(robot_pose)

    heading_len = 3.2 * 1.0
    xdata = [x, x + heading_len * np.cos(heading)]
    ydata = [y, y + heading_len * np.sin(heading)]

    ax.plot(xdata, ydata, color=color, alpha=alpha, linewidth=1.8)[0]


class Trajectory:
    def __init__(self, scene_id, floor_num) -> None:
        self.scene_id = scene_id
        self.floor_num = floor_num
        self.observation = list()
        # self.floor_map = list()
        self.gt_pose = list()
        self.est_pose = list()
        self.likelihood_map = list()
        self.robot_position = list()
        self.robot_orientation = list()

    def add(self, observation, gt_pose, est_pose, likelihood_map, robot_position, robot_orientation):
        self.observation.append(observation)
        self.gt_pose.append(gt_pose)
        self.est_pose.append(est_pose)
        self.likelihood_map.append(likelihood_map)
        self.robot_position.append(robot_position)
        self.robot_orientation.append(robot_orientation)

    def get_figures(self):
        figures = []
        for i in range(len(self.observation)):
            f, axes = self.render_step(self.observation[i], self.likelihood_map[i], self.gt_pose[i], self.est_pose[i],
                                       prev_gt_poses=self.gt_pose[:i],
                                       prev_est_poses=self.est_pose[:i])

            canvas = FigureCanvasAgg(f)
            canvas.draw()
            plt_img = np.array(canvas.renderer._renderer)
            plt_img = cv2.cvtColor(plt_img, cv2.COLOR_RGB2BGR)
            figures.append(plt_img)
            plt.close(f)
        return figures

    def store_video(self, out_folder, episode_number):
        figures = self.get_figures()

        file_path = os.path.join(out_folder, f'episode_run_{episode_number}.mp4')
        LocalizeGibsonEnv.convert_imgs_to_video(figures, file_path)
        print(f'stored img results {len(figures)} eps steps to {file_path}')
        return figures

    @staticmethod
    def render_step(observation, likelihood_map, gt_pose, est_pose, prev_gt_poses, prev_est_poses):
        f, axes = plt.subplots(1, 5, figsize=(30, 7))
        [ax.set_axis_off() for ax in axes]

        # org_map_shape = likelihood_map.shape
        # render.draw_floor_map(0.025 * likelihood_map[..., 0] + likelihood_map[..., 1], org_map_shape, axes[0], None, cmap=None)
        axes[0].imshow(cv2.cvtColor(((1 - likelihood_map[..., 0]) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
        masked_data = np.ma.masked_where(likelihood_map[..., 1] == 0, likelihood_map[..., 1])
        axes[0].imshow(masked_data[..., np.newaxis])

        color = '#7B241C'
        position_plt, heading_plt = render.draw_robot_pose(
            gt_pose,
            color,
            likelihood_map.shape,
            axes[0],
            None,
            None,
            plt_path=True)
        for pose in prev_gt_poses:
            _draw_arrow(axes[0], pose, color=color, alpha=0.7)

        color = '#515A5A'
        position_plt, heading_plt = render.draw_robot_pose(
            est_pose,
            color,
            likelihood_map.shape,
            axes[0],
            None,
            heading_plt,
            plt_path=False)
        for pose in prev_est_poses:
            _draw_arrow(axes[0], pose, color=color, alpha=0.7)

        # axes[0].legend([position_plt,
        #                 heading_plt],
        #                 ["GT Pose", "Est Pose"], loc='upper left', fontsize=12)

        # remove any potential padding / empty space around it
        l = .025 * likelihood_map[..., 0] + likelihood_map[..., 1]
        bb = pfnet.PFCell.bounding_box(l != 0)
        ylim = [max(bb[0] - 5, 0), min(bb[1] + 5, l.shape[0])]
        xlim = [max(bb[2] - 5, 0), min(bb[3] + 5, l.shape[0])]
        # make quadratic to fit plot nicely
        sz = max([ylim[1] - ylim[0], xlim[1] - xlim[0]])
        for lim in (ylim, xlim):
            diff = (lim[1] - lim[0]) // 2
            mid = lim[0] + diff
            lim[0], lim[1] = mid - sz // 2, mid + sz // 2
        axes[0].set_ylim(ylim)
        axes[0].set_xlim(xlim)

        # axes[1].imshow(0.025 * observation['likelihood_map'][..., 0] + observation['likelihood_map'][..., 1])
        axes[1].imshow(
            cv2.cvtColor((((1 - observation['likelihood_map'][..., 0])) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
        masked_data = np.ma.masked_where(observation['likelihood_map'][..., 1] == 0,
                                         observation['likelihood_map'][..., 1])
        axes[1].imshow(masked_data[..., np.newaxis])

        g = np.rot90(observation['occupancy_grid'], 1)
        g = cv2.cvtColor((g * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        axes[2].imshow(g)
        axes[3].imshow(observation['rgb_obs'])
        axes[4].imshow(observation['depth_obs'])

        f.tight_layout()

        return f, axes
