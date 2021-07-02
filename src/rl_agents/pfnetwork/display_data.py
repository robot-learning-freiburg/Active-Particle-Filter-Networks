#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import preprocess, arguments

def display_data(params):
    """
    display data with the parsed arguments
    """

    # testing data
    test_ds = preprocess.get_dataflow(params.testfiles, params.batch_size, is_training=False)

    itr = test_ds.as_numpy_iterator()
    raw_record = next(itr)
    data_sample = preprocess.transform_raw_record(raw_record, params)

    b_idx = 5
    true_states = data_sample['true_states'][b_idx]
    global_map = data_sample['global_map'][b_idx]
    init_particles = data_sample['init_particles'][b_idx]
    org_map_shape = data_sample['org_map_shapes'][b_idx]
    org_map = global_map[:org_map_shape[0], :org_map_shape[1], :org_map_shape[2]]

    plt.imshow(org_map)
    x1, y1, heading = true_states[0]
    for t_idx in range(1, true_states.shape[0]):
        x2, y2, _ = true_states[t_idx]
        plt.arrow(x1, y1, (x2-x1), (y2-y1), head_width=5, head_length=7, fc='black', ec='black')
        x1, y1 = x2, y2

    # plt.show()
    plt.savefig("traj_output.png")

    print('display done')

if __name__ == '__main__':
    params = arguments.parse_args()

    display_data(params)
