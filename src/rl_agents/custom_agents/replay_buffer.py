#!/usr/bin/env python3

import reverb
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils


class ReplayBuffer(object):
    """
    Replay Buffer implementation based on reverb
    Stores experience data collected by Actors and consumed by learner during training
    """

    def __init__(self,
                 table_name='uniform_table',
                 replay_buffer_capacity=1000,
                 ):
        """
        Initialize

        :param table_name: str
            name of reverb table
        :param replay_buffer_capacity: int
            maximum size replay buffer is allowed to hold
        """

        table = reverb.Table(
            name=table_name,
            max_size=replay_buffer_capacity,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            rate_limiter=reverb.rate_limiters.MinSize(1),
        )

        self.__table_name = table_name
        self.__reverb_server = reverb.Server(
            tables=[table]
        )
        self.__reverb_replay = None
        self.rb_traj_observer = None

    def get_dataset(self,
                    collect_data_spec,
                    sequence_length=1,
                    batch_size=1,
                    ):
        """
        Instantiate replay buffer and return as dataset

        :param collect_data_spec:
        :param sequence_length: int
            number of time steps each sample consists / sequence of consecutive items
        :param batch_size: int
            number of items to return
        :return: tf.data.Dataset
            dataset that return entries from replay buffer
        """

        # buffer returns experience data as [batch_size, sequence_length, ....]
        self.__reverb_replay = reverb_replay_buffer.ReverbReplayBuffer(
            data_spec=collect_data_spec,
            table_name=self.__table_name,
            sequence_length=sequence_length,
            local_server=self.__reverb_server,
        )

        ## HACK: replay buffer is efficient if sequence_length == num_steps
        dataset = self.__reverb_replay.as_dataset(
            sample_batch_size=batch_size,
            num_steps=sequence_length
        ).prefetch(50)

        return dataset

    def get_rb_traj_observer(self,
                             sequence_length=1,
                             stride_length=1
                             ):
        """
        Instantiate observer for writing trajectory data to replay buffer

        :param sequence_length:
             number of time steps each sample consists / sequence of consecutive items
        :param stride_length:
            stride value for sliding window for overlapping sequences
        :return: reverb_utils.ReverbAddTrajectoryObserver
            observer instance
        """

        self.rb_traj_observer = reverb_utils.ReverbAddTrajectoryObserver(
            py_client=self.__reverb_replay.py_client,
            table_name=self.__table_name,
            sequence_length=sequence_length,
            stride_length=stride_length,
        )

        return self.rb_traj_observer


    def close(self):
        """
        Close observers and Terminate reverb replay buffer server
        :return:
        """

        if self.rb_traj_observer is not None:
            self.rb_traj_observer.close()
        if self.__reverb_server is not None:
            self.__reverb_server.stop()