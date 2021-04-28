# deep-activate-localization

#### Steps to Install Singularity:
1. Enable NeuroDebian repository by following instructions on [neuro-debian](http://neuro.debian.net/)
2. Install Singularity-Container as follows: \
    `$ sudo apt-get install -y singularity-container`
3. For more details, refer: [singulairty guide](https://sylabs.io/guides/3.7/user-guide/index.html)

#### Steps to create singularity image:
0. We assume corresponding nvidia-driver is already installed in host machine.
1. Bootstrap container definition file. \
    `<path_to_def_file>$ sudo singularity build tensorflow_latest-gpu.sif tensorflow_latest-gpu.def` => command takes few minutes to complete
2. After successfuly build, verify that packages such as tensorflow+gpu, miniconda is installed correctly. \
    `<path_to_sif_file>$ sudo singularity shell --nv tensorflow_latest-gpu.sif`
  Note: --nv ensure cuda related packages are loaded \
    `~>nvidia-smi` => returns driver details if above assumptions are satisfied \
    `~>python` \
    `>>>import tensorflow as tf` \
    `>>>tf.__version__` \
    `>>>>tf.config.list_physical_devices()` => returns list of available devices
3. To Bind directories on host machine to directories within container use --bind [ref](https://sylabs.io/guides/3.0/user-guide/bind_paths_and_mounts.html)\
    `<path_to_sif_file>$ sudo singularity shell --nv --bind ./src:/mnt/src tensorflow_latest-gpu.sif`
