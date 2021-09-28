# deep-activate-localization

#### Verified Setup On:
* Operating System: Ubuntu 20.04.2 LTS
* Nvidia-Driver Version: 460.73.01
* CUDA Version: 11.2

#### Steps to setup project:
0. Install required Nvidia Driver + CUDA + CUDNN for the system. Also refer igibson documentation for system requirements [link](http://svl.stanford.edu/igibson/docs/installation.html).
1. Install virtual environment/package management platform like anaconda/[miniconda](https://docs.conda.io/en/latest/miniconda.html) or python virtualenv. Following assumes anaconda is installed.
2. Create conda environment
   ```
      $ conda create -y -n igibson python=3.7
      $ conda activate igibson
    ```
4. Setup iGibson2.0 environment (forked repository). For latest installation procedure always refer to official doc [link](http://svl.stanford.edu/igibson/docs/installation.html)
   ```
      $ git clone --branch master https://github.com/suresh-guttikonda/iGibson.git --recursive
      $ cd iGibson
      $ pip3 install -e .
   ```
6. Test installation is successful
   ```
      $ python
      >>> import igibson
   ```
7. Download required igibson's assets (robot's urdf, apartments, and others)
   ```
      $ python -m igibson.utils.assets_utils --download_assets
      $ python -m igibson.utils.assets_utils --download_demo_data
      $ python -m igibson.utils.assets_utils --download_dataset https://storage.googleapis.com/gibson_scenes/gibson_v2_4+.tar.gz
   ```
8. 

#### Steps to install singularity:
1. Enable NeuroDebian repository by following instructions on [neuro-debian](http://neuro.debian.net/). Also refer [FAQ](http://neuro.debian.net/faq.html).
2. Install Singularity-Container as follows:
   ```
   $ sudo apt-get install -y singularity-container
   ```
3. For more details, refer: [singularity guide](https://sylabs.io/guides/3.7/user-guide/index.html)

#### Steps to create singularity image to setup project:
0. We assume corresponding nvidia-driver is already installed in host machine.
1. Bootstrap container definition file. 'sudo' ensure proper privileges get assigned.
    ```
    # command takes few minutes to completely build image
    <path_to_def_file>$ sudo singularity build tensorflow_latest-gpu.sif tensorflow_latest-gpu.def
    ```
2. After successful build, verify that packages such as tensorflow+gpu are installed correctly.\
    Note: --nv ensure cuda related packages are loaded
    ```
    <path_to_sif_file>$ sudo singularity shell --nv tensorflow_latest-gpu.sif
    
    # get driver details if above assumptions are satisfied
    ~> nvidia-smi
    
    # activate virtualenv
    ~> source /opt/venvs/py3-igibson/bin/activate
    
    ~> python
    >>> import tensorflow as tf
    >>> tf.__version__
    >>>> tf.config.list_physical_devices() => returns list of available devices
    ```
3. Run demo igibson gui example. Following should open window with robot otherwise we see and error 'ERROR: Unable to initialize EGL'
    ```
    # Note: don't run with sudo once image is created
    <path_to_def_file>$ singularity shell --nv -B /usr/share/glvnd tensorflow_latest-gpu.sif
    
    ~> source /opt/venvs/py3-igibson/bin/activate
    
    # launch igibson gui example
    (py3-igibson)~> python -m gibson2.examples.demo.env_example
    ```
4. To Bind/Mount host machine directories within container use --bind [ref](https://sylabs.io/guides/3.0/user-guide/bind_paths_and_mounts.html)
    ```
    <path_to_sif_file>$ singularity shell --nv --bind /usr/share/glvnd,./src:/mnt/src tensorflow_latest-gpu.sif
    ```
