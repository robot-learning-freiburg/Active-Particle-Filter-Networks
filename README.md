# deep-activate-localization

#### Steps to Install Singularity:
1. Enable NeuroDebian repository by following instructions on [neuro-debian](http://neuro.debian.net/)
2. Install Singularity-Container as follows: \
    `$ sudo apt-get install -y singularity-container`
3. For more details, refer: [singulairty guide](https://sylabs.io/guides/3.7/user-guide/index.html)

#### Steps to create singularity image:
1. Bootstrap container definition file. \
    `<path_to_def_file>$ sudo singularity build tensorflow_latest-gpu.sif tensorflow_latest-gpu.def`
3. 
