# Neural Abstractions

This repository serves as supplementary material for the corresponding paper Neural Abstractions, for the purpose of validating experimental results, and for using the codebase for continued development or research. This repository is maintained at [https://github.com/aleccedwards/neural-abstractions-nips22](https://github.com/aleccedwards/neural-abstractions-nips22), and the corresponding paper can be found at [https://openreview.net/forum?id=jF7u0APnGOv&noteId=dc6oF0HHC8](https://openreview.net/forum?id=jF7u0APnGOv&noteId=dc6oF0HHC8).

We do not provide this repository as a full repeatability package. Due to the certificiation step, the results are sensitive to initial random seeding of the network and may not be entirely reproducable. For this reason, we provide the trained models in the form of hybrid automatons as .xml model files, which can be provided to SpaceEx for the safety verification stage. These files are constructed from a certified abstraction, and can be used to repeat the corresponding safety verification using the procedure described in the sequel.

## Docker Installation

For your convenience we provide a Docker file. Docker can be installed by visiting <https://docs.docker.com/get-docker/>. Note that building the image may take a while due to downloading PyTorch. Within the project root directory, run:

```console
# docker build -t ubuntu:na .
# docker run --name na -it ubuntu:na bash
```

Here, the container is name `na` for simplicity.

You are now inside the container. Move to the project directory.

```console
cd /neural-abstractions
```

You are now able to run the program. The settings for a program are determined using a .yaml config file, the default location for which is `./config.yaml`. The used config file can be changed using the `-c` command line option. Note, the desired network structure must be passed using the command line option `-w` and is not settable from the config file.

### Install SpaceEx into the container

A registration is required to install SpaceEx, though it is available for free. It can be installed into the Docker container as follows.
On your host machine, go to <http://spaceex.imag.fr/download-6>, and download the SpaceEx command line executable v0.9.8f. This README assumes you have a 64 bit architecture.
Extract the corresponding archive to some location on your host machine. The extracted folder spaceex_exe must be copied to the container using using

```console
# docker cp /path/to/spaceex_exe na:/spaceex_exe`
```

## Running Experiments

Each experiment within Table 1 corresponds to a subdirectory within ./experiments named after the corresponding benchmark model. Each directory contains 4 shell scripts:

* A synthesis (synth) script which synthesises the neural abstraction and constructs the xml model file for SpaceEx.
* A SpaceEx script that performs the corresponding safety verification on the constructed abstraction
* An experiment script that performs both synth and safety verification together.
* A table-1-row script that also determines the runtime of the experiment.

Any of these scripts can be run as executables. For instance, to repeat the safety verification step for the Jet Engine benchmark in Table 1, run from the project directory

```console
cd experiments/jet
./jet-spaceex.sh
```

For convenience, the script `experiments/run_experiments.sh` will run all experiments in Table 1.

The experimental results from Table 2 in Appendix B can be re-obtained by running the experiments.sh script. Note, this script is currently set spawn up to 30 different processes in parallel. If you wish to repeat these experiments it is recommended that you either use a machine with this many cores, or reduce the total number of processes that can spawn concurrently.

The results themselves are in `results/results.csv` and `results/hybridisation-results.py`, and Table 2 can be recreated (in two parts) by running:

```console
python3 hybridisation.py
python3 resplot.py
```

which will create the files `table2.tex` and `table2-asm.text`.

We also retain the corresponding model files for Flow*, which can be repeated. This procedure is not part of this README.

## Citation

To cite this work, please use the following bibtex entry:

``` bibtex
@inproceedings{abate2022neural,
  title = {Neural Abstractions},
  booktitle = {Thirty-Sixth Conference on Neural Information Processing Systems},
  author = {Abate, Alessandro and Edwards, Alec and Giacobbe, Mirco},
  year = {2022}
}
```
