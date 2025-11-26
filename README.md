[![CC BY 4.0][cc-by-shield]][cc-by]
# Master Thesis Code
## About
### Attribution
This code is based partially on the previous work by Alexander Ortlieb in his masters thesis (see git submodules in `AlexanderOrtlieb` directory)

### Directory Structure 
- `xray_fov`: Python package
- `notebooks`: Jupyter notebooks using `xray_fov`
- `AlexanderOrtlieb`: previous work
- `data`: the data used for experiments / results of experiments
- `miniconda3` / `venvs`: Installation of virtual environment

### Authorship
- All notebooks in this repository are written by Maximilian Glumann, for his master thesis
- The source files in the `xray_fov` package are attributed individually whether they were initally written by Alexander Ortlieb, written by Maximilian Glumann, or initially written by Alexander Ortlieb and significantly modified by Maximilian Glumann.

## Installation / Setup
### Data
See master thesis for details

### Dependencies
- are specified by the python package in `xray_fov`

OR

- are listed in `requirements.txt` in `xray_fov`

OR

- are automatically installed by `conda_venv_setup.sh`

### Environment Setup Script
- run `conda_venv_setup.sh` in this directory (depends on scripts in xray_fov directory) if no miniconda installation is present before.

OR

- Adapt `conda_venv_setup.sh` and paths to the local system / run commands manually

### Convenience Functions for bash (OPTIONAL)
Add the following to .bashrc or similar for convenience
```bash
condavenv(){
  conda activate $1
  source $HOME/master-thesis/venvs/$1/bin/activate 
}
condavenv_deactivate(){
  deactivate
  conda deactivate
}
ma_lab(){
  condavenv ma
  jupyter lab
  condavenv_deactivate
}
```

## License
This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: https://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg