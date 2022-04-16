# 1 Code and data preparation
## 1.1 Github for code
Please create a Github repository and put all your code in the repository. Invite Xinliang Liu(Github ID: xlliu2017) to the repository with pull and push access. 

## 1.2 Data 
### 1.2.1 Public dataset
torchvision has been installed, see https://pytorch.org/vision/stable/datasets.html
### 1.2.2 Private dataset
Prepare a data download code and put it in github repository.
### 1.2.3 Too large dataset
We need the help from the IT.


# 2 Conda Enviroment
On your local computer, activate the conda evniroment under which your code is running. Then use the following command to export your conda enviroment

conda env export > requirements.yaml

This will generate a file requirements.yaml. Add this file to github repository.

# 3 A script for setup 
In this repository, there is a file install.sh, which need to be modified and then added to your github repository. Check the detail in install.sh.

Note that: the above 3 steps only need to be done when you use the HPC systey at the first time.

# 4 A scipt for running
In this repository, there is a file run.sh, which pull your updated code, submit the jobs with multiple GPUs and push the results when the job is finished. Check the detail in run.sh.

All the output files should be named with "*.out", otherwise they can not be pushed to your github repository.

# Finally, your repository should include at least the following files:
1 Your code and data
2 Conda enviroment file: requirements.yaml
3 A modified install.sh
4 A modified run.sh

# Other things you may want to know

(1) We are using the latest CentOS 7.9, which is standard for HPC systems, which has GCC gcc version 4.8.5 with glibc-2.17-323. Use package competiable with the compiler

