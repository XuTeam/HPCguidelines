# 0 What do you need to prepare?

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
In this repository, there is a file install.sh, which need to be modified and then added to your github repository. Check the detail in the file.

Note that: the above 3 steps only need to be done when you use the HPC systey at the first time.

# 4 A scipt for running
In this repository, there is a file run.sh, which pull your updated code, submit the jobs with multiple GPUs and push the results when the job is finished.
Note that: all the output files should be named with "*.out", otherwise they can not be pushed to your github repository.


