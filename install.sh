#!/bin/bash

cd ~
pwd

# Change to your name (last name and first name, lower case, for example jianqingzhu).
mkdir jianqingzhu     
cd ~/jianqingzhu
pwd

git clone git@github.com:jianqing666/MgNet_code.git

# Change the name of github repository, for example: MgNet_code 
cd ~/jianqingzhu/MgNet_code
pwd

# Prepare your conda env in requirements.yml
eval "$(conda shell.bash hook)"
conda env create -f requirements.yml -n jianqingzhu_env

