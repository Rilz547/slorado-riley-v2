#!/bin/bash

echo "Jetson Startup Script"

echo "Going to root"
cd /home/riley

echo "Allocating 16GB of swap"
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

echo "Increasing Swappiness to 10"
sudo sysctl vm.swappiness=10

echo "Disabling ZRAM Config"
sudo systemctl disable nvzramconfig

echo "Entering the Virtual Environment"
source slorado_venv/bin/activate

echo "Changing Directory" 
cd /home/riley/slorado-riley-v2

echo "Exporting torch path"
export TORCH_PATH=/home/riley/slorado_venv/lib/python3.10/site-packages/torch

echo "Building the Project"
make clean; make -j6 cuda=1 jetson=1 zstd=1 cxx11_abi=1 LIBTORCH_DIR=$TORCH_PATH