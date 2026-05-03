#!/bin/bash

set -e

clear

echo "[INFO] Active Python3 Virtual Environments"
source venv/bin/activate

clear
echo "[INFO] Start to run Cifar100 program"
python3 src/cifar100.py

echo "[INFO] Done"