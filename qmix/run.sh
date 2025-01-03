#!/bin/bash
time=` date +"%Y-%m-%d-%H-%M" `
nohup python main.py > results/$time &
