#!/bin/bash
edfDir=$1
dstDir=$2

for file in ${edfDir}/*.edf
do
	if [ -f $file ]
	then
		python3 ./integrated_demo_cnn.py --edf $file --dest ${dstDir} --gpu=$3
	fi
done