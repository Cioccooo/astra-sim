#!/bin/bash

rm ./*.csv
rm output_files/*.txt
#rm output_files/LLM/*.txt

python training.py

cp -r output_files/ $dir_share
cp *.csv $dir_share
