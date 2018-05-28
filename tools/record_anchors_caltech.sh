#!/usr/bin/env bash

cd ..
val_file="./data/caltech-pedestrian-dataset-converter/data/trainlist.txt"
i=1
while IFS= read -r imagename
do
  if [ $i -le 20000 ]; then
    echo processing "$i" image...
    echo imagename: "$imagename".png
    /home/binghao/software/anaconda2/bin/python ./demo_single.py --mode 2 --images ./data/caltech-pedestrian-dataset-converter/data/images/"$imagename".png
    mv /home/binghao/workspace/MXNet-SSD/matlab/caltechUSA/anchors_all/anchors.txt /home/binghao/workspace/MXNet-SSD/matlab/caltechUSA/anchors_all/"$imagename".txt
  fi
  ((++i))
done < "$val_file"
