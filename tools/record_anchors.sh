#!/usr/bin/env bash

cd ..
val_file="./data/kitti/data_object_image_2/training/val.txt"
i=1
while IFS= read -r imagename
do
    echo processing "$i" image...
    echo imagename: "$imagename".png
    /home/binghao/software/anaconda2/bin/python ./demo.py --mode 2 --images ./data/kitti/data_object_image_2/training/image_2/"$imagename".png
    mv /home/binghao/workspace/MXNet-SSD/matlab/kitti/anchors_customized_outputs/anchors.txt /home/binghao/workspace/MXNet-SSD/matlab/kitti/anchors_customized_outputs/"$imagename".txt
    ((++i))
done < "$val_file"


