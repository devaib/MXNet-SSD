#!/bin/bash
#Should be placed in $KITTI/data_object_image_2/training/ folder
trainfile=$(find image_2/ -type f | shuf)
i=0
for f in $trainfile
do
	filename=${f%.*}
	filename=${filename##*/}
	if (( $i%5 == 0 ));
	then
		echo $filename >> val.txt
	else
		echo $filename >> train.txt
	fi
	i=$((i+1))
done

