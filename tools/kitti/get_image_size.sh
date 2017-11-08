#!/bin/bash
#Should be placed in $KITTI/data_object_image_2/training/ folder
trainfile=$(find image_2/ -type f)
i=0
for f in $trainfile
do
	if (( $i % 100 ==0 ))
	then
		echo "Processing $i images..."
	fi
	width=$(identify -format '%w' $f)
	height=$(identify -format '%h' $f)

	filename=${f%.*}
	filename=${filename##*/}

	echo "$width $height" >> image_size/$filename.txt
	i=$((i+1))
done
