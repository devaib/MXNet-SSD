This is a script to visualize both the standard annotations and out new
annotations side by side.

(1) You need to download and extract the video files from caltech before you
    can visualize the annotations. You can download them here:

    http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/datasets/USA/


(2) Start matlab and run the script. path_to_seq_files should point to the
    directory that contains the set00, set01, ... directories.

    >> visualize_annotations(path_to_seq_files)

    You can also specify specific sets to look at:
    >> visualize_annotations(path_to_seq_files, 6)

    You can also specify a specific video to look at:
    >> visualize_annotations(path_to_seq_files, 6, 2)


(3) You can cycle through the frames by pressing any button. You can stop by
    clicking the matlab window and pressing Ctrl-C.

