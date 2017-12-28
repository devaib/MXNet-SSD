clear; close all; clc;

imgname_list_file = '../../data/kitti/data_object_image_2/training/train_total.txt';
f = fopen(imgname_list_file); imgindex_list_cell = textscan(f, '%s'); fclose(f);
imgindex_list = imgindex_list_cell{1,1};
anno_dir = '../../data/kitti/data_object_label_2/training/label_2/';
small_object_anno_dir = '../../data/kitti/data_object_label_2/training/label_2_small/';
height_limit = 40;

for ind = 1:size(imgindex_list, 1)
    if mod(ind, 20) == 0
        fprintf('Processing %d images...\n', ind);
    end
    
    imgindex = imgindex_list{ind};
    % load annotation
    anno_file = strcat(anno_dir, imgindex, '.txt');
    f = fopen(anno_file); annos = textscan(f,'%s %f %f %f %f %f %f %f %f %f %f %f %f %f %f','delimiter',' '); fclose(f);
    
    removed_ind = [];
    for i = 1:size(annos{1,1},1)
        x = annos{1,5}(i);
        y = annos{1,6}(i);
        h = annos{1,8}(i)-annos{1,6}(i);
        if h > height_limit
            removed_ind = [removed_ind, 1];
        else
            removed_ind = [removed_ind, 0];
        end
    end

    % save image and annotation file
    fileID = fopen(strcat(small_object_anno_dir, imgindex, '.txt'), 'w');
    for i = 1 : size(annos{1,1}, 1)
        if removed_ind(i) == 1
            continue
        end
        fprintf(fileID,'%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n', ...
        annos{1,1}{i}, annos{1,2}(i), annos{1,3}(i), annos{1,4}(i),...
        annos{1,5}(i), annos{1,6}(i), annos{1,7}(i), annos{1,8}(i), ...
        annos{1,9}(i), annos{1,10}(i), annos{1,11}(i), ...
        annos{1,12}(i), annos{1,13}(i), annos{1,14}(i), ...
        annos{1,15}(i));
    end
    fclose(fileID);
end

fprintf('Done\n');


