clear; close all; clc;

imgname_list_file = '../../data/kitti/data_object_image_2/training/train_total.txt';
f = fopen(imgname_list_file); imgindex_list_cell = textscan(f, '%s'); fclose(f);
imgindex_list = imgindex_list_cell{1,1};
image_dir = '../../data/kitti/data_object_image_2/training/image_2/';
anno_dir = '../../data/kitti/data_object_label_2/training/label_2/';
cropped_image_dir = '../../data/kitti/data_object_image_2/training/image_2_central/';
cropped_anno_dir = '../../data/kitti/data_object_label_2/training/label_2_central/';

for ind = 1:size(imgindex_list, 1)
    if mod(ind, 20) == 0
        fprintf('Processing %d images...\n', ind);
    end
    
    imgindex = imgindex_list{ind};
    % load image
    imgname = strcat(image_dir, imgindex, '.png');
    % load annotation
    anno_file = strcat(anno_dir, imgindex, '.txt');
    f = fopen(anno_file); annos = textscan(f,'%s %f %f %f %f %f %f %f %f %f %f %f %f %f %f','delimiter',' '); fclose(f);

    img = imread(imgname);
    height = size(img,1); width = size(img,2);
    crop_x = round(width/4);
    crop_y = round(height/4);
    crop_w = round(width/2);
    crop_h = round(height/2);
    crop_width = crop_x; crop_height = crop_y;
%     subplot(2,1,1);imshow(imgname);
    occlude_level = annos{1,3};
    % bbs
    poss = [];  % [xmin, ymin, xmax, ymax]
    object_sizes = [];
    occlude_level_new = zeros(size(annos{1,3}));
    for i = 1:size(annos{1,1},1)
        x = annos{1,5}(i);
        y = annos{1,6}(i);
        w = annos{1,7}(i)-annos{1,5}(i);
        h = annos{1,8}(i)-annos{1,6}(i);
        pos = [x, y, w, h];
%         rect = rectangle('Position', pos, 'EdgeColor', 'g', 'LineWidth', 2);
        poss = [poss; x y x+w y+h];
        object_sizes = [object_sizes; w*h];
    end 

    cropped = imcrop(img, [crop_x crop_y crop_w crop_h]);
    resized = imresize(cropped, [height width]);
%     subplot(2,1,2);imshow(resized);

    cropped_bound_xmin = 0; cropped_bound_ymin = 0;
    cropped_bound_xmax = crop_w; cropped_bound_ymax = crop_h;
    % crop bb outside boundaries [xmin ymin xmax ymax]
    poss_cropped = [poss(:,1)-crop_x, poss(:,2)-crop_y, poss(:,3)-crop_x, poss(:,4)-crop_y];
    poss_cropped_w_bound = [];
    poss_cropped_w_bound(:,1) = max(cropped_bound_xmin, poss_cropped(:,1));     % crop bb outside left edge
    poss_cropped_w_bound(:,2) = max(cropped_bound_ymin, poss_cropped(:,2));     % crop bb outside top edge
    poss_cropped_w_bound(:,3) = min(cropped_bound_xmax, poss_cropped(:,3));     % crop bb outside right edge
    poss_cropped_w_bound(:,4) = min(cropped_bound_ymax, poss_cropped(:,4));     % crop bb outside bottom edge

    % centralize to [center_x, center_y, w/2, h/2]
    poss_cropped_w_bound_centered = [];
    poss_cropped_w_bound_centered(:,1) = (poss_cropped_w_bound(:,1) + poss_cropped_w_bound(:,3)) / 2;
    poss_cropped_w_bound_centered(:,2) = (poss_cropped_w_bound(:,2) + poss_cropped_w_bound(:,4)) / 2;
    poss_cropped_w_bound_centered(:,3) = (poss_cropped_w_bound(:,3) - poss_cropped_w_bound(:,1)) / 2;
    poss_cropped_w_bound_centered(:,4) = (poss_cropped_w_bound(:,4) - poss_cropped_w_bound(:,2)) / 2;
    
    cropped_sizes = (poss_cropped_w_bound_centered(:,3)*2) .* (poss_cropped_w_bound_centered(:,4)*2);
    
    % regenerate the occlusion level
    shrink_ratio = cropped_sizes ./ object_sizes;
    for i = 1 : size(occlude_level, 1)
        if shrink_ratio(i) <= 0
            occlude_level_new(i) = -1;                      % out of boundary: Don't care
        elseif shrink_ratio(i) == 1
            occlude_level_new(i) = occlude_level(i);        % no shrink: keep original occlude level
        elseif shrink_ratio(i) > .7
            occlude_level_new(i) = occlude_level(i) + 1;    % .7 < ratio < 1: occlude level + 1
        elseif shrink_ratio(i) > .4
            occlude_level_new(i) = occlude_level(i) + 2;    % .4 < ratio < .7: occlude level + 2
        else
            occlude_level_new(i) = occlude_level(i) + 3;    % 0 < ratio < .4: occlude level + 3
        end  
    end
    occlude_level_new(occlude_level_new > 3) = 3;           % set max occlude level = 3
    
    % resize to original size [x, y, w, h]
    ratio = 2;
    poss_resized_centered = poss_cropped_w_bound_centered .* ratio;
    poss_resized = [];
    poss_resized(:,1) = poss_resized_centered(:,1) - poss_resized_centered(:,3);
    poss_resized(:,2) = poss_resized_centered(:,2) - poss_resized_centered(:,4);
    poss_resized(:,3) = poss_resized_centered(:,3) .* 2;
    poss_resized(:,4) = poss_resized_centered(:,4) .* 2;

    % remove bbs with width or height < 0
    removed_ind1 = poss_resized(:,3) <= 0;
    removed_ind2 = poss_resized(:,4) <= 0;
    removed_ind = removed_ind1 | removed_ind2;
    poss_resized(removed_ind, :) = [];

%     % visualize
%     for i = 1:size(poss_resized,1)
%         x = poss_resized(i,1);
%         y = poss_resized(i,2);
%         w = poss_resized(i,3);
%         h = poss_resized(i,4);
%         pos = [x, y, w, h];
%         rect = rectangle('Position', pos, 'EdgeColor', 'g', 'LineWidth', 2);
%     end

    % save image and annotation file
    imwrite(resized, strcat(cropped_image_dir, imgindex, '.png'));
    fileID = fopen(strcat(cropped_anno_dir, imgindex, '.txt'), 'w');
    j = 1;
    for i = 1 : size(annos{1,1}, 1)
        if removed_ind(i) == 1
            continue
        end
        fprintf(fileID,'%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n', ...
        annos{1,1}{i}, annos{1,2}(i), annos{1,3}(i), ...
        occlude_level_new(i), ...
        poss_resized(j,1), poss_resized(j,2), poss_resized(j,1)+poss_resized(j,3), poss_resized(j,2)+poss_resized(j,4), ...
        annos{1,9}(i), annos{1,10}(i), annos{1,11}(i), ...
        annos{1,12}(i), annos{1,13}(i), annos{1,14}(i), ...
        annos{1,15}(i));
        j = j + 1;
    end
    fclose(fileID);

end

fprintf('Done\n');




