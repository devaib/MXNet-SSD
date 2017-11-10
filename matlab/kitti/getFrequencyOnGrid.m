clear; close all;
disp('===== Getting Frequency on Grid =====');

root_dir = '/home/binghao/data/kitti/object_image_2';
data_set = 'training';
cam = 2;
image_dir = fullfile(root_dir,['/data_object_image_2/' data_set '/image_' num2str(cam)]);
label_dir = fullfile(root_dir,['/data_object_label_2/' data_set '/label_' num2str(cam)]);
calib_dir = fullfile(root_dir,['/data_object_calib/' data_set '/calib']);

nimages = length(dir(fullfile(image_dir, '*.png')));

% initialize grids
img_width   = 1242;
img_height  = 375;
grid_height = 1:10:img_height;
grid_height = [grid_height, img_height];
grid_width  = 1:10:img_width;
grid_width  = [grid_width, img_width];
num_height  = size(grid_height, 2);
num_width   = size(grid_width, 2);

for i = 1:num_height-1
    for j = 1:num_width-1
        grids(i, j).center_x  = (grid_width(j)+grid_width(j+1)) / 2;
        grids(i, j).center_y  = (grid_height(i)+grid_height(i+1)) / 2;
        grids(i, j).activated = 0;
    end
end
    

for img_idx = 0:nimages-1
    if (mod(img_idx, 50) == 0)
        fprintf('processing %d imagesfre\n', img_idx+1);
    end
    
    img = imread(sprintf('%s/%06d.png',image_dir,img_idx));
    objects = readLabels(label_dir, img_idx);
    

    % highlight grids within bounding boxes
    for obj_idx = 1:numel(objects)
        object = objects(obj_idx);
        if ~strcmp(object.type, 'DontCare')            
            % iterate over grids
            for i = 1:num_height-1
                for j = 1:num_width-1    
                    grid = grids(i, j);
                    if object.x1 <= grid.center_x && grid.center_x <= object.x2 && object.y1 <= grid.center_y && grid.center_y <= object.y2
                        grids(i,j).activated = grids(i,j).activated + 1;
                    end
                end
            end
            
        end
    end
 
end

freq_map = ones(size(grids));
for i = 1:size(grids, 1)
    for j = 1:size(grids, 2)
        freq_map(i, j) = grids(i, j);
    end
end

% heatmap

colormap('jet');
imagesc(freq_map);
daspect([1 1 1]);
colorbar;



