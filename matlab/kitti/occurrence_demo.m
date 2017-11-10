clear; close all;
disp('===== Object Occurrence In KITTI Demo =====');

% setup
root_dir = '/home/binghao/data/kitti';
data_set = 'training';
cam = 2;
image_dir = fullfile(root_dir,['/data_object_image_2/' data_set '/image_' num2str(cam)]);
label_dir = fullfile(root_dir,['/data_object_label_2/' data_set '/label_' num2str(cam)]);
calib_dir = fullfile(root_dir,['/data_object_calib/' data_set '/calib']);

nimages = length(dir(fullfile(image_dir, '*.png')));
h = visualization('my_init',image_dir);

img_idx = 0;
grid_on = -1;
while 1
    objects = readLabels(label_dir, img_idx);
    
    drawActivatedGrids(image_dir, img_idx, objects);
    visualization('my_update', image_dir, h, img_idx, nimages, data_set, grid_on);


    for obj_idx = 1:numel(objects)
        drawBox2D(h, objects(obj_idx));
    end
    
    waitforbuttonpress; 
    key = get(gcf,'CurrentCharacter');
    switch lower(key)                         
        case 'q',  break;                                 % quit
        case '-',  img_idx = max(img_idx-1,  0);          % previous frame
        case 'x',  img_idx = min(img_idx+1000,nimages-1); % +100 frames
        case 'y',  img_idx = max(img_idx-1000,0);         % -100 frames
        otherwise, img_idx = min(img_idx+1,  nimages-1);  % next frame
    end
end

close all;
