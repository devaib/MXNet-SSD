function drawActivatedGrids(image_dir, img_idx, objects)
    occ_col    = {'g','y','r','w'};
    trun_style = {'-','--'};
    img = imread(sprintf('%s/%06d.png',image_dir,img_idx));
    img(10:10:end, :, :) = 0;
    img(:, 10:10:end, :) = 0;
    figure(2);    imshow(img);
    
    % draw bounding boxes
    for obj_idx = 1:numel(objects)
        object = objects(obj_idx);
        if ~strcmp(object.type,'DontCare')
            pos = [object.x1,object.y1,object.x2-object.x1+1,object.y2-object.y1+1];
            trc = double(object.truncation>0.1)+1;
            rectangle('Position',pos,'EdgeColor',occ_col{object.occlusion+1},...
                    'LineWidth',2,'LineStyle',trun_style{trc})
            rectangle('Position',pos,'EdgeColor','b')
        end
    end
    
    grid_height = 1:10:size(img, 1);
    grid_height = [grid_height, size(img, 1)];
    grid_width  = 1:10:size(img, 2);
    grid_width  = [grid_width, size(img, 2)];
    num_height  = size(grid_height, 2);
    num_width   = size(grid_width, 2);
    
    % initialize grids
    for i = 1:num_height-1
        for j = 1:num_width-1
            grids(i, j).center_x  = (grid_width(j)+grid_width(j+1)) / 2;
            grids(i, j).center_y  = (grid_height(i)+grid_height(i+1)) / 2;
            grids(i, j).activated = 0;
        end
    end
    
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
                        hold on; plot(grid.center_x, grid.center_y, 'rx');
                    end
                end
            end
            
        end
    end
    
    
end
