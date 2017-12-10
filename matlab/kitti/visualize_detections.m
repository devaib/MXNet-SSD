function visualize_detections()
    suffix2 = '_customized_central';
    % load gts of validation set
    layer_name = {'first','second','third','fourth','all'};
    gts_file = '../../data/kitti/results/imginfos_valset.txt';  % load ground truth from validation set
    dts_central_file = strcat('../../data/kitti/results/dts_',layer_name{2},'_layer', suffix2, '.txt');
    f = fopen(gts_file); gt_infos = textscan(f,'%s %d %d %d %d %f %d','delimiter',','); fclose(f);
    f = fopen(dts_central_file); dt_central_infos = textscan(f,'%s %d %d %d %d %f','delimiter',','); fclose(f);
    cropped_image_dir = '../../data/kitti/data_object_image_2/training/image_2_central/';

    imgindexs = dt_central_infos{1,1};
    xs = dt_central_infos{1,2};
    ys = dt_central_infos{1,3};
    ws = dt_central_infos{1,4};
    hs = dt_central_infos{1,5};
    scores = dt_central_infos{1,6};
    for i = 1 : size(imgindexs, 1)
        imgindex = imgindexs{i}
        imgpath = strcat(cropped_image_dir, imgindex, '.png');
        img = imread(imgpath); imshow(img); hold on;
        x = xs(i); y = ys(i); w = ws(i); h = hs(i); score = scores(i);
        pos = [x, y, w, h];
        rect = rectangle('Position', pos, 'EdgeColor', 'g', 'LineWidth', 2);
        waitforbuttonpress;
    end
    
end