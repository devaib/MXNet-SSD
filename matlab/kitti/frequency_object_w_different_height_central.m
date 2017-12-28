function frequency_object_w_different_height_central()
    layer_id = 1; % 1/2/3/4- show pairwise comparison between layer 1/2/3/4 and gts, 5 - show all
    suffix1 = '_customized';
    suffix2 = '_customized_central';
    % load gts of validation set
    layer_name = {'first','second','third','fourth','all'};
    layer_central = layer_name{layer_id};
    gts_file = '../../data/kitti/results/imginfos_valset.txt';  % load ground truth from validation set
    dts_file = strcat('../../data/kitti/results/dts_',layer_name{5},'_layer', suffix1, '.txt');     
    dts_central_file = strcat('../../data/kitti/results/dts_',layer_central,'_layer', suffix2, '.txt');
    f = fopen(gts_file); gt_infos = textscan(f,'%s %d %d %d %d %f %d','delimiter',','); fclose(f);
    f = fopen(dts_file); dt_infos = textscan(f,'%s %d %d %d %d %f','delimiter',','); fclose(f);
    f = fopen(dts_central_file); dt_central_infos = textscan(f,'%s %d %d %d %d %f','delimiter',','); fclose(f);
    
    gt_heights = gt_infos{1,5};
    dt_heights = dt_infos{1,5};
    dt_central_heights = dt_central_infos{1,5} ./ 2;        % bring back bbs to original scale
    [y_gt, b_gt] = hist(gt_heights, 0:10:400);
    [y_dt1, b_dt1] = hist(dt_heights, 0:10:400);
    [y_dt2, b_dt2] = hist(dt_central_heights, 0:10:400);
    
    y_dts = y_dt1;
    y_dts_central = y_dt2;
    assert(isequal(b_gt,b_dt1,b_dt2), 'bin dimension not match');
    
    bin = b_gt;
    b = bar(bin', [y_dts; y_dts_central; y_gt]', 'grouped');
    b(1).FaceColor = [0 0 .8];
    b(2).FaceColor = [0 .8 .8];
    b(3).FaceColor = [1 0 0];
    set(gcf, 'Position', [100, 500, 1000, 500])
    xlim([0 400]);
    xlabel('object height');
    ylabel('num of object');
    legend(b, {'detections from SSD',sprintf('detections from SSD central %s layer', layer_central), 'ground truth'});
    title('dts vs gts');
end