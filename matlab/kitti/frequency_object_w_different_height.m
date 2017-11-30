function frequency_object_w_different_height()
    % load gts of validation set
    gts_file = '../../data/kitti/results/imginfos_valset.txt';
    dts_file = '../../data/kitti/results/dts_first_layer.txt';
    
    f = fopen(gts_file);
    gt_infos = textscan(f,'%s %d %d %d %d %f %d','delimiter',',');
    fclose(f);
    f = fopen(dts_file);
    dt_infos = textscan(f,'%s %d %d %d %d %f','delimiter',',');
    fclose(f);
    
    gt_heights = gt_infos{1,5};
    dt_heights = dt_infos{1,5};
    [y_gt, b_gt] = hist(gt_heights, 0:10:400);
    [y_dt, b_dt] = hist(dt_heights, 0:10:400);
    assert(isequal(b_gt, b_dt), 'bin dimension not match');
    bin = b_gt;
    b = bar(bin', [y_dt; y_gt]', 'grouped');
    b(2).FaceColor = [1 0 0];
    b(1).FaceColor = [0 .5 .5];
    set(gcf, 'Position', [100, 500, 1000, 500])
end