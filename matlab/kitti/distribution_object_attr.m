% inputs
% attribute : which attribute of image to be explored, 'width' or 'ratio'
% nBin : number of bins in histogram
function distribution_object_attr(attribute, nBin)
imginfo_file = '../../data/kitti/results/imginfos.txt';

% load file
f = fopen(imginfo_file);
imginfos = textscan(f,'%s %d %d %d %d %f %d','delimiter',',');
fclose(f);

if strcmp(attribute, 'width')
    attr = imginfos{1,4};
elseif strcmp(attribute, 'ratio')
    attr = imginfos{1,6};
end

histogram(attr, nBin);
[N, edges] = histcounts(attr, nBin);
    
end