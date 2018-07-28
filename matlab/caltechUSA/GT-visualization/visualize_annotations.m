function visualize_annotations(path_to_seq_files, set_n, video_n)
% Shows standard caltech annotations and new annotations of ICCV 2015
% submission #1624.
%
% Arguments:
%     path_to_seq_files    path to the directory containing the set
%                          directories with the video frames in seq format
%     set_n                specify a specific set number to watch videos in
%                          that set
%     video_n              specify a specific video number to watch it
%

  % check if arguments are correct
  if nargin < 1 || ...
      (nargin < 2 && ~exist(fullfile(path_to_seq_files, 'set00', 'V000.seq'), 'file'))
    fprintf('\n\nERROR: You need to supply the path to the videos directory of the caltech data,\nwhich contains the caltech video frames.\n\n');
    fprintf('You can download it at\n  http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/datasets/USA/\n\n');
    return;
  end
  
  load('caltech_layout.mat', 'caltech_layout');
  if nargin >= 2 && ~any(caltech_layout(:,1) == set_n)
    fprintf('\n\nERROR: set %d does not exist, caltech has only set %d - %d (including)\n\n', ...
      set_n, min(caltech_layout(:,1)), max(caltech_layout(:,1)));
  end
  if nargin >= 3
    set_videos = caltech_layout(caltech_layout(:,1) == set_n,2);
    if ~any(set_videos == video_n)
      fprintf('\n\nERROR: set %d does not have video %d, set %d has only video %d - %d (including)\n\n', ...
        set_n, video_n, set_n, min(set_videos), max(set_videos));
    end
  end
  
  % add Piotrs toolbox
  addpath(genpath('toolbox'));
  
  if nargin < 2
    % if no sets and videos are provided, let's show all of them
    fprintf('showing caltech videos, starting with training set\n');
    fprintf('If you want to have a look at specific sets or videos, you can\nspecify set ids and video ids as arguments to the function\n');
    fprintf('example: visualize_annotations(''path_to_seqs'', 6);\n');
    fprintf('Training sets: 0 - 5, test sets: 6 - 10\n');
    videos_to_watch = caltech_layout;
  elseif nargin < 3
    % set specified but no video, let's show the whole set
    fprintf('showing set %d, all videos\n', set_n);
    videos_to_watch = caltech_layout(caltech_layout(:,1) == set_n,:);
  else
    videos_to_watch = [set_n, video_n];
  end

  standard = load('gt-standard.mat', 'gt');
  new = load('gt-new.mat', 'gt');
  
  fprintf('\n\nStarting to show annotations frame by frame. Please press a button\nto go to the next frame. Press Ctrl-C in the matlab window to stop.\n\n');
  fh = figure;
%   finishup = onCleanup(@() close(fh));
  for i = 1:size(videos_to_watch,1)
    set_id = videos_to_watch(i,1);
    video_id = videos_to_watch(i,2);
    seq_file = fullfile(path_to_seq_files, sprintf('set%02d', set_id), ...
      sprintf('V%03d.seq', video_id));
    fprintf('set %02d, video %03d: %s\n', set_id, video_id, seq_file);
    if ~exist(fullfile(path_to_seq_files, 'set00', 'V000.seq'), 'file')
      fprintf('\n\nERROR: Could not find seq file. Either the path_to_seq_files is wrong or you''re missing the video files.\n\n');
      fprintf('You can download them at\n  http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/datasets/USA/\n\n');
      return;
    end
    
    std_video_gt = get_video_gt(standard.gt, set_id, video_id, caltech_layout);
    new_video_gt = get_video_gt(new.gt, set_id, video_id, caltech_layout);
    
    show_video(fh, seq_file, std_video_gt, new_video_gt);
  end
  
  fprintf('\ndone, all is well\n');
end


function video_gt = get_video_gt(gt, set_id, video_id, caltech_layout)
  idx = find((caltech_layout(:,1) == set_id) & (caltech_layout(:,2) == video_id));
  assert(numel(idx) == 1);
  first = sum(caltech_layout(1:(idx-1),3)) + 1;
  num = caltech_layout(idx,3);
  
  video_gt = gt(first:(first+num-1));
end


function show_video(fh, seq_file, std_gt, new_gt)
colors = [228,26,28
55,126,184
77,175,74
152,78,163
255,127,0
166,86,40]/255;

  sr = seqIo(seq_file, 'reader');
  finishup = onCleanup(@() sr.close());
  i = 0;
  delta = 30;
  while 1
      i = i + 1;
      frame = delta * i - 1;
      success = sr.seek(frame);
      if success == 0
        break;
      end
      [im, ~] = sr.getframe();
      if isempty(im)
        break;
      end
      
      figure(fh);
      subplot(1,2,1);
      imshow(im);
      title('standard ground truth');
      
      fa = std_gt{i};
      anno = fa(fa(:,5)==0,:);
      ignore = fa(fa(:,5)==1,:);
      draw_boxes(anno, colors(2,:), '-');
      draw_boxes(ignore, colors(2,:), '--');
      
      subplot(1,2,2);
      imshow(im);
      title('new ground truth');
      
      fa = new_gt{i};
      anno = fa(fa(:,5)==0,:);
      ignore = fa(fa(:,5)==1,:);
      draw_boxes(anno, colors(3,:), '-');
      draw_boxes(ignore, colors(3,:), '--');
      
      pause;
  end
end


function draw_boxes(boxes, color, style)
  if nargin < 3
    style = '-';
  end
  boxes(:,1:2) = boxes(:,1:2)+1; % matlab is 1-based
  for i = 1:size(boxes, 1)
    rectangle('Position', boxes(i,1:4), 'EdgeColor', color, ...
      'LineWidth', 2, ...
      'LineStyle', style);
  end
end