% -----------------------------------------------------------------------
%  This function shows an example how to use function 
%  error_cluster_analysis.
%

% Reference videos: decoded video files with no errors
for i=1:20
    video_ref_files{i} = sprintf('f:\\Videos\\ref_%02d.yuv',i);
    video_test_files{i} = sprintf('f:\\Videos\\test_%02d.yuv',i);
end

% Resolution of the video
resolution = [1920 1080];

% Make a cell array with subjective result files (one file per test 
% person), in this case named as 'result_XX.txt'. The format of the file:
%
% seq frame x y
%
% where seq is the sequence number of the video file, frame is the
% sequence number of the frame that was tapped, and x and y define the
% spatial coordinates where the screen was tapped
%
subj_res_files = {};
for i=1:20
    subj_res_files{i} = sprintf('f:\\Videos\\sim2result_%02d.txt',i);
end

% Open a feature file that can be used in further analysis
fid = fopen('f:\\Videos\\features.csv','w+');

% Loop through all the 
for seq = 0:length(video_ref_files)-1
    
    % Find the number of frames, assuming YUV420 format
    file_stats = dir(video_test_files{seq+1});
    num_frames = floor(file_stats.bytes/resolution(1)/resolution(2)/1.5);
   
    % Use the main function to determine error clusters and assign
    % taps to clusters to find subjective visibilities of clusters
    features = error_cluster_analysis(video_ref_files{seq+1}, ...
                                      video_test_files{seq+1}, ...
                                      [0 num_frames-1], resolution, ...
                                      subj_res_files, seq);
                                  
    % Finally, write features into a csv file for later analysis
    for q=1:length(features(:,1))
        fprintf(fid, '%0.5f', features(q,1));
        for j=2:length(features(q,:))
            fprintf(fid, ',%0.5f', features(q,j));
        end
        fprintf(fid, '\n');
    end    
end

fclose(fid);
                                      