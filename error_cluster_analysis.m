% -----------------------------------------------------------------------
%  This function performs full detection of error clusters and and assigns
%  taps to error clusters, as described in publication:
%
%  J. Korhonen, "Study of the Subjective Visibility of Packet Loss 
%  Artifacts in Decoded Video Sequences," IEEE Transactions on 
%  Broadcasting, June 2018.
%
%  Input: 
%         ref_video:    Path to the decoded reference video (in YUV 
%                       format) without packet loss artifacts, for
%                       example: 'c:\videos\decoded.yuv'
%         test_video:   Path to the decoded reference video (in YUV 
%                       format) without packet loss artifacts, for
%                       example: 'c:\videos\decoded_errors.yuv'
%         fr_range:     Range of frames to be analyzed (e.g. [0,200])
%         reso:         Resolution of yuv video, e.g. [1920 1080]
%         res_files:    Results of the subjective study, one file per
%                       test user, for example {'c:\res\user01.txt',
%                        'c:\res\user02.txt','c:\res\user03.txt'}
%         seq:          Sequence number used in the result files to
%                       distinguish different video clips. See the
%                       error_cluster_analysis_example script.
%
%  Output:
%         features:     Matrix containing the features of error clusters.
%                       Each row corresponds to different error cluster.
%                       The first column contains the subjective
%                       visibility (normalized to [0,1]). Other columns
%                       represent different features, as described in the
%                       paper.
%
function features = error_cluster_analysis(ref_video, test_video, ...
                                           fr_range, reso, res_files, seq)
                                  
    % Use full-reference analysis to find error clusters
    fr_errmap = find_error_clusters(ref_video,test_video, fr_range, reso);
    
    % Assign taps to error clusters
    detections = assign_taps_to_clusters(fr_errmap, res_files, seq);

    % Then, we can find the features related to each error cluster 
    features = get_error_cluster_features(ref_video, test_video, reso, ...
                                          fr_range, fr_errmap);
    
    % Combine subjective visibility and features in the same matrix
    features = [detections(:,2) features];
end

% -----------------------------------------------------------------------
% This function is used to find find error clusters and return the
% error cluster map fr_errmap of size (by,bx,t) where by is the video
% height in macroblocks, bx is the video width in macroblocks, t is
% the length in frames and each element is assigned with an error cluster
% identifier
%
function fr_err_map = find_error_clusters(ref_video, ... 
                                         test_video, ... 
                                         fr_range, reso)

    % Initialize variables
    %                                         
    width=reso(1); 
    height=reso(2);
    b_width = floor(width/16);
    b_height = floor(height/16);
    fr_err_map = zeros(b_height,b_width,fr_range(2)-fr_range(1));
    
    %flg = 0;
    %err_flag_map = zeros(b_height,b_width,10);
    num_clusters = 0;
    
    % Loop through all the frames in the sequence
    % (make sure that fr_range does not exceed YUV file size!)
    %
    for fr=fr_range(1):fr_range(2)
        
        % Read test and reference frames for full reference analysis
        test_fr = YUVread(test_video,[width height],fr);
        ref_fr = YUVread(ref_video,[width height],fr);
        
        % Initialize error cluster map and find if there are error
        % clusters that will be transferred from the previous frame
        init_clust_map = zeros(b_height,b_width);
        if fr>fr_range(1)
            for y=1:b_height
                for x=1:b_width
                    init_clust_map(y,x)=fr_err_map(y,x,fr-fr_range(1));
                end 
            end
        end        
        
        % Compute new error map with cluster indices
        fr_err_map(:,:,fr-fr_range(1)+1) = ...
            fullref_comparison(test_fr,ref_fr,init_clust_map,num_clusters);
        
        num_clusters = max(max(max(fr_err_map(:,:,:))));
    end

    % Now we have the error clusters, but there may be some error
    % cluster indices that are unused (since clusters were combined
    % during the run). So we need to remove the non-existing indices to
    % make sure that the indices run continuously from 1 to n.
    i=1;
    while i<num_clusters
        while isempty(find(fr_err_map==i))
            fr_err_map(fr_err_map>i)=fr_err_map(fr_err_map>i)-1;
            num_clusters=num_clusters-1;
        end
        i=i+1;
    end
end
 
% ----------------------------------------------------------------------
% This function makes full reference comparison between macroblocks in
% the test video and reference video and marks them as belonging to
% a error cluster (or not). Called by function find_error_clusters().
%
function clust_map = fullref_comparison(test_fr, ref_fr, ...
                                        init_clustmap, num_clusters)

    % This function compares target frame against reference frame
    % as output, value 0..1 is given for each macroblock
    % (0: totally visible difference, 1: not visible difference

    [height,width,depth] = size(test_fr);
    height = height - mod(height,16);
        
    b_width = floor(width/16);
    b_height = floor(height/16);

    errormap = zeros(b_height,b_width);
    clust_flags = zeros(b_height,b_width); 
    clust_map = init_clustmap;
    
    ref_sobel = sobelfilt(ref_fr(:,:,1));
    test_sobel = sobelfilt(test_fr(:,:,1));
    
    % First, find perceptual distortion index for each macroblock
    for y=1:16:height
        for x=1:16:width           
            testblock = test_fr(y:y+15,x:x+15,1);
            refblock = ref_fr(y:y+15,x:x+15,1);
            errormap(floor(y/16)+1,floor(x/16)+1) = 0;

            mse = sum(sum((testblock-refblock).^2))/(16*16);
            si_test = std2(test_sobel(3:14,3:14));
            si_ref = std2(ref_sobel(3:14,3:14));

            errormap(floor(y/16)+1,floor(x/16)+1) = ...
                perc_distortion(mse, min(si_test, si_ref));
        end
    end
    
    % Then, combine adjacent distorted macroblocks into error clusters
    for y=1:b_height
        for x=1:b_width           
            dstart_y = max(1,y-1);  dend_y = min(b_height,y+1); 
            d1start_x = max(1,x-1); d1end_x = min(b_width,x+1);
            d2start_x = max(1,x-2); d2end_x = min(b_width,x+2);
            d3start_x = max(1,x-3); d3end_x = min(b_width,x+3);

            % In the original Electronic Imaging paper
            %if dists1(1)>=0.001 || (mean2(dists1)>0.0001) || ...
            %(mean2(dists2)>0.0001) || (mean2(dists3)>0.0001)

            % Modified for the Transactions paper
            if mean2(errormap(dstart_y:dend_y,d3start_x:d3end_x))>0.1
                clust_flags(dstart_y:dend_y,d3start_x:d3end_x) = 1;
            elseif mean2(errormap(dstart_y:dend_y,d2start_x:d2end_x))>0.1
                clust_flags(dstart_y:dend_y,d2start_x:d2end_x) = 1;
            elseif errormap(y,x)>0.25
                clust_flags(dstart_y:dend_y,d1start_x:d1end_x) = 1;
            elseif mean2(errormap(dstart_y:dend_y,d1start_x:d1end_x))>0.1
                clust_flags(dstart_y:dend_y,d1start_x:d1end_x) = 1;
            end
        end
    end
    
    % Finally, adjust the index numbers for error clusters
    for y=1:b_height
        for x=1:b_width     
            if clust_flags(y,x) > 0 && clust_map(y,x) == 0
                num_clusters = num_clusters+1;
                [clust_map,clust_flags] = mark_area(y,x,num_clusters, ...
                                                    clust_map,clust_flags);
                              
                if max(max(clust_map))<num_clusters
                    num_clusters = num_clusters-1;
                end                
            elseif clust_flags(y,x) == 0 && clust_map(y,x) > 0
                clust_map(y,x) = 0;
            end
        end
    end   
end

% -------------------------------------------------------------------
% This function finds recursively if the adjacent macroblocks
% belong to the same cluster and mark them accordingly.
% Called by function fullref_comparison().
%
function [clust_map,clust_flags] = mark_area(y,x,marker, ...
                                             clust_map,clust_flags)
    
    [height,width]=size(clust_map);
    m=[];
    if y>1
        m=[m clust_map(y-1,x)]; 
    end
    if y<height
        m=[m clust_map(y+1,x)];
    end
    if x>1
        m=[m clust_map(y,x-1)];
    end
    if x<width
        m=[m clust_map(y,x+1)];
    end
    m = m(find(m~=marker));
    m = m(find(m>0));
    if ~isempty(m)
        m = [m marker];
        num_m = [];
        for z=1:length(m)
            num_m(z) = length(find(clust_map(:)==m(z)));
        end
        marker = m(find(num_m==max(num_m)));
        marker = marker(1);
    end
    clust_map(y,x)=marker;
    if y>1 && ((clust_flags(y-1,x)>0 && clust_map(y-1,x)==0) || ...
               (clust_map(y-1,x)~=0 && clust_map(y-1,x)~=marker))
        
        [clust_map,clust_flags] = mark_area(y-1,x,marker, ...
                                            clust_map, clust_flags);
    end
    if y<height && ((clust_flags(y+1,x)>0 && clust_map(y+1,x)==0) || ...
                    (clust_map(y+1,x)~=0 && clust_map(y+1,x)~=marker))
        
        [clust_map,clust_flags] = mark_area(y+1,x,marker, ...
                                            clust_map, clust_flags);
    end        
    if x>1 && (((clust_flags(y,x-1)>0 && clust_map(y,x-1)==0) || ...
          (clust_map(y,x-1)~=0 && clust_map(y,x-1)~=marker)) || ...
          (y>1&&((clust_flags(y-1,x-1)>0 && clust_map(y-1,x-1)==0)||...
          (clust_map(y-1,x-1)~=0 && clust_map(y-1,x-1)~=marker))) || ...
          y<height&&((clust_flags(y+1,x-1)>0 && clust_map(y+1,x-1)==0)||...
          (clust_map(y+1,x-1)~=0 && clust_map(y+1,x-1)~=marker)))              
        
      [clust_map,clust_flags] = mark_area(y,x-1,marker, ...
                                          clust_map, clust_flags);
    end
    if x<width && (((clust_flags(y,x+1)>0 && clust_map(y,x+1)==0) || ...
         (clust_map(y,x+1)~=0 && clust_map(y,x+1)~=marker)) || ...
         (y>1&&((clust_flags(y-1,x+1)>0 && clust_map(y-1,x+1)==0)||...
         (clust_map(y-1,x+1)~=0 && clust_map(y-1,x+1)~=marker))) || ...
         y<height&&((clust_flags(y+1,x+1)>0 && clust_map(y+1,x+1)==0)||...
         (clust_map(y+1,x+1)~=0 && clust_map(y+1,x+1)~=marker)))
     
        [clust_map,clust_flags] = mark_area(y,x+1,marker, ...
                                            clust_map, clust_flags);
    end    
end
    
% ---------------------------------------------------------------------
% This function computes the features related to each error cluster
%
function features = get_error_cluster_features(ref_video, test_video, ...
                                               reso, fr_range, fr_errmap)
                                                                               
    % Initialize variables
    width = reso(1);
    height = reso(2);
    [b_height,b_width,~] = size(fr_errmap);
    num_clusters = max(max(max(fr_errmap)));

    % Initialize features
    ts = zeros(num_clusters,1);                % temporal size (frames)
    ss = zeros(num_clusters,1);                % spatiotemporal size (MBs)
    sose = zeros(num_clusters,1);              % sum of squared errors
    max_perc_dists = zeros(num_clusters,1);    % maximum E_MB
    mean_perc_dists = zeros(num_clusters,1);   % mean E_MB
    median_perc_dists = zeros(num_clusters,1); % median E_MB
    perc_dists10pr =  zeros(num_clusters,1);   % mean of 10% highest E_MB
    perc_dists25pr =  zeros(num_clusters,1);   % mean of 25% highest E_MB
    perc_dists50pr =  zeros(num_clusters,1);   % mean of 50% highest E_MB
    si = zeros(num_clusters,1);                % spatial acitivity indices
    ti = zeros(num_clusters,1);                % spatial acitivity indices
    startfr = zeros(num_clusters,1);           % cluster start frame
    e_mb_list = zeros(num_clusters,1);         % list of E_MBs
    
    for fr=fr_range(1):fr_range(end)

        test_fr = YUVread(test_video,[width height],fr);
        ref_fr = YUVread(ref_video,[width height],fr);
        test_sob = sobelfilt(ref_fr(:,:,1));

        marked = [];
        for y=1:b_height
            for x=1:b_width
                nerr = fr_errmap(y,x,fr-fr_range(1)+1);
                sqerr = sum(sum((test_fr(y*16-15:y*16,x*16-15:x*16,1)-...
                        ref_fr(y*16-15:y*16,x*16-15:x*16,1)).^2))/(16*16);
                
                si_tar = std2(test_fr(y*16-15:y*16,x*16-15:x*16,1));
                si_org = std2(ref_fr(y*16-15:y*16,x*16-15:x*16,1));
                
                if nerr>0 && sqerr>0
                    if isempty(find(marked==nerr))
                        marked = [marked nerr];
                        ts(nerr)=ts(nerr)+1;
                    end
                    
                    sose(nerr) = sose(nerr)+sqerr;
                    pdist = perc_distortion(sqerr, min(si_tar, si_org));
             
                    [n_blks,pnumar] = size(e_mb_list);
                    ss(nerr) = ss(nerr)+1;
                    if n_blks < ss(nerr)
                        % add a new list of e_mbs
                        e_mb_list = [e_mb_list zeros(num_clusters,1)];
                    end
                    e_mb_list(nerr,ss(nerr)) = pdist;
                    
                    si(nerr) = si(nerr)+sum(sum(abs( ...
                                test_sob(y*16-15:y*16,x*16-15:x*16,1))));
                    if fr>fr_range(1)
                        ti(nerr) = ti(nerr)+sum(sum(abs(...
                               test_fr(y*16-15:y*16,x*16-15:x*16,1)- ...
                               ref_fr(y*16-15:y*16,x*16-15:x*16,1))));
                    end
                end
            end
        end
    end
    numerr_this = zeros(num_clusters,1);
    numerr_tot = zeros(num_clusters,1);    
    for nerr=1:num_clusters
        for fr=fr_range(1):fr_range(end)
            nerrors = length(find(fr_errmap(:,:,fr-fr_range(1)+1)==nerr));
            if nerrors>0 
                numerr_this(nerr) = numerr_this(nerr) + nerrors;
            end
            numerr_tot(nerr) = numerr_tot(nerr) + ...
                        length(find(fr_errmap(:,:,fr-fr_range(1)+1)>0));
            if startfr(nerr)==0
                startfr(nerr)=fr-fr_range(1)+1;
            end
        end
        
        if numerr_this(nerr) > 0 && ss(nerr)>0
            e_mb_list_temp = e_mb_list(nerr, 1:ss(nerr));
            e_mb_list_temp = sort(e_mb_list_temp, 'descend');
            max_perc_dists(nerr) = max(e_mb_list_temp);
            mean_perc_dists(nerr) = mean(e_mb_list(nerr,1:ss(nerr)));
            median_perc_dists(nerr) = median(e_mb_list(nerr,1:ss(nerr))); 
            perc_dists10pr(nerr) = mean(e_mb_list_temp(...
                                        1:ceil(ss(nerr)/10)));
            perc_dists25pr(nerr) = mean(e_mb_list_temp(...
                                        1:ceil(ss(nerr)/4)));
            perc_dists50pr(nerr) = mean(e_mb_list_temp(...
                                        1:ceil(ss(nerr)/2)));   
        end
    end
      
    % scale ti and si values according to the size of the cluster
    ti = ti./(ss*16*16); 
    si = si./(ss*16*16);
    features = [ts ss ts./ss numerr_this./numerr_tot ...
                10.*log10(ss./sose) 1./(10.*log10(ss./sose)) ...
                max_perc_dists mean_perc_dists median_perc_dists ...
                perc_dists10pr perc_dists25pr perc_dists50pr ...
                si ti ti./(si+0.0001)];
end

% -----------------------------------------------------------------------
% This function reads one frame from YUV420 file
%
function YUV = YUVread(fname,dim,frnum)

    f=fopen(fname,'r');
    fseek(f,dim(1)*dim(2)*1.5*frnum,'bof');
    Y=fread(f,dim(1)*dim(2),'uchar');
    Y=cast(reshape(Y,dim(1),dim(2)),'double')./255;
    U=fread(f,dim(1)*dim(2)/4,'uchar');
    U=cast(reshape(U,dim(1)/2,dim(2)/2),'double')./255;
    U=imresize(U,2.0);
    V=fread(f,dim(1)*dim(2)/4,'uchar');
    V=cast(reshape(V,dim(1)/2,dim(2)/2),'double')./255;
    V=imresize(V,2.0);
    YUV(:,:,1)=Y';
    YUV(:,:,2)=U';
    YUV(:,:,3)=V';
    fclose(f);
end

% -----------------------------------------------------------------------
% This function implements standard Sobel filter
%
function outblock = sobelfilt(imblock)

    imblock = cast(imblock,'double');
    h = [1 2 1; 0 0 0; -1 -2 -1]./8;
    outblock = sqrt(imfilter(imblock, h).^2+imfilter(imblock, h').^2);
end

% -----------------------------------------------------------------------
% This function computes perceptual distortion for a macroblock,
% see the paper for more details
%
function distortion = perc_distortion(mse, si)

    psnr = 10*log10(1/mse);
	distortion = 1-1./(1+exp(-37.0*si-0.06*psnr));

end

% -----------------------------------------------------------------------
% This function assigns taps from result files to error clusters,
% see the paper for more details
%
function detections = assign_taps_to_clusters(fr_errmap, res_files, seq)

    % Initialize variables
    [maxy,maxx,n_frames]=size(fr_errmap);
    num_files = length(res_files);
    clusts = max(max(max(fr_errmap)));
    detections = [(1:clusts)' zeros(clusts,1) zeros(clusts,1)];

    % Read the taps from the results files, file by file
    for file=1:num_files
        fid = fopen(res_files{file},'r');
        x = [];
        y = [];
        detections(:,3) = detections(:,3).*0;
        frame=[];
        while ~feof(fid)
            str=fgets(fid);
            vals=sscanf(str,'%d %d %d %d');
            if length(vals)==4
                % Only consider taps on this video sequence
                if vals(1)==seq
                    frame=[frame vals(2)];
                    x=[x vals(3)];
                    y=[y vals(4)];
                end
            end
        end

        % convert coordinates to macroblock domain
        x = floor(x./16)+1; 
        y = floor(y./16)+1; 

        % Make a 3D matrix with spatiotemporal weights
        % (in this version, weights are binary (0 or 1)
        temporal_weight = zeros(1,40);
        temporal_weight(11:40) = ones(1,30);

        spatial_weight=[ 0 0 1 1 1 0 0 
                         0 1 1 1 1 1 0
                         1 1 1 1 1 1 1
                         1 1 1 1 1 1 1
                         1 1 1 1 1 1 1
                         0 1 1 1 1 1 0
                         0 0 1 1 1 0 0 ];        
        
        for i=1:40
            weight(:,:,i)=temporal_weight(i).*spatial_weight;
        end  
        missed = 0;    
        
        % Loop through all the frames where user has tapped the screen
        if ~isempty(frame)
            for i=1:length(frame)
                missflag = 1;
                
                % Align weights with the video sequence
                xsshift = max(1,x(i)-3)-x(i);
                xeshift = min(maxx,x(i)+3)-x(i);
                ysshift = max(1,y(i)-3)-y(i);
                yeshift = min(maxy,y(i)+3)-y(i);
                tsshift = max(1,frame(i)-39)-frame(i);
                errors = fr_errmap(y(i)+ysshift:y(i)+yeshift,x(i)+...
                         xsshift:x(i)+xeshift,frame(i)+tsshift:frame(i));
                if tsshift>-9
                    tsshift = -9;
                end

                temp_err_clusts = zeros(clusts,1);
                this_weight = weight(4+ysshift:4+yeshift,...
                                     4+xsshift:4+xeshift,tsshift+40:40);
                [ysize,xsize,tsize]=size(errors);
                
                % Loop through the blocks in the detection area
                for p=1:ysize
                    for q=1:xsize
                        for r=1:tsize
                            
                            % Check if the tap occurs in an area where
                            % there is an error cluster
                            if errors(p,q,r)>0
                                index = errors(p,q,r);
                                temp_err_clusts(index) = ...
                                    temp_err_clusts(index) + ...
                                    this_weight(p,q,r);
                                if (this_weight(p,q,r)>0)
                                    missflag = 0;
                                end
                            end
                        end
                    end
                end
                
                % Find the cluster with largest overlap with tap
                for p=1:clusts
                    if temp_err_clusts(p)==max(temp_err_clusts) && ...
                       max(temp_err_clusts)>0 && detections(p,3)==0
                        detections(p,2)=detections(p,2)+1;
                        detections(p,3)=1;
                    end
                end

                if missflag == 1
                    missed = missed + 1;
                end
            end
        end
                    
        fclose(fid);
    end
    
    % Normalize detections to interval [0,1]
    detections(:,2)=detections(:,2)./num_files;
end

    
