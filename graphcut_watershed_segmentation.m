% Algorithm improved by Katarzyna Hajdowska

% Interactive Graph cuts Segmentation Algorithm
% Towards Medical Image Analysis course
% By Ravindra Gadde, Raghu Yalamanchili
 
% Interactive Graph cut segmentation using max flow algorithm as described
% in Y. Boykov, M. Jolly, "Interactive Graph Cuts for Optimal Boundary and
% Region Segmentation of Objects in N-D Images", ICCV, 2001.

clear all
close all
clc

format short g

% mex maxflowmex.cpp graph.cpp maxflow.cpp % Mex
% UNCOMMENT if you need to compile Mex

% !!!
radius_hough=11; % radius of cells
radius_hough_dist = 12; % distance between centers determined to belonging to one cell
% !!!


warning('off','all')

K=10; % Large constant
sigma=0.00001;% Similarity variance
lambda=10^12;% Terminal Constant
c=10^2;% Similarity Constant

y_top = 250;
y_down = 1140;
x_left = 50;
x_right = 850;
% coords of area to segment

cell_area=pi * radius_hough ^2; % area taken by circle of given radius


folder1=sprintf('results'); % save dir
mkdir(folder1)
folder2=sprintf('results_centers');
mkdir(folder2)

for imagenumber=160:163 % index of image to process

txt = sprintf('image nr %d', imagenumber);
disp(txt)

path = sprintf('images/segm1_%d-1.tif',imagenumber); %file path
im = imread(path);

im_original16 = im; 

J = imadjust(im,stretchlim(im));

im = im2uint8(im);
im2 = im2uint8(J);

if size(im,3)==1
    im = repmat(im,[1 1 3]); 
    im2 = repmat(im2,[1 1 3]); 
    % so the code works in 3D
end

imuint8 = im2uint8(rgb2gray(im));

im_original = im; 

m = double(rgb2gray(im)); 
m2 = double(rgb2gray(im2));
[height,width] = size(m);

histogram = imhist(im2);

histogram_mode = find(histogram==max(histogram)); % the most common value of brightness
mid_value = length(histogram)/2;

im_max_fin = zeros(height,width);


m = double(rgb2gray(im)); 
 
disp('building graph');
 
N = height*width;
 
% construct graph
% Calculate weighted graph
E = edges4connected(height,width); % neighbourhoods
V = c*exp(-abs(m(E(:,1))-m(E(:,2))))./(2*sigma^2); % vertices
A = sparse(E(:,1),E(:,2),V,N,N,4*N); % graph-matrix

clear V;
clear E;

% Hough transform to find centers
[centers1,~,~] = imfindcircles(imadjust(rgb2gray(im)),[10 30],'Sensitivity',0.875,'ObjectPolarity','bright','Method','twostage','EdgeThreshold',0.01);
[centers2,~,~] = imfindcircles(imadjust(rgb2gray(im)),[30 100],'Sensitivity',0.875,'ObjectPolarity','bright','Method','twostage','EdgeThreshold',0.01);
centers = [centers1; centers2];

centers = round(centers); 
fg = zeros(size(centers,1),2); %foreground
fg(:,1) = ceil(centers(:,2));
fg(:,2) = ceil(centers(:,1));

[x1, y1] = find(m == max((min(m(x_left:x_right,y_top:y_down))))); % background - max of min to remove artifacts
x1 = x1(1,1);
y1 = y1(1,1);

bg(:,1)=ceil(y1); %background
bg(:,2)=ceil(x1);

T = calc_weights(m,fg,bg,K,lambda); % weights
  
%Max flow Algorithm
disp('calculating maximum flow');
 
[flow,labels] = maxflow(A,T);
labels = reshape(labels,[height width]);

labels = uint16(labels); 

se = strel('disk',3); 
erodedlabel2 = imclose(labels,se); % to remove very small objects
erodedlabel2 = imopen(erodedlabel2,se);
erodedlabel2 = uint16(bwareaopen(imcomplement(imbinarize(erodedlabel2)),floor(0.00002*height*width))); % removes objects smaller than 0.002% of image area
erodedlabel2(erodedlabel2>0) = 1;


erodedlabel2_temp = zeros(height,width);
erodedlabel2_temp(x_left:x_left+size(erodedlabel2(x_left:x_right,y_top:y_down),1)-1, y_top:y_top + size(erodedlabel2(x_left:x_right,y_top:y_down),2) - 1) = erodedlabel2(x_left:x_right,y_top:y_down);
erodedlabel2 = uint16(erodedlabel2_temp);
% removes objects outside compartment

[labeledImage, ~] = bwlabel(((imbinarize(erodedlabel2))), 8); 
coloredLabelsImage = label2rgb (labeledImage, 'hsv', 'k', 'shuffle'); 


clear A;
clear T;


[centers1,r1_1,~] = imfindcircles(im,[9 18],'Sensitivity',0.9,'ObjectPolarity','bright','Method','twostage','EdgeThreshold',0.01); % Hough transform to detect exact positions of centers
[centers2,r1_2,~] = imfindcircles(im,[18 50],'Sensitivity',0.9,'ObjectPolarity','bright','Method','twostage','EdgeThreshold',0.01); % Hough transform to detect exact positions of centers

im = [];

centers = [centers1; centers2]; 
centers = round(centers); 

radiuses = [r1_1; r1_2];
radiuses = round(radiuses);

m_mask_exact = zeros(size(m)); 
m_mask_exact = im2uint8(m_mask_exact);

for i=1:size(centers,1)
    m_mask_exact((centers(i,2)),(centers(i,1))) = 255; % creates a mask with max brightness where Hough transform detected centers of cells
end

im_original_etap2_double = ((im_original16));

result_matrix_segm2 = rgb2gray(coloredLabelsImage);
result_matrix_segm1 = result_matrix_segm2; 

coloredLabelsImage = [];

cut_cells_segm1 = zeros(height,width,max(max(labeledImage)));
cut_cells_segm2 = zeros(height,width,max(max(labeledImage)));
uncut_cells = zeros(height,width,max(max(labeledImage)));

iter = max(max(labeledImage));

slice_parfor = zeros(height,width,iter); % stores layers with detected cells

segm2_keeper = [0 0];
segm2_keeper_circle = [0 0];

for i=1:iter
    
    segm2 = segm2_keeper;
    segm2_circle = segm2_keeper_circle;
       
    mask_temp = uint8(labeledImage);
    mask_temp(mask_temp~=i) = 0; % creates mask with one object of label number equal to iteration number
    mask_temp_double = im2double(mask_temp);
    connected_cells = m_mask_exact .* mask_temp; % stores centers of one object of label number equal to iteration number
    
    mask_temp_hough = im_original_etap2_double; 
    mask_temp_hough(mask_temp_double==0) = 0; % image with only currently segmented cells

    
    [centers_hough1,radiuses_hough1,~] = imfindcircles(mask_temp_hough,[9 18],'Sensitivity',0.9,'ObjectPolarity','bright','Method','twostage','EdgeThreshold',0.01); % Hough transform to detect exact positions of centers
    [centers_hough2,radiuses_hough2,~] = imfindcircles(mask_temp_hough,[18 50],'Sensitivity',0.9,'ObjectPolarity','bright','Method','twostage','EdgeThreshold',0.01); % Hough transform to detect exact positions of centers

    centers_hough = [centers_hough1; centers_hough2]; 
    centers_hough = round(centers_hough); 

    radiuses_hough = [radiuses_hough1; radiuses_hough2];
    radiuses_hough = round(radiuses_hough);
      
    area_before = bwarea(mask_temp_hough);
    
    circles = zeros(height,width);
       
    for j=1:length(radiuses_hough)   


       % as was written here: https://ch.mathworks.com/matlabcentral/answers/87111-how-can-i-draw-a-filled-circle
        [columnsInImage, rowsInImage] = meshgrid(1:width, 1:height);

        circlePixels = (rowsInImage - round(centers_hough(j,2))).^2 ...
            + (columnsInImage - round(centers_hough(j,1))).^2 < (floor(radiuses_hough(j))+0.75).^2;

        circlePixels = double((circlePixels));

       circles = circles + circlePixels;


    end
    
    circles(circles>0)=1;
    
    mask_temp_hough_double = im2double(mask_temp_hough);
    mask_temp_hough_double(mask_temp_hough_double>0) = 1;
    
    mask_temp_hough_double = mask_temp_hough_double - circles; 
    
    area_after = bwarea(mask_temp_hough_double);
    
    % checks convexity of object
    mask_temp_hough_convex = im2double(mask_temp_hough);
    mask_temp_hough_convex(mask_temp_hough_convex>0)=1;
    
    convex = regionprops(mask_temp_hough_convex,'ConvexArea');
    convex = convex.ConvexArea;
    area_before;
    
    isConvex = (convex>0.94*area_before && convex<1.06*area_before);
        
    if (sum(sum(connected_cells))>255 && sum(sum(mask_temp))>0 && ~(isConvex)) % if sum is more than 255, and there exists an object in mask, and object is not convex, then there is a need to segment further
        
       if ((area_after) >= pi*min(radiuses)^2) 
        [centers_hough1_add,~,~] = imfindcircles(mask_temp_hough_double,[9 18],'Sensitivity',0.9,'ObjectPolarity','bright','Method','twostage','EdgeThreshold',0.01); % Hough transform to detect exact positions of centers
        [centers_hough2_add,~,~] = imfindcircles(mask_temp_hough_double,[18 50],'Sensitivity',0.9,'ObjectPolarity','bright','Method','twostage','EdgeThreshold',0.01); % Hough transform to detect exact positions of centers

        centers_hough_add = [centers_hough1_add; centers_hough2_add];
        centers_hough_add = round(centers_hough_add);
        
        if ~isempty(centers_hough_add)
            centers_hough = cat(1,centers_hough,centers_hough_add);
        end
       end
        
       slice = im_original_etap2_double;
       slice(mask_temp_double==0) = 0; % stores centers of one object of label number equal to iteration number
       
       slice_parfor(:,:,i) = slice;
       
       % as shown here https://www.youtube.com/watch?v=Tf5buFFgnSU
       % watershed
        
       slice_eq_segm1 = imadjust(slice,stretchlim(slice));     
       
       slice_eq_segm1_slice = slice_eq_segm1;
       slice_eq_segm1_slice(slice_eq_segm1_slice~=max(max(slice_eq_segm1_slice))) = 0;
       se = strel('disk',5);
       slice_eq_segm1_slice = imclose(slice_eq_segm1_slice,se);   % so the cell is more uniform
       
       slice_eq_segm1(slice_eq_segm1_slice==max(max(slice_eq_segm1_slice)))=max(max(slice_eq_segm1_slice));
       
       
       bw_segm1 = im2bw(slice_eq_segm1, graythresh(slice_eq_segm1)/10); 

       bw2_segm1 = imfill(bw_segm1,'holes'); % remove holes       
       bw3_segm1 = imopen(bw2_segm1,ones(5,5)); 

       bw4_segm1 = bwareaopen(bw3_segm1,floor(0.00002*height*width)); % removes objects smaller than 0.002% of image area 
                  
       [centers_segm2_1,~,~] = imfindcircles(slice,[11 18],'Sensitivity',0.925,'ObjectPolarity','bright','Method','twostage','EdgeThreshold',0.01);% finds centers only for one slice
       [centers_segm2_2,~,~] = imfindcircles(slice,[18 50],'Sensitivity',0.925,'ObjectPolarity','bright','Method','twostage','EdgeThreshold',0.01);% finds centers only for one slice

        centers_segm2 = centers_hough;
       
    if ~isempty(centers_segm2)   
 
    x_segm2 = centers_segm2(:,1);
    x_segm2( ~any(x_segm2,2), : ) = []; 
    
    y_segm2 = centers_segm2(:,2);
    y_segm2( ~any(y_segm2,2), : ) = [];
      
    segm2 = [x_segm2 y_segm2];
    segm2 = round(segm2);
    
    segm2_temp = segm2;
    
    segm2_keeper = segm2;

    se = strel('disk',2);
    slice_open = imopen(im2uint8(slice),se);
    
    
    slice_open_cut = slice_open(2:(end-1),2:(end-1));
    
    slice_open_cut(slice_open_cut == 0) = NaN;
    
    minimum = max(min(slice_open_cut))+0.01/2^16; % black border!
    
    
    img_canny = slice_open;
    img_canny(img_canny>0)=1;
    img_canny = imbinarize(img_canny);
    
    img_canny = edge(img_canny,'canny');
      
    dist2 = [];

    for k=1:size(segm2_temp,1)
        for j=1:size(segm2,1)
           dist2(j) = sqrt(sum((segm2_temp(k,:) - segm2(j,:)) .^ 2, 2)); % euclidean distance
        end
        
        checker = 0;
        
        if isempty(segm2)
            break
        end
        
        if ismember(segm2_temp(k,:),segm2,'rows')
        else
           segm2_checker = segm2;
           checker = 1;
        end
      
        
        if min(dist2(dist2>0)) <= 2*radius_hough_dist  % if close
            idx = find(dist2==min(dist2(dist2>0))); % only 1st value
            idx = idx(1);
            closest = segm2(idx,:); % looking for x, y with smallest distance 

            % matrix with lines between points
            z = zeros(size(im_original,1),size(im_original,2),'uint8');
            b = [closest(2), segm2_temp(k,2)];
            a = [closest(1), segm2_temp(k,1)];
            nPoints = max(abs(diff(a)), abs(diff(b)))+1;
            rIndex = round(linspace(b(1), b(2), nPoints));  % Row indices
            cIndex = round(linspace(a(1), a(2), nPoints));  % Column indices
            index = sub2ind(size(z), rIndex, cIndex);     % Linear indices
            z(index) = 255;  % Set the line pixels to the max value of 255 for uint8 types
            
            z_brightness = img_canny;
            z_brightness(z==0)=0;
            avg_value = sum(sum(z_brightness))/nPoints; 
            
            if ((sum(sum(z_brightness))==0)) % checks whether lines cut through canny edges - if they don't, objects are to be merged!
            
            segm2_delete = segm2;
            delete_which = ismember(segm2_delete,closest);
            delete_which = (delete_which(:,1) + delete_which(:,2));
            segm2_delete_1 = segm2_delete(:,1);
            segm2_delete_2 = segm2_delete(:,2);
            segm2_delete_1(delete_which==2) = NaN;
            segm2_delete_2(delete_which==2) = NaN;
            segm2_delete = cat(2,segm2_delete_1,segm2_delete_2);
             
            segm2_delete(all(isnan(segm2_delete), 2), :) = [];
            segm2_delete(isnan(segm2_delete)) = segm2(isnan(segm2_delete)); % makes sure [x, NaN] won't be deleted
            segm2 = segm2_delete;
            dist2(dist2 == min(dist2)) = NaN;
            dist2(any(isnan(dist2), 2), :) = [];
                       
            if checker==1
                segm2 = segm2_checker;
            end
                      
            end
            
        end        
                    dist2 = [];       
    end
      

       
       mask_em_segm1 = zeros(size(m));
       mask_em_segm1 = im2uint8(mask_em_segm1);
        
       centers_segm2 = (floor(segm2));
       
       
       for k=1:size(centers_segm2,1) % to not escape image boundaries
          if centers_segm2(k,1) < 5
              centers_segm2(k,1) = 5;
          end
          if centers_segm2(k,1) > (height - 5)
              centers_segm2(k,1) = (height - 5);
          end
          if centers_segm2(k,2) < 5
              centers_segm2(k,2) = 5;
          end
          if centers_segm2(k,2) > (width - 5)
              centers_segm2(k,2) = (width - 5);
          end
       end

       for r=1:size(centers_segm2,1)      
            mask_em_segm1(centers_segm2(r,2)-3:centers_segm2(r,2)+3,centers_segm2(r,1)-2:centers_segm2(r,1)+2) = 255; % creates a mask with max brightness where Hough transform detected centers of cells
            mask_em_segm1(centers_segm2(r,2)-2:centers_segm2(r,2)+2,centers_segm2(r,1)-3:centers_segm2(r,1)+3) = 255; % creates a mask with max brightness where Hough transform detected centers of cells
       end
       
       slice_eq_c_segm1 = imcomplement(slice_eq_segm1); 
       
       slice_mod_segm1 = imimposemin(slice_eq_c_segm1, ~bw4_segm1 | mask_em_segm1); 
       
	   L_segm1 = watershed(slice_mod_segm1);        
       
       L_ones_segm1 = double(L_segm1);
       L_ones_segm1(L_ones_segm1==1) = 0; % 1 is BACKGROUND!
       L_ones_segm1(L_ones_segm1>0) = 1;
       
       if sum(sum(L_ones_segm1))==0
          slice_open_cut_temp = im2double(slice_open);
          slice_open_cut_temp(slice_open_cut_temp>0)=1;
          L_ones_segm1 = slice_open_cut_temp;
       end
       
       cut_cells_segm1(:,:,i) = L_ones_segm1;
       
       if size(segm2,1)<2
        cut_cells_segm1(:,:,i) = slice;
       end     
       
       
       cut_cells_segm1(:,:,i) = L_ones_segm1;
       
       if size(segm2,1)<2
        cut_cells_segm1(:,:,i) = slice;
       end
       
    else
        cut_cells_segm1(:,:,i) = slice;
    end
       
       % removal of small 'cells'
       [labeledImage_segm2_fix, ~] = bwlabel((((cut_cells_segm1(:,:,i)))), 8);
       
       iter_fix = max(max(labeledImage_segm2_fix));
       centers_fix = zeros(iter_fix,2);
       area_fix = zeros(iter_fix,1);
      

       slice_eq_segm1 = [];
       bw_segm1 = [];
       bw2_segm1 = [];
       bw3_segm1 = [];
       bw4_segm1 = [];
       overlay1_segm1 = [];
       mask_em_segm1 = [];
       overlay2_segm1 = [];
       slice_eq_c_segm1 = [];
       slice_mod_segm1 = [];
       L_segm1 = [];
       L_ones_segm1 = [];
       
  
    else
      
       slice = im_original_etap2_double;
       slice(mask_temp_double==0) = 0;
       slice(slice>0) = 1; % binarized layer with only object of label number equal to iteration number
       
       uncut_cells(:,:,i) = slice;
       
    end
      
end

for i=1:iter
    slice_parfor_for = slice_parfor(:,:,i); % matrix with group of merged cells
        
    result_matrix_segm1(slice_parfor_for~=0) = 0;
    result_matrix_segm1 = result_matrix_segm1 + im2uint8(cut_cells_segm1(:,:,i)) + im2uint8(uncut_cells(:,:,i));
end

im_max_fin(im_max_fin>0)=1;
im_max_fin = uint8(im_max_fin);
im_max_fin(im_max_fin>0)=1;

BW2 = bwperim(im_max_fin);

result_matrix_segm1 = result_matrix_segm1 + im_max_fin;

result_matrix_segm1(BW2==1)=0;

result_matrix_segm1(result_matrix_segm1>0)=1;

result_matrix_segm1(result_matrix_segm1~=0) = 1;

se = strel('disk',1);

result_matrix_segm1 = imopen(result_matrix_segm1,se);

[labeledImage_segm1, numberOfBlobs_segm1] = bwlabel(((imbinarize(result_matrix_segm1))), 8);
coloredLabelsImage_segm1 = label2rgb (labeledImage_segm1, 'hsv', 'k', 'shuffle'); 

erodedlabel2_uint8 = erodedlabel2_temp;
erodedlabel2_uint8 = im2uint8(erodedlabel2_uint8);
erodedlabel2_uint8(erodedlabel2_uint8>0)=1;

erodedlabel2_uint8 = erodedlabel2_uint8 + im_max_fin;
erodedlabel2_uint8(erodedlabel2_uint8>0)=1;

[labeledImage_fin, ~] = bwlabel(((imbinarize(erodedlabel2_uint8))), 8); 

labeledImage_fix = labeledImage_fin;
labeledImage_fix(labeledImage_fix>0)=1; 

labeledImage_fix_label = bwlabel(labeledImage_fix);

coloredLabelsImage_fix = label2rgb(labeledImage_fix_label ,'hsv', 'k', 'shuffle');

%%

        coloredLabelsImage_segm1_fix = im2double(coloredLabelsImage_segm1);
        coloredLabelsImage_segm1_fix = coloredLabelsImage_segm1_fix .* labeledImage_fix; % removes additional connections that do not exist in original image and only exist because of morphological operations
        se = strel('disk',2);
        binarize_temp = (rgb2gray(coloredLabelsImage_segm1_fix));
        binarize_temp(binarize_temp>0)=1;

       labeledImage_segm2_fix = bwlabel(binarize_temp);

       iter_fix = max(max(labeledImage_segm2_fix));
             
       labeledImage_segm2_fix_mask = labeledImage_segm2_fix;
       labeledImage_segm2_fix_mask(labeledImage_segm2_fix_mask>0)=1;

       % checks histograms of cells, divides brightest
       labeledImage_segm2_fix_old = labeledImage_segm2_fix;
              
       mid_value = length(histogram)/2;
       
       im_max_fin = uint8(zeros(height,width));
                   
       for ii=1:iter_fix
           mask_prefix_temp = (labeledImage_segm2_fix);
           mask_prefix_temp(mask_prefix_temp~=ii)=0;
           mask_prefix_temp(mask_prefix_temp>0)=1;
           mask_prefix_temp = uint8(mask_prefix_temp);
           
           mask_histogram = mask_prefix_temp .* im2;
           
           histogram = imhist(mask_histogram);
           
           histogram(1,1)=0; % removal of 0
           histogram_mode = find(histogram==max(histogram),1,'first'); % the most common brightness value
           
           if (histogram_mode<0.7*mid_value || histogram_mode>1.3*mid_value) % if gauss is crooked
              labeledImage_segm2_fix_mask(mask_prefix_temp>0)=0;
              im_max_fin = im_max_fin + mask_prefix_temp;
           end

                      
       end       
       
       labeledImage_segm2_fix = bwlabel(labeledImage_segm2_fix_mask);

       iter_fix = max(max(labeledImage_segm2_fix));

       iter_fix = max(max(labeledImage_segm2_fix));
  
       centers_fix = zeros(iter_fix,2);
       area_fix = [];
       
       mask_labeledImage_segm2_fix = zeros(size(labeledImage_segm2_fix,1),size(labeledImage_segm2_fix,2),iter_fix);
       canny_labeledImage_segm2_fix = zeros(size(labeledImage_segm2_fix,1),size(labeledImage_segm2_fix,2),iter_fix);
       final_result = zeros(size(labeledImage_segm2_fix,1),size(labeledImage_segm2_fix,2));      
       
       % checks if area of those brightest cells is big enough to merge the rest     
       
       labeledImage_brightest = bwlabel(im_max_fin);
       iter_brightest = max(max(labeledImage_brightest)); 
       
       labeledImage_segm2_fix_brightest = labeledImage_segm2_fix;
       
       for ii=1:iter_brightest
           
      
          im_max = im2double(im2);
          max_value = (max(max(im_max(x_left:x_right,y_top:y_down))));          
           
          labeledImage_brightest_temp = labeledImage_brightest;
          labeledImage_brightest_temp(labeledImage_brightest_temp~=ii)=0;
          
          labeledImage_brightest_fin = labeledImage_brightest_temp .* im_max;
          labeledImage_brightest_fin_BRIGHT = labeledImage_brightest_fin(:,:,1);
          labeledImage_brightest_fin_BRIGHT(labeledImage_brightest_fin_BRIGHT<0.9*max_value)=0;
          labeledImage_brightest_fin_DARK = labeledImage_brightest_fin(:,:,1);
          labeledImage_brightest_fin_DARK(labeledImage_brightest_fin_BRIGHT>0)=0;
          
          se = strel('disk',5);
          
          labeledImage_brightest_fin_DARK_area = labeledImage_brightest_fin_DARK;
          labeledImage_brightest_fin_DARK_area(labeledImage_brightest_fin_DARK_area>0)=1;
          labeledImage_brightest_fin_DARK_area = imclose(labeledImage_brightest_fin_DARK_area,se);          
          labeledImage_brightest_fin_DARK_area = imopen(labeledImage_brightest_fin_DARK_area,se);
          labeledImage_brightest_fin_BRIGHT_area = labeledImage_brightest_fin_BRIGHT;
          labeledImage_brightest_fin_BRIGHT_area(labeledImage_brightest_fin_BRIGHT_area>0)=1;
          labeledImage_brightest_fin_BRIGHT_area = imclose(labeledImage_brightest_fin_BRIGHT_area,se);
          labeledImage_brightest_fin_BRIGHT_area = imopen(labeledImage_brightest_fin_BRIGHT_area,se);        

          area_brightest = sum(sum(labeledImage_brightest_fin_BRIGHT_area));
          
          area_all = sum(sum(labeledImage_brightest_fin_BRIGHT_area))+sum(sum(labeledImage_brightest_fin_DARK_area));
          
          if (area_brightest/area_all > 0.5 || area_brightest<200) % if it covers most of the cell then merge
              
             labeledImage_segm2_fix(labeledImage_brightest_temp>0)=ii;
             
          else
             labeledImage_segm2_fix(labeledImage_brightest_temp>0)=0; % bwlabel is borked
             labeledImage_segm2_fix(labeledImage_brightest_fin_BRIGHT_area>0)=iter_fix+1; 
             labeledImage_segm2_fix(labeledImage_brightest_fin_DARK_area>0)=ii; 
             BW3 = bwperim(labeledImage_brightest_fin_DARK_area);
             labeledImage_segm2_fix(BW3==1)=0;
          end   
          
          iter_fix = max(max(labeledImage_segm2_fix));
       end
              
       labeledImage_segm2_fix = bwlabel(labeledImage_segm2_fix,4);
       iter_fix = max(max(labeledImage_segm2_fix));
       
       for ii=1:iter_fix
           
           labeledImage_segm2_fix_temp = labeledImage_segm2_fix;
           labeledImage_segm2_fix_temp(labeledImage_segm2_fix_temp~=ii) = 0; 
           labeledImage_segm2_fix_temp(labeledImage_segm2_fix_temp~=0) = 1;
           mask_labeledImage_segm2_fix(:,:,ii) = labeledImage_segm2_fix_temp;
           
           stats = regionprops('table',labeledImage_segm2_fix_temp,'Centroid');
           centers_fix = cat(1,centers_fix,stats.Centroid);
           
           area = sum(sum(labeledImage_segm2_fix_temp));
           area_fix = cat(1,area_fix,area);
                     
       end
       
       centers_fix( ~any(centers_fix,2), : ) = [];
       centers_fix = round(centers_fix);
       
       dist2_fix = [];
       
       for ii=1:iter_fix
           
           dist2_fix = [];
           merge_cells_value=pi * radius_hough ^2; 
           if area_fix(ii) < merge_cells_value % if area is smaller than area of cell of given radius, merge with closest neighbour with morphological operation
              
               
               mask_temp = labeledImage_segm2_fix;
               mask_temp(mask_temp==ii)=0;
               boundaries = bwboundaries(mask_temp);
               boundaries = (cell2mat(boundaries));
               
               boundariesflip = fliplr(boundaries);
               
               if ~isempty(boundariesflip)
               
               dist2 = zeros(1,j);
               
               for j=1:size(boundaries,1)
                   dist2_fix(j) = sqrt(sum((boundariesflip(j,:) - centers_fix(ii,:)) .^ 2, 2)); % euclidean distance
               end
                       
               idx_fix=0;
               
               while (idx_fix==0)
               
               idx_fix1 = find(dist2_fix == min(dist2_fix(dist2_fix>0)),1,'first');
               value = boundaries(idx_fix1,:);
               
               idx_fix = labeledImage_segm2_fix(value(1,1),value(1,2));
               
               if idx_fix==0
                   dist2_fix(1,idx_fix1)=0;               
               end
               
               end
               
               mask_labeledImage_segm2_fix(:,:,idx_fix) = mask_labeledImage_segm2_fix(:,:,idx_fix) + mask_labeledImage_segm2_fix(:,:,ii);
               mask_labeledImage_segm2_fix_temp(:,:,idx_fix) = mask_labeledImage_segm2_fix(:,:,idx_fix);
               mask_labeledImage_segm2_fix(:,:,ii) = zeros(size(labeledImage_segm2_fix,1),size(labeledImage_segm2_fix,2));
               
               mask_labeledImage_segm2_fix(:,:,idx_fix) = mask_labeledImage_segm2_fix_temp(:,:,idx_fix); 
               
               se = strel('disk',2);
               mask_labeledImage_segm2_fix(:,:,idx_fix) = imclose(mask_labeledImage_segm2_fix(:,:,idx_fix),se);
               
               area_fix(idx_fix) = sum(sum(mask_labeledImage_segm2_fix(:,:,idx_fix))); % update of area and center
               stats = regionprops('table',mask_labeledImage_segm2_fix(:,:,idx_fix),'Centroid'); 
               centroids = table2array(stats);
               centroids(any(isnan(centroids), 2), :) = [];
               centroids = centroids(1,:);
               centers_fix(idx_fix,:) = round(centroids);
               
               end
           else
                % do nothing
           end           
       end

       for ii=1:iter_fix
            final_result = final_result + mask_labeledImage_segm2_fix(:,:,ii); % flattens results to 2D
       end

coloredLabelsImage_segm1_fix_gray = rgb2gray(coloredLabelsImage_segm1_fix);
coloredLabelsImage_segm1_fix_gray(coloredLabelsImage_segm1_fix_gray>0)=1;
coloredLabelsImage_segm1_fix_gray(final_result==1)=1;
coloredLabelsImage_segm1_fix_gray = coloredLabelsImage_segm1_fix_gray + final_result;

coloredLabelsImage_segm1_fix_gray = final_result; 

coloredLabelsImage_segm1_fix_gray_label = bwlabel(coloredLabelsImage_segm1_fix_gray,4);

coloredLabelsImage_segm1_fix_gray_temp = zeros(height,width);

for labelfin=1:max(max(coloredLabelsImage_segm1_fix_gray_label))
    coloredLabelsImage_segm1_fix_gray_label_temp = coloredLabelsImage_segm1_fix_gray_label;
    coloredLabelsImage_segm1_fix_gray_label_temp(coloredLabelsImage_segm1_fix_gray_label_temp~=labelfin) = 0;
    coloredLabelsImage_segm1_fix_gray_label_temp(coloredLabelsImage_segm1_fix_gray_label_temp>0) = 1;
    
    if sum(sum(coloredLabelsImage_segm1_fix_gray_label_temp))<0.0001*(width*height) % small area
        coloredLabelsImage_segm1_fix_gray_label_temp = zeros(height,width);
    end
    
    coloredLabelsImage_segm1_fix_gray_temp = coloredLabelsImage_segm1_fix_gray_temp + coloredLabelsImage_segm1_fix_gray_label_temp;
end

coloredLabelsImage_segm1_fix_gray = coloredLabelsImage_segm1_fix_gray_temp;

se = strel('disk',1);
coloredLabelsImage_segm1_fix_gray_label = imopen(coloredLabelsImage_segm1_fix_gray,se);

iter_fin = max(max(bwlabel(coloredLabelsImage_segm1_fix_gray_label,4)));

im_max_fin_double = im2double(im_max_fin);
im_max_fin_double(im_max_fin_double>0)=1;

coloredLabelsImage_segm1_fix_gray_label2 = zeros(height,width);

for ii=1:iter_fin % removal of small 'debris', but not brightest!
    coloredLabelsImage_segm1_fix_gray_label_temp = bwlabel(coloredLabelsImage_segm1_fix_gray_label,4);
    coloredLabelsImage_segm1_fix_gray_label_temp(coloredLabelsImage_segm1_fix_gray_label_temp~=ii)=0;
    coloredLabelsImage_segm1_fix_gray_label_temp(coloredLabelsImage_segm1_fix_gray_label_temp>0)=1;
    
    checker_temp = im_max_fin_double;
    checker_temp(coloredLabelsImage_segm1_fix_gray_label_temp==0)=0;
    
    if ((sum(sum(coloredLabelsImage_segm1_fix_gray_label_temp))<round(0.25*pi*radius_hough^2)) && sum(sum(checker_temp))==0) % small and NOT BRIGHTEST
        coloredLabelsImage_segm1_fix_gray_label_temp = zeros(height,width);
        disp('delet')
    end
    
    coloredLabelsImage_segm1_fix_gray_label2 = coloredLabelsImage_segm1_fix_gray_label2 + coloredLabelsImage_segm1_fix_gray_label_temp;
end

coloredLabelsImage_segm1_fix_gray_label2(coloredLabelsImage_segm1_fix_gray_label2>0)=1;
coloredLabelsImage_segm1_fix_gray_label = bwlabel(coloredLabelsImage_segm1_fix_gray_label2,4);


coloredLabelsImage_segm1_fix_color = label2rgb(coloredLabelsImage_segm1_fix_gray_label, 'hsv', 'k', 'shuffle');

labeledImage_segm1 = [];

coloredLabelsImage_segm1_fix_gray_label_uint8 = im2uint8(coloredLabelsImage_segm1_fix_gray_label);
coloredLabelsImage_segm1_fix_gray_label_uint8(coloredLabelsImage_segm1_fix_gray_label_uint8>0)=255;

image_final_segm1 =  coloredLabelsImage_segm1_fix_gray_label_uint8 .* im_original;


coloredLabelsImage_segm1_fix_gray_label_uint16 = uint16(coloredLabelsImage_segm1_fix_gray_label);
coloredLabelsImage_segm1_fix_gray_label_uint16(coloredLabelsImage_segm1_fix_gray_label_uint16>0)=1;
image_final_segm1_uint16 = coloredLabelsImage_segm1_fix_gray_label_uint16 .* J;



fnamesave = sprintf('/segm1_%d-1.png',imagenumber);
fnamesave = [folder1 fnamesave];
imwrite(J,fnamesave);

fnamesave = sprintf('/segm1_%d-2.png',imagenumber);
fnamesave = [folder1 fnamesave];
imwrite(coloredLabelsImage_fix,fnamesave);

fnamesave = sprintf('/segm1_%d-3.png',imagenumber);
fnamesave = [folder1 fnamesave];
imwrite(coloredLabelsImage_segm1_fix_color,fnamesave);

fnamesave = sprintf('/segm1_%d-4.png',imagenumber);
fnamesave = [folder1 fnamesave];
imwrite(image_final_segm1,fnamesave);

fnamesave = sprintf('/segm1_%d-5.png',imagenumber);
fnamesave = [folder1 fnamesave];
imwrite(image_final_segm1_uint16,fnamesave);

labeled_objects = bwlabel(image_final_segm1_uint16);

% in case only centers are needed
image_centers = zeros(height,width);

for z=1:max(max(labeled_objects))
    img_centers_temp = labeled_objects;
    img_centers_temp(img_centers_temp~=z) = 0;
    
   centers_regionprops = regionprops(imbinarize(img_centers_temp),'centroid'); 
   centers_regionprops2 = centers_regionprops.Centroid;
   centers_regionprops2 = round(centers_regionprops2);
   
   for r=1:size(centers_regionprops2,1)
        image_centers(centers_regionprops2(r,2)-4:centers_regionprops2(r,2)+4,centers_regionprops2(r,1)-3:centers_regionprops2(r,1)+3) = 1; % creates a mask with max brightness where Hough transform detected centers of cells
        image_centers(centers_regionprops2(r,2)-3:centers_regionprops2(r,2)+3,centers_regionprops2(r,1)-4:centers_regionprops2(r,1)+4) = 1; % creates a mask with max brightness where Hough transform detected centers of cells
   end
   
end

fnamesave = sprintf('/segm1_%d-6.png',imagenumber);
fnamesave = [folder2 fnamesave];
imwrite(image_centers,fnamesave);

end 