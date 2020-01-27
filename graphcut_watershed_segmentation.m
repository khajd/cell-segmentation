% Algorithm improved by Katarzyna Hajdowska

% Interactive Graph cuts Segmentation Algorithm
% Towards Medical Image Analysis course
% By Ravindra Gadde, Raghu Yalamanchili
 
% Interactive Graph cut segmentation using max flow algorithm as described
% in Y. Boykov, M. Jolly, "Interactive Graph Cuts for Optimal Boundary and
% Region Segmentation of Objects in N-D Images", ICCV, 2001.

clear all;
close all;
clc;
%wyczyszczenie pamieci 

mkdir 'wyniki'
%utworzenie folderu

% mex maxflowmex.cpp graph.cpp maxflow.cpp % Mex

% delete(gcp('nocreate'))
% parpool
%usuniecie i zainicjowanie workerow

warning('off','all')
%wylaczenie ostrzezen

K=10; % Large constant
sigma=0.00001;% Similarity variance
lambda=10^12;% Terminal Constant
c=10^2;% Similarity Constant

y_top = 250;
y_down = 1140;
x_left = 50;
x_right = 850;
%inicjalizacja stalych

fnameread = '027 FITC_C.tif';
info = imfinfo(fnameread);
%wczytanie nazwy stosu plikow i informacji o niej

time_parfor = zeros(1,163); 
%inicjalizacja macierzy mierzacej czas wykonania

for imagenumber=316:2:326
   
tic
    
im = imread(fnameread, imagenumber, 'Info', info);

im_original16 = im; 

im = im2uint8(im);
%wczytanie i zmiana typu obrazu

if size(im,3)==1
    im = repmat(im,[1 1 3]); 
    %replikacja w celu zaspokojenia dostosowania programu do pracy na obrazach barwnych
end

imuint8 = im2uint8(rgb2gray(im));

im_original = im; 
%obraz oryginalny

m = double(rgb2gray(im)); 
%dwuwymiarowy szary obraz-oryginal
[height,width] = size(m);
%wymiary obrazu

% se = strel('disk',1);
% im_open = imopen(im,se);
% 
% % maximum = max(max(max(im(2:(end-1),2:(end-1)))));
% 
% minimum = max(min(m(2:(end-1),2:(end-1))))/2^16;

%img_canny = edge((rgb2gray(im_open)));
% img_canny = edge(imbinarize(imuint8,(minimum+1)/255));
%img_canny = edge(imbinarize(m,(0.07*2^16)));
% img_canny = edge(imbinarize,
%krawedzie
 
disp('building graph');
 
N = height*width;
%liczba pikseli w obrazie
 
% construct graph
% Calculate weighted graph
E = edges4connected(height,width);
%macierz zawierajaca informacje o sasiedztwach
V = c*exp(-abs(m(E(:,1))-m(E(:,2))))./(2*sigma^2);
%macierz zawierajaca wierzcholki
A = sparse(E(:,1),E(:,2),V,N,N,4*N);
%macierz-graf z obrazu
 
clear V;
clear E;

%transformata Hougha znajduje srodki kol
[centers1,~,~] = imfindcircles(im,[10 29],'Sensitivity',0.85,'ObjectPolarity','bright','Method','twostage','EdgeThreshold',0.01);
[centers2,~,~] = imfindcircles(im,[30 50],'Sensitivity',0.85,'ObjectPolarity','bright','Method','twostage','EdgeThreshold',0.01);

centers = [centers1; centers2];
%informacja o polozeniu srodkow

centers = round(centers); %przyblizenie do liczb calkowitych
fg = zeros(size(centers,1),2); %foreground, zapisanie pozycji znalezionych srodkow
fg(:,1) = ceil(centers(:,2));
fg(:,2) = ceil(centers(:,1));

[x1, y1] = find(m == max((min(m(x_left:x_right,y_top:y_down))))); %znajduje punkt tla
x1 = x1(1,1);
y1 = y1(1,1);

bg(:,1)=ceil(y1); %background, zapisanie pozycji znalezionego maksymalnego minimum
bg(:,2)=ceil(x1);

T = calc_weights(m,fg,bg,K,lambda); 
%obliczenie wag
  
%Max flow Algorithm
disp('calculating maximum flow');
 
[flow,labels] = maxflow(A,T);
%problem maksymalnego przeplywu
labels = reshape(labels,[height width]);
%zmiana kszta³tu

labels = uint16(labels); 
%zmiana typu

se = strel('square',5); %tworzy obiekt do operacji morfologicznej
erodedlabel2 = imopen(labels,se); %operacja morfoliczna otwarcia - by pozbyc sie malutkich obiektow

erodedlabel2 = uint16(bwareaopen(imcomplement(imbinarize(erodedlabel2)),floor(0.00002*height*width))); %usuwa z obrazu elementy mniejsze niz [zaokraglone w dol] 0.00002 * pole obrazu

erodedlabel2(erodedlabel2>0) = 1;
%binaryzacja obrazu

erodedlabel2_temp = zeros(height,width);
erodedlabel2_temp(x_left:x_left+size(erodedlabel2(x_left:x_right,y_top:y_down),1)-1, y_top:y_top + size(erodedlabel2(x_left:x_right,y_top:y_down),2) - 1) = erodedlabel2(x_left:x_right,y_top:y_down);
erodedlabel2 = uint16(erodedlabel2_temp);
%usuwa z obrazu komorki nie przebywajace w badanym kompartmencie

[labeledImage, ~] = bwlabel(((imbinarize(erodedlabel2))), 8); %nadaje etykiety
coloredLabelsImage = label2rgb (labeledImage, 'hsv', 'k', 'shuffle'); %koloruje etykiety

fnamesave = sprintf('wyniki/segm1_%d-2.png',(imagenumber/2));
imwrite(coloredLabelsImage,fnamesave);
fnamesave = sprintf('wyniki/segm2_%d-2.png',(imagenumber/2));
imwrite(coloredLabelsImage,fnamesave);
%zapis do pliku

clear A;
clear T;


[centers1,r1_1,~] = imfindcircles(im,[9 17],'Sensitivity',0.9,'ObjectPolarity','bright','Method','twostage','EdgeThreshold',0.01); %powtorna transformata Hougha by dokladniej wyznaczyc srodki komorek - konieczna do podzialu
[centers2,r1_2,~] = imfindcircles(im,[18 50],'Sensitivity',0.9,'ObjectPolarity','bright','Method','twostage','EdgeThreshold',0.01); %powtorna transformata Hougha by dokladniej wyznaczyc srodki komorek - konieczna do podzialu

im = [];

centers = [centers1; centers2]; %informacja o polozeniu srodkow
centers = round(centers); %zaokraglenie do liczb calkowitych

radiuses = [r1_1; r1_2];
radiuses = round(radiuses);

m_mask_exact = zeros(size(m)); 
m_mask_exact = im2uint8(m_mask_exact);

for i=1:size(centers,1)
    m_mask_exact((centers(i,2)),(centers(i,1))) = 255; %tworzy maske z max jasnoscia w miejscach znalezionych przez transformate Hougha srodkow kol/komorek
end

im_original_etap2_double = ((im_original16));
% im_original_etap2_double = im2double(rgb2gray(im_original));
%kopia obrazu oryginalnego

result_matrix_segm2 = rgb2gray(coloredLabelsImage);
result_matrix_segm1 = result_matrix_segm2;
%kopie masek z segmentacji teoriografowej

coloredLabelsImage = [];

cut_cells_segm1 = zeros(height,width,max(max(labeledImage)));
cut_cells_segm2 = zeros(height,width,max(max(labeledImage)));
uncut_cells = zeros(height,width,max(max(labeledImage)));
%macierze na wyniki

iter = max(max(labeledImage));
%ilosc iteracji

slice_parfor = zeros(height,width,iter);
%macierz do przechowywania kolejnych warstw z polaczonymi komorkami

segm2_keeper = [0 0];
segm2_keeper_circle = [0 0];

for i=1:iter
    
    if i==10
        disp('o')
    end
    
    segm2 = segm2_keeper;
    segm2_circle = segm2_keeper_circle;
       
    mask_temp = uint8(labeledImage);
    mask_temp(mask_temp~=i) = 0; %tworzenie maski zawierajacej wylacznie obiekt o numerze etykiety bedacy numerem iteracji
    mask_temp_double = im2double(mask_temp);
    connected_cells = m_mask_exact .* mask_temp; %macierz zawierajaca jedynie srodki obiektu, ktorego numer etykiety jest zgodny z numerem iteracji
    
%     mask_temp_hough = mask_temp;
%     mask_temp_hough = uint16(mask_temp_hough);
%     mask_temp_hough(mask_temp_hough>0) = 2^16 - 1;
%     
    mask_temp_hough = im_original_etap2_double; %to nie jest maska xd
    mask_temp_hough(mask_temp_double==0) = 0; %zmienna przechowuje oryginalny obraz tylko z akurat dzielonymi komorkami
    
%     mask_temp_hough = mask_temp_hough .*  im_original_etap2_double;

    
    [centers_hough1,radiuses_hough1,~] = imfindcircles(mask_temp_hough,[9 17],'Sensitivity',0.9,'ObjectPolarity','bright','Method','twostage','EdgeThreshold',0.01); %powtorna transformata Hougha by dokladniej wyznaczyc srodki komorek - konieczna do podzialu
    [centers_hough2,radiuses_hough2,~] = imfindcircles(mask_temp_hough,[18 50],'Sensitivity',0.9,'ObjectPolarity','bright','Method','twostage','EdgeThreshold',0.01); %powtorna transformata Hougha by dokladniej wyznaczyc srodki komorek - konieczna do podzialu

    centers_hough = [centers_hough1; centers_hough2]; %informacja o polozeniu srodkow
    centers_hough = round(centers_hough); %zaokraglenie do liczb calkowitych

    radiuses_hough = [radiuses_hough1; radiuses_hough2];
    radiuses_hough = round(radiuses_hough);
      
    area_before = bwarea(mask_temp_hough);
    
    circles = zeros(height,width);
       
    for j=1:length(radiuses_hough)   
        


%            mark_circles = insertShape(slice,'FilledCircle',[floor(centers_segm2(j,:)) floor(0.95*radii_segm2(j))],'Color',[255 0 0],'Opacity',1); %rysuje kolka na podstawie danych uzyskanych z transformaty Hougha, choc o delikatnie mniejszym promieniu
%            mark_circles = rgb2gray(mark_circles);
%            mark_circles(mark_circles~=1) = 0;


       %na podstawie https://ch.mathworks.com/matlabcentral/answers/87111-how-can-i-draw-a-filled-circle
        [columnsInImage, rowsInImage] = meshgrid(1:width, 1:height);

        circlePixels = (rowsInImage - round(centers_hough(j,2))).^2 ...
            + (columnsInImage - round(centers_hough(j,1))).^2 < (floor(radiuses_hough(j))+0.75).^2;

        circlePixels = double((circlePixels));

       %circles = circles + mark_circles;
       circles = circles + circlePixels;

       %3 linijki odnosnie mark_circles i circles = circles +
       %mark_circles mozna odkomentowac, jesli dostepny jest toolbox
       %'Computer Vision Toolbox'


    end
    
    circles(circles>0)=1;
    
    mask_temp_hough_double = im2double(mask_temp_hough);
    mask_temp_hough_double(mask_temp_hough_double>0) = 1;
    
    mask_temp_hough_double = mask_temp_hough_double - circles; %po odjeciu
    
    area_after = bwarea(mask_temp_hough_double);
    
    
    if (sum(sum(connected_cells))>255 && sum(sum(mask_temp))>0) %jak jest 1 (255) I HOUGH ZAJMUJE DOSTATECZNIE DUZA POWIERZCHNIE, to znajduje jedna komorke na danym slice'u, jak wiecej, to trzeba ponownie segmentowac      
       
       if ((area_after) >= pi*min(radiuses)^2) 
        [centers_hough1_add,~,~] = imfindcircles(mask_temp_hough_double,[9 17],'Sensitivity',0.9,'ObjectPolarity','bright','Method','twostage','EdgeThreshold',0.01); %powtorna transformata Hougha by dokladniej wyznaczyc srodki komorek - konieczna do podzialu
        [centers_hough2_add,~,~] = imfindcircles(mask_temp_hough_double,[18 50],'Sensitivity',0.9,'ObjectPolarity','bright','Method','twostage','EdgeThreshold',0.01); %powtorna transformata Hougha by dokladniej wyznaczyc srodki komorek - konieczna do podzialu

        centers_hough_add = [centers_hough1_add; centers_hough2_add]; %informacja o polozeniu srodkow
        centers_hough_add = round(centers_hough_add); %zaokraglenie do liczb calkowitych
        
        if ~isempty(centers_hough_add)
            centers_hough = cat(1,centers_hough,centers_hough_add);
        end
       end
        
       slice = im_original_etap2_double;
       slice(mask_temp_double==0) = 0; %zmienna przechowuje oryginalny obraz tylko z akurat dzielonymi komorkami
       
       slice_parfor(:,:,i) = slice;
       
       %na podstawie https://www.youtube.com/watch?v=Tf5buFFgnSU
       %segmentacja wododzialowa bezpoœrednio na wynikach segmentacji
       %teoriografowej

       slice_eq_segm1 = adapthisteq(slice); %wyrownanie histogramu       

       bw_segm1 = im2bw(slice_eq_segm1, graythresh(slice_eq_segm1)); %obraz do czerni i bieli

       bw2_segm1 = imfill(bw_segm1,'holes'); %usuniecie  dziur z obrazu
       
       bw3_segm1 = imopen(bw2_segm1,ones(5,5)); %operacja morfologiczna otwarcia

       bw4_segm1 = bwareaopen(bw3_segm1,floor(0.00002*height*width)); %usuniecie z obrazu obiektow o powierzchni mniejszej niz 0.002% calkowitej powierzchni obrazu
                  
       [centers_segm2_1,~,~] = imfindcircles(slice,[11 17],'Sensitivity',0.925,'ObjectPolarity','bright','Method','twostage','EdgeThreshold',0.01);%znalezienie srodkow komorek tylko dla danego slice'a
       [centers_segm2_2,~,~] = imfindcircles(slice,[18 50],'Sensitivity',0.925,'ObjectPolarity','bright','Method','twostage','EdgeThreshold',0.01);%znalezienie srodkow komorek tylko dla danego slice'a

%        centers_segm2 = [centers_segm2_1; centers_segm2_2];

        centers_segm2 = centers_hough;
       
       %poczatek - eliminacja srodkow
       
       promien_hough = 14; %odleglosc miedzy srodkami uznanymi za nalezace do jednej komorki, na podstawie piku z histogramu promieni dla obrazow 60:90

       odrzucone = 0;

 
    x_segm2 = centers_segm2(:,1);
    x_segm2( ~any(x_segm2,2), : ) = []; %wyrzucenie 0 z macierzy
    
    y_segm2 = centers_segm2(:,2);
    y_segm2( ~any(y_segm2,2), : ) = []; %wyrzucenie 0 z macierzy
      
    segm2 = [x_segm2 y_segm2];
    segm2 = round(segm2);
    
    segm2_temp = segm2;
    
    segm2_keeper = segm2;
    
    % segm2 - macierz znajdowanych z hougha
    
%     for j=1:size(segm2,1)
%         for k=1:size(segm2,1)
%             dist2(k) = sum((segm2(k,:) - segm2(j,:)) .^ 2, 2);

%POCZATEK KOMENTOWANIA

%     if i==10
%         disp('o')
%     end

    se = strel('disk',2);
    slice_open = imopen(slice,se);
    
    slice_open_cut = slice_open(2:(end-1),2:(end-1));
    
    slice_open_cut(slice_open_cut == 0) = NaN;
    
%     min_temp = max(min(slice_open_cut));
%     
%     slice_open_cut(slice_open_cut <= min_temp) = NaN;
    
    minimum = max(min(slice_open_cut))+0.01*2^16;
    
    img_canny = imbinarize(slice_open,1950/(2^16)); %FIX
    
    img_canny = edge(img_canny,'canny');
    
    dist2 = [];

    for k=1:size(segm2_temp,1)
        for j=1:size(segm2,1)
 %            dist2(j) = sqrt(sum((segm2(j,:) - segm2_temp(k,:)) .^ 2, 2)); %odleglosc
           dist2(j) = sqrt(sum((segm2_temp(k,:) - segm2(j,:)) .^ 2, 2)); %odleglosc
 %           dist2(j) = norm(segm2_temp(k,:) - segm2(j,:));
            %closest = recznie(dist2 == min(dist2),:);
            %recznie(recznie==closest)=NaN;
            %recznie(any(isnan(recznie), 2), :) = [];
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
        
        min(dist2(dist2>0))
        
        %dist2(:, ~any(dist2,1)) = []; %wyrzucenie 0 z macierzy
        
        if min(dist2(dist2>0)) <= 2*promien_hough  %jesli jest blisko
            close all
            %min(dist2)
                        
%             idx = find((dist2 == min(dist2))~=0, 'last'); %tylko pierwsza wartosc brana pod uwage
            idx = find(dist2==min(dist2(dist2>0))); %tylko pierwsza wartosc brana pod uwage
            closest = segm2(idx,:); %szukanie punktow x, y dajacej najmniejsza odleglosc 
%             if closest==[626 675]
%                disp('ooo') 
%             end
            %closest = closest(1,:); 

            %macierz rysujaca miedzy punktami linie
            z = zeros(size(im_original,1),size(im_original,2),'uint8');
            b = [closest(2), segm2_temp(k,2)];
            a = [closest(1), segm2_temp(k,1)];
            nPoints = max(abs(diff(a)), abs(diff(b)))+1;
            rIndex = round(linspace(b(1), b(2), nPoints));  % Row indices
            cIndex = round(linspace(a(1), a(2), nPoints));  % Column indices
            index = sub2ind(size(z), rIndex, cIndex);     % Linear indices
            z(index) = 255;  % Set the line pixels to the max value of 255 for uint8 types
%             xy = [a; b];
%             t = diff(xy);
%             t0 = t(:,1)./t(:,2);
%             y = @(x)(x - a(2))*t0 + a(1);
%             x1 = (1:size(z,2))';
%             y1 = round(y(x1));
%             z(size(z,1)*(x1 - 1) + y1) = 1;
%             
            z_brightness = img_canny;
            z_brightness(z==0)=0;
            avg_value = sum(sum(z_brightness))/nPoints; 
%            imshow(histeq(rgb2gray(im_open))+z,[]);
            sum(sum(z_brightness))
            
                           %      && (avg_value < (1.1 * min(z_brightness(z_brightness > 7))) || avg_value > 0.9 * min(z_brightness(z_brightness > 7)))...
            
            
            %pomyslec nad lepsza metoda sprawdzania
            if ((sum(sum(z_brightness))==0))
%              if ((avg_value < (1.1 * img(closest(2),closest(1))) || avg_value > (0.9 * img(closest(2),closest(1))) || avg_value < (1.1 * img(segm2_temp(k,2),segm2_temp(k,1))) || avg_value > (0.9 * img(segm2_temp(k,2),segm2_temp(k,1))))...
%                      && (((all(z_brightness(z_brightness>0)==img(closest(2),closest(1)))) >= 0.95 || (all(z_brightness(z_brightness>0)==img(segm2_temp(k,2),segm2_temp(k,1)))))>=0.95))
%                      
%             avg_value
%             0.9*img(closest(1))
%             1.1*img(segm2_temp(k,1))
%             0.9*img(segm2_temp(k,1))
%             1.1*img(segm2_temp(k,1))

%            if ((avg_value == img(closest(1))) && (avg_value == img(segm2_temp(k,1))))
                
%            imshow(histeq(rgb2gray(im_open))+z,[]);
%             avg_value
%             img(closest(2),closest(1))
%             img(segm2_temp(k,2),segm2_temp(k,1))
%             min(z_brightness(z_brightness > 7))

%             figure
%             img2 = img;
%             img2 = img2 + z;
% %             imshow(z_brightness,[]);
%             %set(gcf, 'Position', get(0, 'Screensize'));
%             hold on
%             plot(segm2(:,1),segm2(:,2),'LineStyle','none','MarkerSize',10,'Marker','*')
            
            segm2_delete = segm2;
            delete_which = ismember(segm2_delete,closest);
            delete_which = (delete_which(:,1) + delete_which(:,2));
            segm2_delete_1 = segm2_delete(:,1);
            segm2_delete_2 = segm2_delete(:,2);
            segm2_delete_1(delete_which==2) = NaN;
            segm2_delete_2(delete_which==2) = NaN;
            segm2_delete = cat(2,segm2_delete_1,segm2_delete_2);
%             segm2(segm2==closest) = NaN;
%            segm2_delete(segm2_delete==closest) = NaN; %%KOMENT
%             location_1 = find(isnan(segm2_delete(:,1)));
%             location_2 = find(isnan(segm2_delete(:,2)));
%             location_check = find(location_1 ~= location_2); %upewnia sie ze nie wykasuje przypadkiem [x, NaN]
%             if length(location_1) < length(location_2)
%                 location_1 = location_2;
%             else
%                 location_2 = location_1;
%             end
%             segm2_delete(location_1(location_check),:) = segm2(location_1(location_check),:);
%             
            segm2_delete(all(isnan(segm2_delete), 2), :) = [];
            segm2_delete(isnan(segm2_delete)) = segm2(isnan(segm2_delete)); %upewnia sie ze nie wykasuje przypadkiem [x, NaN]
            segm2 = segm2_delete;
            dist2(dist2 == min(dist2)) = NaN;
            dist2(any(isnan(dist2), 2), :) = [];
            odrzucone = odrzucone+1;
            %imshow(z_brightness,[]);
            
            if checker==1
                segm2 = segm2_checker;
            end
            

            
            end
            
        end
        
                    dist2 = [];
        
    end
      
% KONIEC KOMENTOWANIA    
       
       mask_em_segm1 = zeros(size(m));
       mask_em_segm1 = im2uint8(mask_em_segm1);
        
%        centers_segm2 = (floor(centers_segm2));
       centers_segm2 = (floor(segm2));
       
       
       for k=1:size(centers_segm2,1) %zabezpieczenie by nie wyjsc za wymiary obrazu
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
            mask_em_segm1(centers_segm2(r,2)-4:centers_segm2(r,2)+4,centers_segm2(r,1)-3:centers_segm2(r,1)+3) = 255; %tworzy maske z max jasnoscia w miejscach znalezionych przez transformate Hougha srodkow kol/komorek
            mask_em_segm1(centers_segm2(r,2)-3:centers_segm2(r,2)+3,centers_segm2(r,1)-4:centers_segm2(r,1)+4) = 255; %tworzy maske z max jasnoscia w miejscach znalezionych przez transformate Hougha srodkow kol/komorek
       end
       
       slice_eq_c_segm1 = imcomplement(slice_eq_segm1); %odwrocenie kolorow 
       
       slice_mod_segm1 = imimposemin(slice_eq_c_segm1, ~bw4_segm1 | mask_em_segm1); %zwraca obraz po rekonstrukcji morfologicznej takiej, ze ma tylko lokalne minimum gdzie wynik logicznego OR nie jest rowny 0
	   
	   L_segm1 = watershed(slice_mod_segm1); %segmentacja wododzialowa        
       
       L_ones_segm1 = double(L_segm1);
       L_ones_segm1(L_ones_segm1==1) = 0; %1 to jest tlo, dlatego jest usuwane
       L_ones_segm1(L_ones_segm1>0) = 1;
              
       cut_cells_segm1(:,:,i) = L_ones_segm1;
       
       if size(segm2,1)<2
        cut_cells_segm1(:,:,i) = slice;
       end

       %usuwanie malych powierzchniowo 'komorek'
       [labeledImage_segm2_fix, ~] = bwlabel((((cut_cells_segm1(:,:,i)))), 8);
       
       iter_fix = max(max(labeledImage_segm2_fix));
       centers_fix = zeros(iter_fix,2);
       area_fix = zeros(iter_fix,1);
       
       mask_labeledImage_segm2_fix = zeros(size(labeledImage_segm2_fix,1),size(labeledImage_segm2_fix,2),iter_fix);
       canny_labeledImage_segm2_fix = zeros(size(labeledImage_segm2_fix,1),size(labeledImage_segm2_fix,2),iter_fix);
                    
       for ii=1:iter_fix
           
           labeledImage_segm2_fix_temp = labeledImage_segm2_fix;
           labeledImage_segm2_fix_temp(labeledImage_segm2_fix_temp~=ii) = 0; %tworzenie maski zawierajacej wylacznie obiekt o numerze etykiety bedacy numerem iteracji
           labeledImage_segm2_fix_temp(labeledImage_segm2_fix_temp~=0) = 1;
           mask_labeledImage_segm2_fix(:,:,ii) = labeledImage_segm2_fix_temp;
           
           stats = regionprops('table',labeledImage_segm2_fix_temp,'Centroid');
           centers_fix = cat(1,centers_fix,stats.Centroid);
           
           area = sum(sum(labeledImage_segm2_fix_temp));
           area_fix = cat(1,area_fix,area);
           
           %canny_labeledImage_segm2_fix(:,:,ii) = edge(labeledImage_segm2_fix_temp,'canny');
           
       end
       
       centers_fix( ~any(centers_fix,2), : ) = [];
       centers_fix = round(centers_fix);
       
       area_fix( ~any(area_fix,2), : ) = [];
       
       dist2_fix = [];
       
       for ii=1:iter_fix
           
           dist2_fix = [];
           
           if area_fix(ii) < pi * promien_hough ^2 %jesli powierzchnia jest mniejsza niz  powierzchnia komorki o promieniu rownym peak value histogramu
               
               mask_temp = labeledImage_segm2_fix;
               mask_temp(mask_temp==ii)=0;
               boundaries = bwboundaries(mask_temp);
               boundaries = (cell2mat(boundaries));
               
               boundariesflip = fliplr(boundaries);
               
               dist2 = zeros(1,j);
               
%                for j=1:size(centers_fix,1)
               for j=1:size(boundaries,1)
%                    dist2_fix(j) = sqrt(sum((centers_fix(j,:) - centers_fix(ii,:)) .^ 2, 2)); %odleglosc
                     dist2_fix(j) = sqrt(sum((boundariesflip(j,:) - centers_fix(ii,:)) .^ 2, 2)); %odleglosc
                        %closest = recznie(dist2 == min(dist2),:);
                        %recznie(recznie==closest)=NaN;
                        %recznie(any(isnan(recznie), 2), :) = [];
               end
               
               idx_fix=0;
               
               while idx_fix==0
               
               idx_fix1 = find(dist2_fix == min(dist2_fix(dist2_fix>0)),1,'first');
               %closest_fix = centers_fix(idx_fix,:); %szukanie punktow x, y dajacej najmniejsza odleglosc 
               value = boundaries(idx_fix1,:); %FLIPPED! 1sza kolumna to y, 2ga to x
               
               idx_fix = labeledImage_segm2_fix(value(1,1),value(1,2));
               
               if idx_fix==0
                   dist2_fix(1,idx_fix1)=0;               
               end
               
               end
               
               se = strel('disk',2);
               mask_labeledImage_segm2_fix(:,:,idx_fix) = imopen(mask_labeledImage_segm2_fix(:,:,idx_fix),se);
               mask_labeledImage_segm2_fix(:,:,idx_fix) = mask_labeledImage_segm2_fix(:,:,idx_fix) + mask_labeledImage_segm2_fix(:,:,ii);
               mask_labeledImage_segm2_fix_temp(:,:,idx_fix) = mask_labeledImage_segm2_fix(:,:,idx_fix);
               mask_labeledImage_segm2_fix(:,:,ii) = zeros(size(labeledImage_segm2_fix,1),size(labeledImage_segm2_fix,2));
               
               mask_labeledImage_segm2_fix(:,:,idx_fix) = mask_labeledImage_segm2_fix_temp(:,:,idx_fix); %jakby skasowalo to co mazemy
               
               se = strel('square',3);
               mask_labeledImage_segm2_fix(:,:,idx_fix) = imclose(mask_labeledImage_segm2_fix(:,:,idx_fix),se);
               
               area_fix(idx_fix) = sum(sum(mask_labeledImage_segm2_fix(:,:,idx_fix))); %update pola i centrum
              % stats = regionprops('table',mask_labeledImage_segm2_fix(:,:,idx_fix),'Centroid'); 
              % centers_fix(idx_fix,:) = round(table2array(stats));
               %min(dist2_fix(dist2_fix>0))
           end           
       end
       
       for ii=1:iter_fix
            cut_cells_segm1(:,:,i) = cut_cells_segm1(:,:,i) + mask_labeledImage_segm2_fix(:,:,ii); %splaszczanie macierzy do 2d
       end

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
       
     
       %segmentacja wododzialowa PO operacji wykreslenia kolek na
       %polaczonych komorkach
       
    
       [centers_segm2_1,radii_segm2_1,~] = imfindcircles(slice,[9 17],'Sensitivity',0.925,'ObjectPolarity','bright','Method','twostage','EdgeThreshold',0.01);%znalezienie srodkow komorek tylko dla danego slice'a
       [centers_segm2_2,radii_segm2_2,~] = imfindcircles(slice,[18 50],'Sensitivity',0.925,'ObjectPolarity','bright','Method','twostage','EdgeThreshold',0.01);%znalezienie srodkow komorek tylko dla danego slice'a
       
       centers_segm2 = [centers_segm2_1; centers_segm2_2];
       radii_segm2 = [radii_segm2_1; radii_segm2_2];
       %zapisuje lokacje srodkow i promienie znalezionych kol   
       
       
              %poczatek - eliminacja srodkow
       
%        promien_hough = 50; %odleglosc miedzy srodkami uznanymi za nalezace do jednej komorki

       odrzucone = 0;

 
    x_segm2 = centers_segm2(:,1);
    x_segm2( ~any(x_segm2,2), : ) = []; %wyrzucenie 0 z macierzy
    
    y_segm2 = centers_segm2(:,2);
    y_segm2( ~any(y_segm2,2), : ) = []; %wyrzucenie 0 z macierzy
      
    segm2_circle = [x_segm2 y_segm2];
    segm2_circle = round(segm2_circle);
    
    segm2_temp = segm2_circle;
    
    segm2_keeper_circle = segm2_circle;
    
    % segm2 - macierz znajdowanych z hougha
    
%     for j=1:size(segm2,1)
%         for k=1:size(segm2,1)
%             dist2(k) = sum((segm2(k,:) - segm2(j,:)) .^ 2, 2);

%POCZATEK KOMENTOWANIA

%     if i==10
%         disp('o')
%     end

    se = strel('disk',2);
    slice_open = imopen(slice,se);
    
    slice_open_cut = slice_open(2:(end-1),2:(end-1));
    
    slice_open_cut(slice_open_cut == 0) = NaN;
    
%     min_temp = max(min(slice_open_cut));
%     
%     slice_open_cut(slice_open_cut <= min_temp) = NaN;
    
    minimum = max(min(slice_open_cut))+0.01*2^16;
    
    img_canny = imbinarize(slice_open,2050/(2^16)); %FIX
    
    img_canny = edge(img_canny,'canny');

    for k=1:size(segm2_temp,1)
        for j=1:size(segm2_circle,1)
            dist2(j) = sqrt(sum((segm2_circle(j,:) - segm2_temp(k,:)) .^ 2, 2)); %odleglosc
            %closest = recznie(dist2 == min(dist2),:);
            %recznie(recznie==closest)=NaN;
            %recznie(any(isnan(recznie), 2), :) = [];
        end
        
        if isempty(segm2_circle)
            break
        end
        
        dist2(:, ~any(dist2,1)) = []; %wyrzucenie 0 z macierzy
        
        if min(dist2) < promien_hough  %jesli jest blisko
            close all
            min(dist2)
                        
            idx = find((dist2 == min(dist2))~=0, 1, 'first'); %tylko pierwsza wartosc brana pod uwage
            closest = segm2_circle(idx,:); %szukanie punktow x, y dajacej najmniejsza odleglosc 
            %closest = closest(1,:); 

            %macierz rysujaca miedzy punktami linie
            z = zeros(size(im_original,1),size(im_original,2),'uint8');
            b = [closest(2), segm2_temp(k,2)];
            a = [closest(1), segm2_temp(k,1)];
            nPoints = max(abs(diff(a)), abs(diff(b)))+1;
            rIndex = round(linspace(b(1), b(2), nPoints));  % Row indices
            cIndex = round(linspace(a(1), a(2), nPoints));  % Column indices
            index = sub2ind(size(z), rIndex, cIndex);     % Linear indices
            z(index) = 255;  % Set the line pixels to the max value of 255 for uint8 types
%             xy = [a; b];
%             t = diff(xy);
%             t0 = t(:,1)./t(:,2);
%             y = @(x)(x - a(2))*t0 + a(1);
%             x1 = (1:size(z,2))';
%             y1 = round(y(x1));
%             z(size(z,1)*(x1 - 1) + y1) = 1;
%             
            z_brightness = img_canny;
            z_brightness(z==0)=0;
            avg_value = sum(sum(z_brightness))/nPoints; 
%            imshow(histeq(rgb2gray(im_open))+z,[]);
            sum(sum(z_brightness))
            
                           %      && (avg_value < (1.1 * min(z_brightness(z_brightness > 7))) || avg_value > 0.9 * min(z_brightness(z_brightness > 7)))...
            
            
            %pomyslec nad lepsza metoda sprawdzania
            if (sum(sum(z_brightness))==0)
%              if ((avg_value < (1.1 * img(closest(2),closest(1))) || avg_value > (0.9 * img(closest(2),closest(1))) || avg_value < (1.1 * img(segm2_temp(k,2),segm2_temp(k,1))) || avg_value > (0.9 * img(segm2_temp(k,2),segm2_temp(k,1))))...
%                      && (((all(z_brightness(z_brightness>0)==img(closest(2),closest(1)))) >= 0.95 || (all(z_brightness(z_brightness>0)==img(segm2_temp(k,2),segm2_temp(k,1)))))>=0.95))
%                      
%             avg_value
%             0.9*img(closest(1))
%             1.1*img(segm2_temp(k,1))
%             0.9*img(segm2_temp(k,1))
%             1.1*img(segm2_temp(k,1))

%            if ((avg_value == img(closest(1))) && (avg_value == img(segm2_temp(k,1))))
                
%            imshow(histeq(rgb2gray(im_open))+z,[]);
%             avg_value
%             img(closest(2),closest(1))
%             img(segm2_temp(k,2),segm2_temp(k,1))
%             min(z_brightness(z_brightness > 7))

%             figure
%             img2 = img;
%             img2 = img2 + z;
% %             imshow(z_brightness,[]);
%             %set(gcf, 'Position', get(0, 'Screensize'));
%             hold on
%             plot(segm2(:,1),segm2(:,2),'LineStyle','none','MarkerSize',10,'Marker','*')
 
            segm2_delete = segm2_circle;
            delete_which = ismember(segm2_delete,closest);
            delete_which = (delete_which(:,1) + delete_which(:,2));
            segm2_delete_1 = segm2_delete(:,1);
            segm2_delete_2 = segm2_delete(:,2);
            segm2_delete_1(delete_which==2) = NaN;
            segm2_delete_2(delete_which==2) = NaN;
            segm2_delete = cat(2,segm2_delete_1,segm2_delete_2);

%             segm2_delete = segm2_circle;
% %             segm2(segm2==closest) = NaN;
%             segm2_delete(segm2_delete==closest) = NaN;
%             location_1 = find(isnan(segm2_delete(:,1)));
%             location_2 = find(isnan(segm2_delete(:,2)));
%             location_check = find(location_1 ~= location_2); %upewnia sie ze nie wykasuje przypadkiem [x, NaN]
%             if length(location_1) < length(location_2)
%                 location_1 = location_2;
%             else
%                 location_2 = location_1;
%             end
%             segm2_delete(location_1(location_check),:) = segm2(location_1(location_check),:);
%             
            segm2_delete(all(isnan(segm2_delete), 2), :) = [];
            segm2_delete(isnan(segm2_delete)) = segm2_circle(isnan(segm2_delete)); %upewnia sie ze nie wykasuje przypadkiem [x, NaN]
            segm2_circle = segm2_delete;
            dist2(dist2 == min(dist2)) = NaN;
            dist2(any(isnan(dist2), 2), :) = [];
            odrzucone = odrzucone+1;
            %imshow(z_brightness,[]);
            dist2 = [];
            end
        end
    end

    % KONIEC KOMENTOWANIA    
    
%     radii_segm2_check1 = ismember(round(centers_segm2(:,1)),segm2_circle(:,1));
%     radii_segm2_check2 = ismember(round(centers_segm2(:,2)),segm2_circle(:,2));

    centers_segm2_round = round(centers_segm2);

    radii_segm2_check1 = ismember((centers_segm2_round(:,1)),segm2_circle(:,1));
    radii_segm2_check2 = ismember((centers_segm2_round(:,2)),segm2_circle(:,2));

    radii_segm2_check = radii_segm2_check1 + radii_segm2_check2;
    
    if sum(radii_segm2_check)<2
        segm2_checker = 0;
    else
        segm2_checker = 1;
    end
    
    %wyrzucanie zbednych centersow ciag dalszy
       slice_segm2 = im_original_etap2_double;

       slice_segm2_ones_before = slice_segm2;
       slice_segm2_ones_before(slice_segm2_ones_before>0) = 1;

    if segm2_checker==1
    
    %radii_segm2_ok = radii_segm2;
    radii_segm2_check(radii_segm2_check~=2)=0;
    radii_segm2=radii_segm2.*radii_segm2_check;
    radii_segm2( ~any(radii_segm2,2), : ) = [];  %rows
    
    
       m_mask_exact_check = zeros(size(m));
       m_mask_exact_check = im2uint8(m_mask_exact_check); %analogicznie do m_mask_exact
        
%        centers_segm2 = (floor(centers_segm2));     
       centers_segm2 = (round(segm2_circle));

       for r=1:size(centers_segm2,1)
            m_mask_exact_check(centers_segm2(r,2),centers_segm2(r,1)) = 255; %tworzy maske z max jasnoscia w miejscach znalezionych przez transformate Hougha srodkow kol/komorek
       end
       
       circles = zeros(height,width);
       
%        for j=1:length(radii_segm2)    %ZNALEZC BLAD - pewnie znajduje
%        nieunikalne wartosci ismember
%        
%            
% %            mark_circles = insertShape(slice,'FilledCircle',[floor(centers_segm2(j,:)) floor(0.95*radii_segm2(j))],'Color',[255 0 0],'Opacity',1); %rysuje kolka na podstawie danych uzyskanych z transformaty Hougha, choc o delikatnie mniejszym promieniu
% %            mark_circles = rgb2gray(mark_circles);
% %            mark_circles(mark_circles~=1) = 0;
% 
% 
%            %na podstawie https://ch.mathworks.com/matlabcentral/answers/87111-how-can-i-draw-a-filled-circle
%             [columnsInImage, rowsInImage] = meshgrid(1:width, 1:height);
% 
%             circlePixels = (rowsInImage - round(centers_segm2(j,2))).^2 ...
%                 + (columnsInImage - round(centers_segm2(j,1))).^2 < (floor(0.95*radii_segm2(j))+0.75).^2;
%             
%             circlePixels = double((circlePixels));
%            
%            %circles = circles + mark_circles;
%            circles = circles + circlePixels;
%            
%            %3 linijki odnosnie mark_circles i circles = circles +
%            %mark_circles mozna odkomentowac, jesli dostepny jest toolbox
%            %'Computer Vision Toolbox'
%            
%            
%        end
       
             
       slice_ones = im2double(slice);
       slice_ones(slice~=0)=1;
       
       pokrojona_komorka = slice_ones + circles;
       pokrojona_komorka(pokrojona_komorka<2) = 0; %maska jest jedynka tylko w miejscu, gdzie zarowno slice jak i narysowane kolko ma jedynke
       
       circles = [];
       
       [labeledImage_check, ~] = bwlabel(((imbinarize(pokrojona_komorka))), 8); %nadaje etykiety
       
       pokrojona_komorka = [];
       
       iter_segm2 = max(max(labeledImage_check));     
                  
       for z=1:iter_segm2
       
           mask_temp_check = uint8(labeledImage_check);
           mask_temp_check(mask_temp_check~=z) = 0;
           mask_temp_double_check = im2double(mask_temp_check);
           connected_cells_check = m_mask_exact_check .* mask_temp_check;

           slice_segm2 = im_original_etap2_double;
           slice_segm2(mask_temp_double_check==0) = 0;

           slice_segm2_ones_before = slice_segm2;
           slice_segm2_ones_before(slice_segm2_ones_before>0) = 1;

           if (sum(sum(connected_cells_check))>255 && sum(sum(mask_temp_check))>0) %jak jest 1 (255), to znajduje jedna komorke na danym slice'u, jak wiecej, to trzeba ponownie segmentowac

               %na podstawie https://www.youtube.com/watch?v=Tf5buFFgnSU
               %segmentacja wododzialowa po narysowaniu kol na wynikach
               %segmentacji teoriografowej

               slice_eq_segm2 = adapthisteq(slice_segm2); %wyrownanie histogramu       

               bw_segm2 = im2bw(slice_eq_segm2, graythresh(slice_eq_segm2)); %obraz do czerni i bieli

               bw2_segm2 = imfill(bw_segm2,'holes'); %usuniecie  dziur z obrazu

               bw3_segm2 = imopen(bw2_segm2,ones(5,5)); %operacja morfologiczna otwarcia
               
               bw4_segm2 = bwareaopen(bw3_segm2,floor(0.00002*height*width)); %usuniecie  z obrazu obiektow o powierzchni mniejszej niz 0.002% calkowitej powierzchni obrazu             

               [centers_segm2_1,~,~] = imfindcircles(slice,[6 17],'Sensitivity',0.85,'ObjectPolarity','bright','Method','twostage','EdgeThreshold',0.01); %znalezienie srodkow komorek tylko dla danego slice'a
               [centers_segm2_2,~,~] = imfindcircles(slice,[18 50],'Sensitivity',0.85,'ObjectPolarity','bright','Method','twostage','EdgeThreshold',0.01); %znalezienie srodkow komorek tylko dla danego slice'a

               centers_segm2 = [centers_segm2_1; centers_segm2_2];

               mask_em_segm2 = zeros(size(m));
               mask_em_segm2 = im2uint8(mask_em_segm2);

               centers_segm2 = (floor(centers_segm2));
               
               for k=1:size(centers_segm2,1) %zabezpieczenie by nie wyjsc za wymiary obrazu
                  if centers_segm2(k,1) < 5
                      centers_segm2(k,1) = 5;
                  end
                  if centers_segm2(k,1) > (width - 5) 
                      centers_segm2(k,1) = (width - 5);
                  end
                  if centers_segm2(k,2) < 5
                      centers_segm2(k,2) = 5;
                  end
                  if centers_segm2(k,2) > (height - 5)
                      centers_segm2(k,2) = (height - 5);
                  end
                end

               for r=1:size(centers_segm2,1)
                    mask_em_segm2(centers_segm2(r,2)-4:centers_segm2(r,2)+4,centers_segm2(r,1)-3:centers_segm2(r,1)+3) = 255; %tworzy maske z max jasnoscia w miejscach znalezionych przez transformate Hougha srodkow kol/komorek
                    mask_em_segm2(centers_segm2(r,2)-3:centers_segm2(r,2)+3,centers_segm2(r,1)-4:centers_segm2(r,1)+4) = 255; %tworzy maske z max jasnoscia w miejscach znalezionych przez transformate Hougha srodkow kol/komorek
               end

               slice_eq_c_segm2 = imcomplement(slice_eq_segm2); %odwrocenie kolorow

               slice_mod_segm2 = imimposemin(slice_eq_c_segm2, ~bw4_segm2 | mask_em_segm2); %zwraca obraz po rekonstrukcji morfologicznej takiej, ze ma tylko lokalne minimum gdzie wynik logicznego OR nie jest rowny 0

               L_segm2 = watershed(slice_mod_segm2); %segmentacja wododzialowa

               L_ones_segm2 = double(L_segm2);
               L_ones_segm2(L_ones_segm2==1) = 0; %1 to jest tlo, dlatego jest usuwane
               L_ones_segm2(L_ones_segm2>0) = 1;      

               slice_segm2_ones = slice_segm2;
               slice_segm2_ones(slice_segm2_ones>0) = 1;
               
               if size(segm2_circle,1)<2
                cut_cells_segm2(:,:,i) = slice;
               end

               slice_eq_segm2 = [];
               bw_segm2 = [];
               bw2_segm2 = [];
               bw3_segm2 = [];
               bw4_segm2 = [];
               overlay1_segm2 = [];
               mask_em_segm2 = [];
               overlay2_segm2 = [];
               slice_eq_c_segm2 = [];
               slice_mod_segm2 = [];
               L_segm2 = [];
               

           else
                    cut_cells_segm2(:,:,i) = cut_cells_segm2(:,:,i) + im2double(slice_segm2_ones_before);

                    L_ones_segm2 = double(zeros(height,width));
           end

                  cut_cells_segm2(:,:,i) = cut_cells_segm2(:,:,i) + L_ones_segm2;
              
       end
    else
%                     cut_cells_segm2(:,:,i) = cut_cells_segm2(:,:,i) + im2double(slice_segm2_ones_before);
% 
%                     L_ones_segm2 = double(zeros(height,width));

                cut_cells_segm2(:,:,i) = slice;
    end
    
    else
      
       slice = im_original_etap2_double;
       slice(mask_temp_double==0) = 0;
       slice(slice>0) = 1; %tworzenie zbinaryzowanej warstwy zawierajacej wylacznie obiekt o numerze etykiety zgodnym z numerem iteracji
       
       uncut_cells(:,:,i) = slice; %zapis do zmiennej przechowujacej wyniki
       
    end
      
end

for i=1:iter
    slice_parfor_for = slice_parfor(:,:,i); %przypisuje do zmiennej macierz zawierajaca wylacznie grupe polaczonych komorek
    
    result_matrix_segm2(slice_parfor_for~=0) = 0; %usuwa z koncowej macierzy wynikowej powyzsza grupe
    result_matrix_segm2 = result_matrix_segm2 + im2uint8(cut_cells_segm2(:,:,i)) + im2uint8(uncut_cells(:,:,i)); %na jej miejsce dodawane sa wyniki 2 czesci programu
    
    result_matrix_segm1(slice_parfor_for~=0) = 0;
    result_matrix_segm1 = result_matrix_segm1 + im2uint8(cut_cells_segm1(:,:,i)) + im2uint8(uncut_cells(:,:,i));
end

%dla watershed po segm2
result_matrix_segm2(result_matrix_segm2~=0) = 1;

result_matrix_segm2 = imopen(result_matrix_segm2,se); %operacja morfoliczna otwarcia - by pozbyc sie malutkich obiektow

[labeledImage_segm2, numberOfBlobs_segm_2] = bwlabel(((imbinarize(result_matrix_segm2))), 4); %nadaje etykiety - 4-sasiedztwo
coloredLabelsImage_segm2 = label2rgb (labeledImage_segm2, 'hsv', 'k', 'shuffle'); %koloruje etykiety

labeledImage_segm2 = [];

image_final_segm2 = result_matrix_segm2 .* im_original;

%dla watershed po segm1
result_matrix_segm1(result_matrix_segm1~=0) = 1;

result_matrix_segm1 = imopen(result_matrix_segm1,se);

[labeledImage_segm1, numberOfBlobs_segm1] = bwlabel(((imbinarize(result_matrix_segm1))), 8); %nadaje etykiety
coloredLabelsImage_segm1 = label2rgb (labeledImage_segm1, 'hsv', 'k', 'shuffle'); %koloruje etykiety

labeledImage_fix = labeledImage;
labeledImage_fix(labeledImage_fix>0)=1;

%%

        coloredLabelsImage_segm1_fix = im2double(coloredLabelsImage_segm1);
        coloredLabelsImage_segm1_fix = coloredLabelsImage_segm1_fix .* labeledImage_fix;
        se = strel('disk',2);
         %coloredLabelsImage_segm1_fix = imopen(coloredLabelsImage_segm1_fix,se);
        binarize_temp = (rgb2gray(coloredLabelsImage_segm1_fix));
        binarize_temp(binarize_temp>0)=1;

       labeledImage_segm2_fix = bwlabel(binarize_temp);

       iter_fix = max(max(labeledImage_segm2_fix));
       centers_fix = zeros(iter_fix,2);
       area_fix = zeros(iter_fix,1);
       
       mask_labeledImage_segm2_fix = zeros(size(labeledImage_segm2_fix,1),size(labeledImage_segm2_fix,2),iter_fix);
       canny_labeledImage_segm2_fix = zeros(size(labeledImage_segm2_fix,1),size(labeledImage_segm2_fix,2),iter_fix);
       final_result = zeros(size(labeledImage_segm2_fix,1),size(labeledImage_segm2_fix,2));
       
       for ii=1:iter_fix
           
           labeledImage_segm2_fix_temp = labeledImage_segm2_fix;
           labeledImage_segm2_fix_temp(labeledImage_segm2_fix_temp~=ii) = 0; %tworzenie maski zawierajacej wylacznie obiekt o numerze etykiety bedacy numerem iteracji
           labeledImage_segm2_fix_temp(labeledImage_segm2_fix_temp~=0) = 1;
           mask_labeledImage_segm2_fix(:,:,ii) = labeledImage_segm2_fix_temp;
           
           stats = regionprops('table',labeledImage_segm2_fix_temp,'Centroid');
           centers_fix = cat(1,centers_fix,stats.Centroid);
           
           area = sum(sum(labeledImage_segm2_fix_temp));
           area_fix = cat(1,area_fix,area);
           
          % canny_labeledImage_segm2_fix(:,:,ii) = edge(labeledImage_segm2_fix_temp,'canny');
           
       end
       
       centers_fix( ~any(centers_fix,2), : ) = [];
       centers_fix = round(centers_fix);
       
       area_fix( ~any(area_fix,2), : ) = [];
       
       dist2_fix = [];
       
%        promien_hough = 11;
       
       for ii=1:iter_fix
           
           dist2_fix = [];
           
           if ii == 11
              disp('o') 
           end
           if area_fix(ii) < pi * promien_hough ^2 * 0.75 %jesli powierzchnia jest mniejsza niz  powierzchnia komorki o promieniu rownym peak value histogramu
               
               mask_temp = labeledImage_segm2_fix;
               mask_temp(mask_temp==ii)=0;
               boundaries = bwboundaries(mask_temp);
               boundaries = (cell2mat(boundaries));
               
               boundariesflip = fliplr(boundaries);
               
               dist2 = zeros(1,j);
               
               for j=1:size(boundaries,1)
                  % dist2_fix(j) = sqrt(sum((centers_fix(j,:) - centers_fix(ii,:)) .^ 2, 2)); %odleglosc
                   dist2_fix(j) = sqrt(sum((boundariesflip(j,:) - centers_fix(ii,:)) .^ 2, 2)); %odleglosc
                        %closest = recznie(dist2 == min(dist2),:);
                        %recznie(recznie==closest)=NaN;
                        %recznie(any(isnan(recznie), 2), :) = [];
               end
                             
               idx_fix1 = find(dist2_fix == min(dist2_fix(dist2_fix>0)),1,'first');
               %closest_fix = centers_fix(idx_fix,:); %szukanie punktow x, y dajacej najmniejsza odleglosc 
               value = boundaries(idx_fix1,:);
               
               idx_fix = labeledImage_segm2_fix(value(1,1),value(1,2));
               %closest_fix = centers_fix(idx_fix,:); %szukanie punktow x, y dajacej najmniejsza odleglosc 
%                se = strel('disk',2);
%                mask_labeledImage_segm2_fix(:,:,idx_fix) = imopen(mask_labeledImage_segm2_fix(:,:,idx_fix),se);
               mask_labeledImage_segm2_fix(:,:,idx_fix) = mask_labeledImage_segm2_fix(:,:,idx_fix) + mask_labeledImage_segm2_fix(:,:,ii);
               mask_labeledImage_segm2_fix_temp(:,:,idx_fix) = mask_labeledImage_segm2_fix(:,:,idx_fix);
               mask_labeledImage_segm2_fix(:,:,ii) = zeros(size(labeledImage_segm2_fix,1),size(labeledImage_segm2_fix,2));
               
               mask_labeledImage_segm2_fix(:,:,idx_fix) = mask_labeledImage_segm2_fix_temp(:,:,idx_fix); %jakby skasowalo to co mazemy
               
               se = strel('disk',2);
               mask_labeledImage_segm2_fix(:,:,idx_fix) = imclose(mask_labeledImage_segm2_fix(:,:,idx_fix),se);
               
               area_fix(idx_fix) = sum(sum(mask_labeledImage_segm2_fix(:,:,idx_fix))); %update pola i centrum
               stats = regionprops('table',mask_labeledImage_segm2_fix(:,:,idx_fix),'Centroid'); 
               centers_fix(idx_fix,:) = round(table2array(stats));
               %min(dist2_fix(dist2_fix>0))
           end           
       end

       for ii=1:iter_fix
            final_result = final_result + mask_labeledImage_segm2_fix(:,:,ii); %splaszczanie macierzy do 2d
       end

coloredLabelsImage_segm1_fix_gray = rgb2gray(coloredLabelsImage_segm1_fix);
coloredLabelsImage_segm1_fix_gray(coloredLabelsImage_segm1_fix_gray>0)=1;
coloredLabelsImage_segm1_fix_gray(final_result==1)=1;
coloredLabelsImage_segm1_fix_gray = coloredLabelsImage_segm1_fix_gray + final_result;

%


% 
% %         coloredLabelsImage_segm1_fix = im2double(coloredLabelsImage_segm1);
% %         coloredLabelsImage_segm1_fix = coloredLabelsImage_segm1_fix .* labeledImage_fix;
% %         se = strel('disk',2);
%          %coloredLabelsImage_segm1_fix = imopen(coloredLabelsImage_segm1_fix,se);
% 
%        labeledImage_segm2_fix = bwlabel(coloredLabelsImage_segm1_fix_gray);
% 
%        iter_fix = max(max(labeledImage_segm2_fix));
%        centers_fix = zeros(iter_fix,2);
%        area_fix = zeros(iter_fix,1);
%        
%        mask_labeledImage_segm2_fix = zeros(size(labeledImage_segm2_fix,1),size(labeledImage_segm2_fix,2),iter_fix);
%        canny_labeledImage_segm2_fix = zeros(size(labeledImage_segm2_fix,1),size(labeledImage_segm2_fix,2),iter_fix);
%        final_result = zeros(size(labeledImage_segm2_fix,1),size(labeledImage_segm2_fix,2));
%        
%        for ii=1:iter_fix
%            
%            labeledImage_segm2_fix_temp = labeledImage_segm2_fix;
%            labeledImage_segm2_fix_temp(labeledImage_segm2_fix_temp~=ii) = 0; %tworzenie maski zawierajacej wylacznie obiekt o numerze etykiety bedacy numerem iteracji
%            labeledImage_segm2_fix_temp(labeledImage_segm2_fix_temp~=0) = 1;
%            mask_labeledImage_segm2_fix(:,:,ii) = labeledImage_segm2_fix_temp;
%            
%            stats = regionprops('table',labeledImage_segm2_fix_temp,'Centroid');
%            centers_fix = cat(1,centers_fix,stats.Centroid);
%            
%            area = sum(sum(labeledImage_segm2_fix_temp));
%            area_fix = cat(1,area_fix,area);
%            
%           % canny_labeledImage_segm2_fix(:,:,ii) = edge(labeledImage_segm2_fix_temp,'canny');
%            
%        end
%        
%        centers_fix( ~any(centers_fix,2), : ) = [];
%        centers_fix = round(centers_fix);
%        
%        area_fix( ~any(area_fix,2), : ) = [];
%        
%        dist2_fix = [];
%        
% %        promien_hough = 11;
%        
%        for ii=1:iter_fix
%            
%            dist2_fix = [];
%            
%            if ii == 11
%               disp('o') 
%            end
%            if area_fix(ii) < pi * promien_hough ^2 * 0.75 %jesli powierzchnia jest mniejsza niz  powierzchnia komorki o promieniu rownym peak value histogramu
%                
%                mask_temp = labeledImage_segm2_fix;
%                mask_temp(mask_temp==ii)=0;
%                boundaries = bwboundaries(mask_temp);
%                boundaries = (cell2mat(boundaries));
%                
%                boundariesflip = fliplr(boundaries);
%                
%                dist2 = zeros(1,j);
%                
%                for j=1:size(boundaries,1)
%                   % dist2_fix(j) = sqrt(sum((centers_fix(j,:) - centers_fix(ii,:)) .^ 2, 2)); %odleglosc
%                    dist2_fix(j) = sqrt(sum((boundariesflip(j,:) - centers_fix(ii,:)) .^ 2, 2)); %odleglosc
%                         %closest = recznie(dist2 == min(dist2),:);
%                         %recznie(recznie==closest)=NaN;
%                         %recznie(any(isnan(recznie), 2), :) = [];
%                end
%                              
%                idx_fix1 = find(dist2_fix == min(dist2_fix(dist2_fix>0)),1,'first');
%                %closest_fix = centers_fix(idx_fix,:); %szukanie punktow x, y dajacej najmniejsza odleglosc 
%                value = boundaries(idx_fix1,:);
%                
%                idx_fix = labeledImage_segm2_fix(value(1,1),value(1,2));
%                %closest_fix = centers_fix(idx_fix,:); %szukanie punktow x, y dajacej najmniejsza odleglosc 
% %                se = strel('disk',2);
% %                mask_labeledImage_segm2_fix(:,:,idx_fix) = imopen(mask_labeledImage_segm2_fix(:,:,idx_fix),se);
%                mask_labeledImage_segm2_fix(:,:,idx_fix) = mask_labeledImage_segm2_fix(:,:,idx_fix) + mask_labeledImage_segm2_fix(:,:,ii);
%                mask_labeledImage_segm2_fix_temp(:,:,idx_fix) = mask_labeledImage_segm2_fix(:,:,idx_fix);
%                mask_labeledImage_segm2_fix(:,:,ii) = zeros(size(labeledImage_segm2_fix,1),size(labeledImage_segm2_fix,2));
%                
%                mask_labeledImage_segm2_fix(:,:,idx_fix) = mask_labeledImage_segm2_fix_temp(:,:,idx_fix); %jakby skasowalo to co mazemy
%                
%                se = strel('disk',2);
%                mask_labeledImage_segm2_fix(:,:,idx_fix) = imclose(mask_labeledImage_segm2_fix(:,:,idx_fix),se);
%                
%                area_fix(idx_fix) = sum(sum(mask_labeledImage_segm2_fix(:,:,idx_fix))); %update pola i centrum
%                stats = regionprops('table',mask_labeledImage_segm2_fix(:,:,idx_fix),'Centroid'); 
%                centers_fix(idx_fix,:) = round(table2array(stats));
%                %min(dist2_fix(dist2_fix>0))
%            end           
%        end
% 
%        for ii=1:iter_fix
%             final_result = final_result + mask_labeledImage_segm2_fix(:,:,ii); %splaszczanie macierzy do 2d
%        end
% 
% coloredLabelsImage_segm1_fix_gray = rgb2gray(coloredLabelsImage_segm1_fix);
% coloredLabelsImage_segm1_fix_gray(coloredLabelsImage_segm1_fix_gray>0)=1;
% coloredLabelsImage_segm1_fix_gray = coloredLabelsImage_segm1_fix_gray + final_result;
% 
% 
% 

%

imshow(coloredLabelsImage_segm1_fix_gray);
figure
imshow(final_result)

se = strel('disk',2);
coloredLabelsImage_segm1_fix_gray = imopen(coloredLabelsImage_segm1_fix_gray,se);

coloredLabelsImage_segm1_fix_gray = bwareaopen(coloredLabelsImage_segm1_fix_gray,round(0.1*pi*promien_hough^2)); %usuwanie zbednych fragmentow i smieci

coloredLabelsImage_segm1_fix_gray_label = bwlabel(coloredLabelsImage_segm1_fix_gray);



% max_label = max(max(coloredLabelsImage_segm1_fix_gray_label));
% 
% 
% 
% 
% for i=1:max_label
%    area_final 
%     
% end

coloredLabelsImage_segm1_fix_color = label2rgb(coloredLabelsImage_segm1_fix_gray_label, 'hsv', 'k', 'shuffle');

labeledImage_segm1 = [];

image_final_segm1 = result_matrix_segm1 .* im_original;

%zapis do plikow
fnamesave = sprintf('wyniki/segm1_%d-1.png',(imagenumber/2));
imwrite(im_original,fnamesave);
fnamesave = sprintf('wyniki/segm2_%d-1.png',(imagenumber/2));
imwrite(im_original,fnamesave);
fnamesave = sprintf('wyniki/segm1_%d-3.png',(imagenumber/2));
imwrite(coloredLabelsImage_segm1_fix_color,fnamesave);
fnamesave = sprintf('wyniki/segm2_%d-3.png',(imagenumber/2));
imwrite(coloredLabelsImage_segm2,fnamesave);
fnamesave = sprintf('wyniki/segm1_%d-4.png',(imagenumber/2));
imwrite(image_final_segm1,fnamesave);
fnamesave = sprintf('wyniki/segm2_%d-4.png',(imagenumber/2));
imwrite(image_final_segm2,fnamesave);

time_parfor(imagenumber/2) = toc;

end 