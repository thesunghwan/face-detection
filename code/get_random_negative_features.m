% Starter code prepared by James Hays for CS 143, Brown University
% This function should return negative training examples (non-faces) from
% any images in 'non_face_scn_path'. Images should be converted to
% grayscale, because the positive training data is only available in
% grayscale. For best performance, you should sample random negative
% examples at multiple scales.

function features_neg = get_random_negative_features(non_face_scn_path, feature_params, num_samples)
% 'non_face_scn_path' is a string. This directory contains many images
%   which have no faces in them.
% 'feature_params' is a struct, with fields
%   feature_params.template_size (probably 36), the number of pixels
%      spanned by each train / test template and
%   feature_params.hog_cell_size (default 6), the number of pixels in each
%      HoG cell. template size should be evenly divisible by hog_cell_size.
%      Smaller HoG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality increases and more
%      importantly the step size of the classifier decreases at test time.
% 'num_samples' is the number of random negatives to be mined, it's not
%   important for the function to find exactly 'num_samples' non-face
%   features, e.g. you might try to sample some number from each image, but
%   some images might be too small to find enough.

% 'features_neg' is N by D matrix where N is the number of non-faces and D
% is the template dimensionality, which would be
%   (feature_params.template_size / feature_params.hog_cell_size)^2 * 31
% if you're using the default vl_hog parameters

% Useful functions:
% vl_hog, HOG = VL_HOG(IM, CELLSIZE)
%  http://www.vlfeat.org/matlab/vl_hog.html  (API)
%  http://www.vlfeat.org/overview/hog.html   (Tutorial)
% rgb2gray

image_files = dir( fullfile( non_face_scn_path, '*.jpg' ));
num_images = length(image_files);


features_neg = [];
i = 0;
while i < num_samples
    n = randi(num_images);
    absolute_path = strcat(non_face_scn_path, '/', image_files(n).name);
    negative_image = imread(absolute_path);
    negative_image = single(negative_image)/255;
    if(size(negative_image,3) > 1)
        negative_image = rgb2gray(negative_image);
    end
    
    negative_image = imresize(negative_image, (1 - 0.3)*rand(1) + 0.3);
    hog = vl_hog(negative_image, feature_params.hog_cell_size);
    
    
    hog_size = feature_params.template_size / feature_params.hog_cell_size;

    if size(hog, 1) > hog_size && size(hog, 2) > hog_size
        for n = 1:50
            x = size(hog, 1) - hog_size;
            y = size(hog, 2) - hog_size;

            random_x_position = randi(x);
            random_y_position = randi(y);
            
            image = hog(random_x_position:random_x_position+hog_size-1, random_y_position:random_y_position+hog_size-1, :);
            hog_vec = transpose(image(:));
            features_neg = [features_neg; hog_vec];
            i = i + 1;
        end
    end

    %random_x_position = randi(size(negative_image, 2) - 36);
    %random_y_position = randi(size(negative_image, 1) - 36);

    %negative_image = imcrop(negative_image, [random_x_position random_y_position 35 35]);

    %hog = vl_hog(negative_image, feature_params.hog_cell_size);
    %hog_vec = transpose(hog(:));
    %features_neg = [features_neg; hog_vec];
    %i = i + 1;
end

% placeholder to be deleted
%features_neg = rand(100, (feature_params.template_size / feature_params.hog_cell_size)^2 * 31);