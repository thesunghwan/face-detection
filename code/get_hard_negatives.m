function [ hard_negatives ] = get_hard_negatives( non_face_scn_path, feature_params, num_samples, w, b )
%GET_HARD_NEGATIVES Summary of this function goes here
%   Detailed explanation goes here
image_files = dir( fullfile( non_face_scn_path, '*.jpg' ));
num_images = length(image_files);

hard_negatives = [];
i = 0;
ratio_reduction = 0.09;
threshold = 0;
hog_size = feature_params.template_size / feature_params.hog_cell_size;

while i < num_samples
    n = randi(num_images);
    absolute_path = strcat(non_face_scn_path, '/', image_files(n).name);
    negative_image = imread(absolute_path);
    negative_image = single(negative_image)/255;
    if(size(negative_image,3) > 1)
        negative_image = rgb2gray(negative_image);
    end
    
    j = 1;
    resized_img = negative_image;
    while size(resized_img, 1) > 36 && size(resized_img, 2) > 36
        img_hog = vl_hog(resized_img, feature_params.hog_cell_size);

        for m = 1:(size(img_hog, 1) - hog_size)
            for n = 1:(size(img_hog, 2) - hog_size)
                hog_window = img_hog(m:m+hog_size-1, n:n+hog_size-1, :);
                hog_window_vec = transpose(hog_window(:));
                confidence = hog_window_vec * w + b;
                if confidence > threshold
                    hard_negatives = [hard_negatives; hog_window_vec];
                    i = i + 1;
                end
            end
        end
        j = j - ratio_reduction;
        resized_img = imresize(negative_image, j,'Antialiasing',true);
    end
    
    
end


end

