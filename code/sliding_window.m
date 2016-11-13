function [ bboxes, confidences, image_ids ] = sliding_window( image, filename, feature_params, w, b )
%SLIDING_WINDOW Summary of this function goes here
%   Detailed explanation goes here
    bboxes = zeros(0,4);
    confidences = zeros(0,1);
    image_ids = cell(0,1);
    
    for n = 1:6:(size(image, 1) - feature_params.template_size)
        for m = 1:6:(size(image, 2) - feature_params.template_size)
           %window = imcrop(image, [m n feature_params.template_size-1 feature_params.template_size-1]);
           window = image(n:n+feature_params.template_size-1, m:m+feature_params.template_size-1);
           hog = vl_hog(window, feature_params.hog_cell_size);
           hog_vec = transpose(hog(:));
           
           confidence = hog_vec * w + b;
           if confidence >= 1
              cur_x_min = m;
              cur_y_min = n;
              cur_bbox = [cur_x_min, cur_y_min, cur_x_min + feature_params.template_size, cur_y_min + feature_params.template_size];
              cur_confidence = confidence;
              cur_image_id = filename;
              
              bboxes = [bboxes; cur_bbox];
              confidences = [confidences; cur_confidence];
              image_ids = [image_ids; cur_image_id];
           end
        end
    end
end

