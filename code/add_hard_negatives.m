function [ output_args ] = add_hard_negatives( fp, gt_bboxes, gt_ids, test_scn_path )
%ADD_HARD_NEGATIVES Summary of this function goes here
%   Detailed explanation goes here
    current_time = datestr(clock);
    for n = 1:size(fp)
        if fp(n) == 1
            x_start = int16(gt_bboxes(n,1));
            x_end = int16(gt_bboxes(n,3));
            y_start = int16(gt_bboxes(n,2));
            y_end = int16(gt_bboxes(n,4));
            if x_start > 0 && y_start > 0
                image_path = strcat( test_scn_path, '/', gt_ids(n) );

                image = imread(image_path{1});
                if y_end < size(image, 1) && x_end < size(image, 2)
                    negative_part = image(y_start:y_end, x_start:x_end);
                    negative_part = imresize(negative_part, [36 36]);
                    filename = strcat('../data/hard_negatives/', current_time, gt_ids(n), int2str(n), '.jpg');
                    imwrite(negative_part, filename{1});
                end
            end
        end
    end
    
    output_args = 1;
end

