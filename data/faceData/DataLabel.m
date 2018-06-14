clear
close all
% for i = 1:20
%     iStr = num2str(i);
%     im = imread([iStr,'.jpg']);
%     im = imresize(im, [227, 227]);
%     imwrite(im,['resize\',iStr,'.jpg']);
% end
landmark = zeros(20, 128, 2);
for i = 1:20
    iStr = num2str(i);
    landmark(i,:, :) = zeros(128,2);
    for j = 1:128
        x = rand() * 227;
        y = rand() * 227;
        landmark(i,j,:) = [x, y];
    end
    data = squeeze(landmark(i,:,:));
    save(['landMarkData\',iStr,'.mat'], 'data');
end

