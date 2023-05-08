
% commond: [input_channels, channels] = get_channels(mn,3);
function [input_channels, channels] = get_channels(mn,layer_num)
    input_channels = {};
    for k=1:3
       input_channels{k,1} = mn.input(:,:,k); 
    end

    layerout = mn.mynet{layer_num,1}.output;
    channels = {};
    for k = 1:3
    %     figure(k)
        channels{k,1} = layerout(:,:,k);
    %     imshow(layerout(:,:,k),[]);
    end
end
