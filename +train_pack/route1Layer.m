% route中存在组的形式的网络层构造
classdef route1Layer < nnet.layer.Layer
    
    properties(Access=public)
        layers
        groups
        group_id
    end
    
    methods
        function obj = route1Layer(name,layers)
            obj.Name = name;
            obj.layers = layers;
            obj.Type = 'route';
        end
        
        
        function Z = predict(alayer,X)
            if ~isempty(alayer.groups)
                X = reshape(X,[size(X),1]);
                channels = size(X,3);
                channels_per_group = channels/alayer.groups;
                start_idx = (alayer.group_id - 1)*channels_per_group + 1;
                end_idx = alayer.group_id*channels_per_group;
                Z = X(:,:,start_idx:end_idx,:);
            else
                Z = X;
            end
        end
        
        % backward(aLayer, X, Z, dZ, memory)
        function dX = backward(this,X,~,dZ,~)
            if ~isempty(this.groups)
                X = reshape(X,[size(X),1]);
                channels = size(X,3);
                channels_per_group = channels/this.groups;
                start_idx = (this.group_id - 1)*channels_per_group + 1;
                end_idx = this.group_id*channels_per_group;
                dX = single(zeros(size(X)));
                dX(:,:,start_idx:end_idx,:) = dZ;
            else
                dX = dZ;
            end
        end
    end
    
end