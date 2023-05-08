classdef reorgLayer < nnet.layer.Layer
    
    properties(Access=public)
        stride
    end
    
    methods
        function obj = reorgLayer(name,stride)
            obj.Name = name;
            obj.stride = stride;
        end
        
        % 前向传播调用predict函数
        function Z = predict(this,X) 
            X = reshape(X,[size(X),1]); % add one dim,ensure 4 dim 
            tmp = permute(X,[2,1,3,4]);
            [rows,cols,channels,~] = size(tmp);
            tmp = reshape(tmp,rows/this.stride, this.stride, cols/this.stride, this.stride, channels,[]);
            tmp = permute(tmp,[1,3,2,4,5,6]);
            Z = reshape(tmp,rows/this.stride, cols/this.stride, this.stride^2*channels,[]);
        end
        
    end
end

