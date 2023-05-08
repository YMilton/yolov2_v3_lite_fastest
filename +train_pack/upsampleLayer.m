classdef upsampleLayer < nnet.layer.Layer
    properties(Access=public)
        stride
    end
    
    methods
        function obj = upsampleLayer(name,stride)
            obj.Name = name;
            obj.stride = stride;
            obj.Type = 'upsample';
        end
        
        function Z = predict(this,X)
            Z = repelem(X,this.stride,this.stride);
        end
        
        % 反向传播override backward(aLayer, X, Z, dZ, memory)
        function [dX] = backward(this,~,~,dZ,~)
            dX = dZ(1:this.stride:end,1:this.stride:end,:,:);
        end
    end
end