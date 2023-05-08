classdef mishLayer < nnet.layer.Layer
    % mish激活函数优于ReLU
    % reference: 《Mish: A Self Regularized Non-Monotonic Neural Activation Function》
    
    methods
        function obj = mishLayer(name)
            obj.Name = name;
            obj.Description = 'activation mishLayer';
            obj.Type = 'activation';
        end
        
        function Z = predict(~,X)
            Z = X.*tanh(log(1 + exp(X)));
        end
    end
end