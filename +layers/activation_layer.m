
% 激活层
classdef activation_layer
   
    methods(Static)
       % leaky_relu激活函数
       % leaky(input) or leaky(input,scale)
        function output = leaky(varargin)
            if nargin==1
                scale = 0.1;
            else
                scale = varargin{2};
            end
            xx = varargin{1};
            out_x = xx.*(xx>0) + scale*xx.*(xx<0);
            output = out_x;
        end
        
        % linear激活函数
        function output = linear(input)
            output = input;
        end
        
        % 反向传播，求梯度 xx为卷积后结果
        % leaky_backward(delta,xx) or leaky_backward(delta,xx,scale)
        function update_delta = leaky_backward(varargin)
            if nargin>=2 && nargin<=3
                delta = varargin{1};
                xx = varargin{2};
                if nargin==2
                    scale = 0.1;
                    delta_leaky = double(xx>0) + scale*double(xx<0);
                    delta = delta.*delta_leaky;
                else
                    scale = varargin{3};
                    delta_leaky = double(xx>0) + scale*double(xx<0);
                    delta = delta.*delta_leaky;
                end
                update_delta = delta;
            end
        end
        
        function update_delta = linear_backward(delta)
            update_delta = delta;
        end
    end
    
end