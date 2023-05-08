% shortcut层,解决梯度发散问题
classdef shortcut_layer
    properties(Access=public)
        name='shortcut'
        from
        activation
        input
        output
    end
    
    methods(Access=public)
        % 构造函数
        function obj=shortcut_layer(from,activation)
            obj.from = from;
            obj.activation = activation;
        end
        
        % current_layer_num当前层的index,mynet整个神经网络
        function obj = forward(this, mynet, current_layer_num)
            this.output = this.input + mynet{current_layer_num + this.from}.output;
            % activation
            obj = this;
            
%             [h,w,c] = size(obj.output);
%             fprintf('shortcut_layer complete! shortcut_size [w,h,c]: [%d,%d,%d]\n', w,h,c);
        end
    end
    
    methods(Static)
    end
end