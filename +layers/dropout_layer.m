% dropout层
classdef dropout_layer
    properties(Access=public)
        name='dropout'
        mode % 表示训练还是测试
        probability
        rand_matrix % input大小的随机数生成
        input
        output
    end
    
    methods(Access=public)
        function obj = dropout_layer(prob)
            obj.probability = prob;
        end
        
        function obj = forward(this)
            % 非训练模式
            if ~strcmp(this.mode,'train')
                this.output = this.input;
            else
                this.rand_matrix = rand(size(this.input));
                this.output = this.input;
                this.output(this.rand_matrix<this.probability) = 0;
                this.output(this.rand_matrix>=this.probability) = this.output(this.rand_matrix>=this.probability)*(1 - this.probability);
            end
            
            obj = this;
            
%             [h,w,c] = size(obj.output);
%             fprintf('dropout_layer complete! dropout_size [w,h,c]: [%d,%d,%d]\n', w,h,c);
        end
    end
end