
% Batch Normalize 
classdef batch_norm_layer
    properties(Access=public)
        name='batch_norm'
        filters
        input % [rows*cols*c*batch]
        
        % 模型类型与动量
        mode
        momentum
        
        % 学习到的gamma，beat
        gamma 
        beta
        
        % 训练是计算的均值与方差
        bn_mean
        bn_var
        % 滑动均值与方差，训练过程得到的值
        rolling_mean
        rolling_var
        x_hat % 归一化的像素值
        output
        
        delta
        dinput
        dgamma
        dbeta
    end
    
    
    methods(Access=public)
        % batch_norm_layer() or batch_norm_layer(filters)
        function obj=batch_norm_layer(varargin)
            if nargin==0
                return;
            elseif nargin==1
                % 初始化beta和gamma
                obj.filters = varargin{1};
                obj.gamma = normrnd(1,0.02,[1,1,obj.filters]);
                obj.beta = zeros(1,1,obj.filters);
                obj.rolling_mean = zeros(1,1,obj.filters);
                obj.rolling_var = zeros(1,1,obj.filters);
                obj.momentum = 0.9;
            else
                disp('Please input right parameters(batch_norm)!')
                return;
            end
        end
        
        
        % gamma,beta: 表示有通道个这样的值
        function obj = forward(this)    
            eps = 1e-5;
            if strcmp(this.mode,'train')
                % batch*rows*cols不同通道的均值与方差 size(this.input)=[rows,cols,filters,batch]
                this.bn_mean = mean(this.input,[1,2,4]);
                this.bn_var = var(this.input,0,[1,2,4]);
                % 归一化
                this.x_hat = (this.input - this.bn_mean)./sqrt(this.bn_var + eps);
                % 滑动均值与方差计算,训练时计算，测试时使用
                this.rolling_mean = this.momentum*this.rolling_mean + (1 - this.momentum)*this.bn_mean;
                this.rolling_var = this.momentum*this.rolling_var + (1 - this.momentum)*this.bn_var;

                this.output = this.gamma .* this.x_hat + this.beta;
            elseif strcmp(this.mode,'test') % conv与bn融合后不使用
                % 根据weight文件的权重计算归一化
                this.x_hat = (this.input - this.rolling_mean)./sqrt(this.rolling_var + eps);
                this.output = this.gamma .* this.x_hat + this.beta; 
            else
                fprintf('invalid forward batchnorm mode %s!\n',this.mode);
            end
            obj = this;
        end
        
        % 一个batch的反向传播
        function obj = backward(this)
            eps = 1e-5;
            m = size(this.input,1)*size(intput,2)*size(this.input,4);
            this.dgamma = sum(this.delta.*this.x_hat,[1,2,4]);
            this.dbeta = sum(this.delta,[1,2,4]);
            % 更新gamma与beta
            this.gamma = this.gamma - this.dgamma;
            this.beta = this.beta - this.dbeta;
            
            dx_hat = this.delta.*this.gamma;
            dsigma = sum(dx_hat.*(this.input - this.bn_mean),[1,2,4]).*(-0.5*(this.bn_var+eps).^(-1.5));
            dmu = sum(dx_hat.*(-1./(sqrt(this.bn_var + eps))),[1,2,4]);
            this.dinput = dx_hat.*(1./sqrt(this.bn_var + eps)) + dsigma.*(2*(this.input - this.bn_mean)/m) + dmu/m;
            
            obj = this;
        end
    end
end