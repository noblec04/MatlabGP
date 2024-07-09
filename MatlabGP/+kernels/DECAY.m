classdef DECAY<kernels.Kernel
    
    properties
        theta
        alpha
        beta
    end

    methods

        function obj = DECAY(alpha, beta, scale,theta)

            obj.alpha = alpha;
            obj.beta = beta;
            obj.scale = scale;
            obj.scales{1} = scale;
            obj.theta = theta;
            obj.thetas{1} = theta;
            obj.kernels{1} = obj;
            obj.w.map = 'none';
            obj.warping{1} = obj.w;
        end

        function [K] = forward_(obj,x1,x2,theta)

            d = obj.dist(x1./theta,x2./theta);

            K = (abs(obj.beta).^obj.alpha)./(abs(d+obj.beta).^obj.alpha);

        end

    end
end