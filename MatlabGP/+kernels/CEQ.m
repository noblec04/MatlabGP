classdef CEQ<kernels.Kernel
    
    properties
        theta
        alpha
    end

    methods

        function obj = CEQ(alpha,scale,theta)
            obj.alpha = alpha;
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

            K = (1 - erf(d*obj.alpha/4)).*exp(-d.^2);

        end
    end
end