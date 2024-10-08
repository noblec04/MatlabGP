classdef RQ<kernels.Kernel
    
    properties
        theta
        alpha
    end

    methods

        function obj = RQ(alpha,scale,theta)
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
        
            K = (1 + (d.^2)/(2*obj.alpha)).^(-1*obj.alpha);
            
        end
    end
end