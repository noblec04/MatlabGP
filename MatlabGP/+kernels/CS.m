classdef CS<kernels.Kernel
    
    properties
        theta
        dist
    end

    methods

        function obj = CS(scale,theta,dist)
            obj.scale = scale;
            obj.scales{1} = scale;
            obj.theta = theta;
            obj.thetas{1} = theta;
            obj.w.map = 'none';
            obj.warping{1} = obj.w;
            obj.dist = dist;
            obj.kernels{1} = obj;
        end

        function [K] = forward_(obj,x1,x2,theta)

            d = pdist2(x1./theta,x2./theta,obj.dist);

            K = exp(-d.^2);

        end

        
        
    end
end