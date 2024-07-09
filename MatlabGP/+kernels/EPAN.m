classdef EPAN<kernels.Kernel
    
    properties
        theta
    end

    methods

        function obj = EPAN(scale,theta)
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

            K = (0.75/prod(theta))*(1 - d.^2);
            K(d>1) = 0;

        end

    end
end