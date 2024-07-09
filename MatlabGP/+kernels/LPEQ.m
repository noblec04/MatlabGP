classdef LPEQ<kernels.Kernel
    
    properties
        theta
    end

    methods

        function obj = LPEQ(scale,theta)
            obj.scale = scale;
            obj.scales{1} = scale;
            obj.theta = theta;
            obj.thetas{1} = theta;
            obj.kernels{1} = obj;
            obj.w.map = 'none';
            obj.warping{1} = obj.w;
        end

        function [K] = forward_(obj,x1,x2,theta)

            P = theta(1);

            theta1 = theta(2);
            theta2=theta(3:end);

            d1 = obj.dist(x1,x2)/P;
            d2 = obj.dist(x1./theta2,x2./theta2);

            K = exp(-2*(sin(pi*d1).^2)./theta1.^2).*exp(-d2.^2);
        end
        
    end
end