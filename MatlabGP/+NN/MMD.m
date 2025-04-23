classdef MMD

    properties
        sigma = 1;
    end

    methods

        function obj = MMD(sigma)

            if nargin==1
                obj.sigma = sigma;
            end

        end

        function [e] = forward(obj,x,y)

            xx = obj.dist(x,x);
            yy = obj.dist(y,y);
            xy = obj.dist(x,y);

            Kxx = exp(-xx/(2*obj.sigma));
            Kyy = exp(-yy/(2*obj.sigma));
            Kxy = exp(-xy/(2*obj.sigma));

            e = sum(sum(Kxx) + sum(Kyy) - 2*sum(Kxy));

        end

        function D = dist(~,x1,x2)
            
            a = dot(x1,x1,2);
            b = dot(x2,x2,2);
            c = x1*x2';

            D = sqrt(abs(a + b' - 2*c));

        end
    end
end