classdef GLOW

    properties
        c
        sigma
    end

    methods

        function obj = GLOW(c,sigma)

            obj.c = c;
            obj.sigma = sigma;

        end

        function [y] = forward(obj,x)

            sx = sign(x);

            A = x.^2/2;

            B = (obj.sigma/2)*(1 - exp((obj.c - x.^2)/obj.sigma));

            y = A;

            if isa(x,'AutoDiff')
                xv = getvalue(x);
            else
                xv = x;
            end

            y(abs(xv)>obj.c) = B(abs(xv)>obj.c);

            y = sx.*y;

        end
    end
end