classdef CE

    properties

    end

    methods

        function obj = CE()

        end

        function [e] = forward(~,y,yp)

            P = exp(yp)./sum(exp(yp),2);

            e = -1*dot(y,log(P));

        end
    end
end