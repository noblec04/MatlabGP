classdef NLL

    methods

        function obj = NLL()


        end

        function [e,de] = forward(~,y,yp)

            mu = yp(:,1);
            sig = exp(yp(:,2));

            muy = y(:,1);
            sigy = y(:,2);

            e = log(sig+eps) + (sigy-sig).^2 + ((muy-mu).^2)./(sig.^2+eps);

            de(1,:) = -2*(muy-mu)./(sig.^2+eps);

            de(2,:) = 1 - 2*(sigy-sig).*sig - 2*((muy-mu).^2)./(sig.^3+eps);

        end
    end
end