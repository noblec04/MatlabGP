classdef EQ_special<kernels.Kernel
    
    properties
        theta
    end

    methods

        function obj = EQ_special(scale,theta)
            obj.scale = scale;
            obj.scales{1} = scale;
            obj.theta = theta;
            obj.thetas{1} = theta;
            obj.kernels{1} = obj;
            obj.w.map = 'none';
            obj.warping{1} = obj.w;
        end

        function D = dist(~,x1,x2)
            
            alphax = @(x) 3*(x).^2 - 2*(x).^3;

            a1 = alphax(x1(:,2)*2);
            a2 = alphax(x2(:,2)*2);

            a1(x1(:,2)>1/2)=1;
            a2(x2(:,2)>1/2)=1;

            D = zeros(size(x1,1),size(x2,1));

            for i = 1:size(x1,2)
                a = dot(x1(:,i),x1(:,i),2);
                b = dot(x2(:,i),x2(:,i),2);
                c = x1(:,i)*x2(:,i)';
                
                if i == 1
                    alph = a1*a2';
                    D = D + alph.*abs(sqrt(abs(a + b' - 2*c)));
                else
                    D = D + abs(sqrt(abs(a + b' - 2*c)));
                end
            end

        end

        function [K] = forward_(obj,x1,x2,theta)

            d = obj.dist(x1./theta,x2./theta);

            K = exp(-d.^2);

        end


        % function [K,dK] = forward(obj,x1,x2,theta)
        % 
        %     nD = size(x1,2);
        %     nT = numel(theta);
        % 
        %     d = obj.dist(x1./theta,x2./theta);
        % 
        %     K = exp(-d.^2);
        % 
        %     if nargout>1
        %         dK = zeros(size(K,1),size(K,2),nT);
        %         for i = 1:nT
        %             dK(:,:,i) = (2/theta(i))*((x1(:,i) - x2(:,i)').^2).*K;
        %         end
        %         dK(abs(dK(:,:,1:nT)) < 1e-12) = 0;
        %     end
        % end
    end
end