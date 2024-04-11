classdef Kernel
    
    properties
        kernels
        operations
        warping
        thetas
        scales
        scale
        w
        signn=0;
    end

    methods

        function obj = Kernel()

        end

        function D = dist(~,x1,x2)
            
            a = dot(x1,x1,2);
            b = dot(x2,x2,2);
            c = x1*x2';

            D = sqrt(abs(a + b' - 2*c));

        end

        function [x,dx] = warp(~,x,warping)

            switch warping.map
                case 'periodic'
                    P = warping.period;
                    dim = warping.dim;
                    x(:,dim) = sin(2*pi*x(:,dim)./P);
                    
                    dx = 0*x + 1;
                    dx(:,dim) = (2*pi./P).*cos(2*pi*x(:,dim)./P);

                case 'none'
                    x = x;
                    dx = 0*x + 1;
            end

        end


        function [K,dK] = build(obj,x1,x2)
            
            nk = numel(obj.kernels);
            
            xi1 = obj.warp(x1,obj.warping{1});
            xi2 = obj.warp(x2,obj.warping{1});

            if nargout>1
                [K,dK] = obj.kernels{1}.forward(xi1,xi2,obj.thetas{1});
                K = obj.scales{1}*K;
                dK = obj.scales{1}*dK;
            else
                K = obj.scales{1}*obj.kernels{1}.forward(xi1,xi2,obj.thetas{1});
            end

            for i = 2:nk

                xi1 = obj.warp(x1,obj.warping{i});
                xi2 = obj.warp(x2,obj.warping{i});

                switch obj.operations{i-1}
                    case '+'
                        if nargout>1
                            [K1,dK1] = obj.kernels{i}.forward(xi1,xi2,obj.thetas{i});
                            K = K + obj.scales{i}*K1;
                            dK = dK + obj.scales{i}*dK1;
                        else
                            K = K + obj.scales{i}*obj.kernels{i}.forward(xi1,xi2,obj.thetas{i});
                        end

                    case '-'
                        if nargout>1
                            [K1,dK1] = obj.kernels{i}.forward(xi1,xi2,obj.thetas{i});
                            K = K - obj.scales{i}*K1;
                            dK = dK - obj.scales{i}*dK1;
                        else
                            K = K - obj.scales{i}*obj.kernels{i}.forward(xi1,xi2,obj.thetas{i});
                        end

                    case '*'
                        if nargout>1
                            [K1,dK1] = obj.kernels{i}.forward(xi1,xi2,obj.thetas{i});
                            K = K.*obj.scales{i}.*K1;
                            dK = dK.*obj.scales{i}*dK1;
                        else
                            K = K.*obj.scales{i}.*obj.kernels{i}.forward(xi1,xi2,obj.thetas{i});
                        end

                     case '/'
                        if nargout>1
                            [K1,dK1] = obj.kernels{i}.forward(xi1,xi2,obj.thetas{i});
                            K = K./obj.scales{i}.*K1;
                            dK = dK./obj.scales{i}*dK1;
                        else
                            K = K./obj.scales{i}.*obj.kernels{i}.forward(xi1,xi2,obj.thetas{i});
                        end
                        
                end
            end

        end

        function dK = grad(obj,x1,x2,theta)

        end

        function V = getHPs(obj)
            V = cell2mat(obj.thetas);
            V = [V cell2mat(obj.scales)];
        end

        function obj = setHPs(obj,V)
            nT = numel(obj.thetas);
            nS = numel(obj.scales);

            for i = 1:nT
                nTs(i) = numel(obj.thetas{i});
                nSs(i) = numel(obj.scales{i});
            end

            obj.thetas = mat2cell(V(1:sum(nTs)),1,nTs);
            obj.scales = mat2cell(V(sum(nTs)+1:end),1,nSs);
        end

        function obj = plus(obj,K2)

            obj.kernels{end+1} = K2;
            obj.operations{end+1} = '+';
            obj.thetas{end+1} = K2.theta;
            obj.scales{end+1} = K2.scale;
            obj.warping{end+1} = K2.w;
        end

        function obj = subtract(obj,K2)

            obj.kernels{end+1} = K2;
            obj.operations{end+1} = '-';
            obj.thetas{end+1} = K2.theta;
            obj.scales{end+1} = K2.scale;
            obj.warping{end+1} = K2.w;
        end

        function obj = mtimes(obj,K2)

            obj.kernels{end+1} = K2;
            obj.operations{end+1} = '*';
            obj.thetas{end+1} = K2.theta;
            obj.scales{end+1} = K2.scale;
            obj.warping{end+1} = K2.w;
        end

        function obj = mrdivide(obj,K2)

            obj.kernels{end+1} = K2;
            obj.operations{end+1} = '/';
            obj.thetas{end+1} = K2.theta;
            obj.scales{end+1} = K2.scale;
            obj.warping{end+1} = K2.w;
        end

    end
end