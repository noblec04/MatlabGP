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
        X=[];
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

        function [K,dK] = forward_(obj,x1,x2,theta)

        end

        function [K] = build(obj,x1,x2)
            
            nk = numel(obj.kernels);
            
            xi1 = obj.warp(x1,obj.warping{1});
            xi2 = obj.warp(x2,obj.warping{1});

            K = obj.scales{1}*obj.kernels{1}.forward_(xi1,xi2,obj.thetas{1});

            for i = 2:nk

                xi1 = obj.warp(x1,obj.warping{i});
                xi2 = obj.warp(x2,obj.warping{i});

                switch obj.operations{i-1}
                    case '+'

                        K = K + obj.scales{i}*obj.kernels{i}.forward_(xi1,xi2,obj.thetas{i});

                    case '*'

                        K = obj.scales{i}*K.*obj.kernels{i}.forward_(xi1,xi2,obj.thetas{i});
 
                end
            end

            K = obj.scale*K;

        end

        function K = buildReflect()

        end

        function dK = grad(obj,x1,x2)

            a = size(x1,1);
            b = size(x1,2);
            c = size(x2,1);

            dK = full(AutoDiffJacobianAutoDiff(@(x) obj.build(x,x2),x1));

            dK = squeeze(reshape(dK,[a c b]));

        end

        function obj = periodic(obj,dim,P)
            obj.w.period = P;
            obj.w.dim = dim;
            for i = 1:numel(obj.kernels)
                obj.warping{i} = obj.w;
                obj.warping{i}.map = 'periodic';
            end
        end

        function V = getHPs(obj)
            V = cell2mat(obj.thetas);
        end

        function obj = setHPs(obj,V)
            nT = numel(obj.thetas);

            for i = 1:nT
                nTs(i) = numel(obj.thetas{i});
            end

            obj.thetas = mat2cell(V(1:sum(nTs)),1,nTs);
        end

        function [lb,ub] = getHPBounds(obj)
            
        end

        function obj = plus(obj,K2)

            obj.kernels{end+1} = K2;
            obj.operations{end+1} = '+';
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

    end
end