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

            D = abs(sqrt(abs(a + b' - 2*c)));

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
                dKs{1} = obj.scales{1}*dK;
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
                            dKs{i} = obj.scales{i}*dK1;
                        else
                            K = K + obj.scales{i}*obj.kernels{i}.forward(xi1,xi2,obj.thetas{i});
                        end

                    case '*'
                        if nargout>1
                            [K1,dK1] = obj.kernels{i}.forward(xi1,xi2,obj.thetas{i});

                            for kk = 1:size(dK1,3)
                                dki(:,:,kk) = obj.scales{i}*K.*dK1(:,:,kk);
                            end

                            if i>1
                                for jj = 1:i-1
                                    dKs{jj} = K1.*dKs{jj};
                                end
                            end

                            dKs{i} = dki;

                            K = K.*obj.scales{i}.*K1;
                        else
                            K = obj.scales{i}*K.*obj.kernels{i}.forward(xi1,xi2,obj.thetas{i});
                        end
                        
                end
            end

            K = obj.scale*K;

            if nargout>1
                jj = 1;
                for i = 1:numel(dKs)
                    nn = size(dKs{i},3);
                    dK(:,:,jj:jj+nn-1) = obj.scale*cell2mat(dKs(i));
                    jj=jj+nn;
                end
            end

        end

        function dK = grad(obj,x1,x2)

            a = size(x1,1);
            b = size(x1,2);
            c = size(x2,1);

            dK = full(AutoDiffJacobianAutoDiff(@(x) obj.build(x,x2),x1));

            dK = reshape(dK,[a c b]);

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