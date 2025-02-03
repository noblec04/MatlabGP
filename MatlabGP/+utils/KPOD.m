%{
    Kernal Proper Orthogonal Decomposition
%}

classdef KPOD
    
    properties
        Y
        kernel

        lb
        ub

        nmodes

        mu
        score
        phi
        explained

    end

    methods

        function obj = KPOD(nmodes,kernel)
            
            obj.nmodes = nmodes;
            obj.kernel = kernel;

        end

        function obj = train(obj,X)

            obj.lb = min(X,[],'all');
            obj.ub = max(X,[],'all');
            
            X = (X - obj.lb)./(obj.ub-obj.lb);

            obj.Y = X;

            L = obj.Y(:,:);

            [obj.phi,obj.score,obj.explained,obj.mu] = utils.kpca(L,obj.nmodes,obj.kernel);

        end

        function y = eval(obj,coeffs)
            
            A = size(obj.Y);
            
            A(1) = size(coeffs,1);
            
            y = reshape(coeffs(:,1:obj.nmodes) * obj.phi(:,1:obj.nmodes)' + obj.mu,A);
            
            y = obj.lb + (obj.ub - obj.lb)*y;
        
            y = squeeze(y);

        end

        function score = project(obj,X)
           
            X = X(:)' - obj.mu;
            
            score = X*obj.phi(:,1:obj.nmodes);
            
        end

        function plotModes(obj,i)
           
            A = size(obj.Y);
            
            A(1) = 1;

            if i~=0

                y = squeeze(reshape(obj.phi(:,i)' ,A));

            else

                y = squeeze(reshape(obj.mu(:)' ,A));
               
            end
            
            if length(A)==3
            
                pcolor(y)
                shading interp
                utils.cmocean('thermal')
                
            elseif length(A)==4
                
                nn = 10;
                vals = linspace(min(squeeze(y(:))),max(squeeze(y(:))),nn);
                for i = 1:nn
                    isosurface(squeeze(y),vals(i));
                end
                view([1 1 1])
                
            end
            
        end

    end         
end