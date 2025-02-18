%{
    Dynamic Mode Decomposition
%}

classdef DMD
    
    properties
        Y

        lb
        ub

        nmodes

        mu
        score
        phi
        explained

    end

    methods

        function obj = DMD(nmodes)
            
            obj.nmodes = nmodes;

        end

        function obj = train(obj,X)


            obj.lb = min(X,[],'all');
            obj.ub = max(X,[],'all');
            
            X = (X - obj.lb)./(obj.ub-obj.lb);

            obj.Y = X;

            L = obj.Y(:,:)';

            L1 = L(:,1:end-1);
            L2 = L(:,2:end);

            [U, S, V]=svd(L1,0);

            E = diag(S).^2;

            U=U(:,1:obj.nmodes);
            V=V(:,1:obj.nmodes);
            S=S(1:obj.nmodes,1:obj.nmodes);

            A_tilde=U'*L2*V/S;

            [eVecs, Eigenvalues] = eig(A_tilde);

            Eigenvectors=Y*V*inv(S)*eVecs;
            Eigenvalues=diag(Eigenvalues);

            ModeAmplitudes=Eigenvectors\X(:,1);

            [obj.phi,obj.score,~,~,obj.explained,obj.mu] = pca(L);

        end

        function y = eval(obj,coeffs)
            
            A = size(obj.Y);
            
            A(1) = size(coeffs,1);
            
            y = reshape(coeffs(:,1:obj.nmodes) * obj.phi(:,1:obj.nmodes)' + obj.mu,A);
            
            y = obj.lb + (obj.ub - obj.lb)*y;
        
            y = squeeze(y);

        end

        function y = eval_var(obj,coeffs)
            
            A = size(obj.Y);
            
            A(1) = size(coeffs,1);
            
            y = reshape(coeffs(:,1:obj.nmodes) * (obj.phi(:,1:obj.nmodes).^2)',A);
            
            y = ((obj.ub - obj.lb).^2)*y;
        
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

        function p = getSensors(obj,r)

            if obj.nmodes==r

                [~,~,P] = qr(obj.phi','econ','vector');

                p = squeeze(double(P<=r));

            elseif r>obj.nmodes
                
                [~,~,P] = qr(obj.phi*obj.phi','econ','vector');

                p = squeeze(double(P<=r));

            end

        end

        function y = reconstruct_from_sensors(obj,r)

            p = obj.getSensors(r);

            A = obj.Y(:,:)';

            coeffs = obj.phi(p==1,:)\A(p==1,:);

            y = obj.phi*coeffs;

            y = reshape(y',size(obj.Y));

        end

        function y = reconstruct(obj,index,num_modes)

            if nargin<3
                num_modes = obj.nmodes;
            end

            A = size(obj.Y);
            
            A(1) = 1;

            if size(obj.score,2)<num_modes
                warning('Requested number of modes not available')
                num_modes = size(obj.score,2);
            end
            
            y = reshape(obj.score(index,1:num_modes) * obj.phi(:,1:num_modes)' + obj.mu,A);
            
            y = obj.lb + (obj.ub - obj.lb)*y;
        
            y = squeeze(y);

        end

    end         
end