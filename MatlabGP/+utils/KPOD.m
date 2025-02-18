%{
    Kernal Proper Orthogonal Decomposition
%}

classdef KPOD<utils.POD
    
    properties
        kernel
    end

    methods

        function obj = KPOD(nmodes,kernel)
            
            obj=obj@utils.POD(nmodes);
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

    end         
end