%{
    Robust Proper Orthogonal Decomposition
%}

classdef RPOD<utils.POD

    methods

        function obj = RPOD(nmodes)
            
            obj=obj@utils.POD(nmodes);

        end

        function obj = train(obj,X)


            obj.lb = min(X,[],'all');
            obj.ub = max(X,[],'all');
            
            X = (X - obj.lb)./(obj.ub-obj.lb);

            obj.Y = X;

            L = utils.RPCA(obj.Y(:,:));

            [obj.phi,obj.score,~,~,obj.explained,obj.mu] = pca(L);

        end
        
    end         
end