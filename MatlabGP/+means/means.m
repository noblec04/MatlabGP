classdef means

    properties
        meanz
        coeffs
        operations
    end

    methods

        function y = forward(obj,x)

        end

        function dm = grad(obj,x1)

            a = size(x1,1);
            b = size(x1,2);

            dm = full(AutoDiffJacobianAutoDiff(@(x) obj.eval(x),x1));
            dm = squeeze(reshape(dm,[a b]));

        end

        function [y,dY] = eval(obj,x)

            nb = numel(obj.meanz);

            if nargout>1
                [y,dy{1}] = obj.meanz{1}.forward(x,obj.coeffs{1});
            else
                y = obj.meanz{1}.forward(x,obj.coeffs{1});
            end

            for i = 2:nb
                switch obj.operations{i-1}
                    case '+'
                        if nargout>1
                            [y1,dy1] = obj.meanz{i}.forward(x,obj.coeffs{i});
                            y = y + y1;
                            dy{i} = dy1;
                        else
                            y = y + obj.meanz{i}.forward(x,obj.coeffs{i});
                        end

                    case '-'
                        if nargout>1
                            [y1,dy1] = obj.meanz{i}.forward(x,obj.coeffs{i});
                            y = y - y1;
                            dy{i} = dy1;
                        else
                            y = y - obj.meanz{i}.forward(x,obj.coeffs{i});
                        end

                    case '*'
                        if nargout>1
                            [y1,dy1] = obj.meanz{i}.forward(x,obj.coeffs{i});
                            dy{i} = y.*dy1;
                            y = y.*y1;
                        else
                            y = y.*obj.meanz{i}.forward(x,obj.coeffs{i});
                        end

                     case '/'
                        if nargout>1
                            [y1,dy1] = obj.meanz{i}.forward(x,obj.coeffs{i});
                            
                            dy{i} = -1*y.*(dy1)./y1.^2;

                            if i>1
                                for jj = 1:i-1
                                    dy{jj} = y1.*dy{jj};
                                end
                            end

                            y = y./y1;
                        else
                            y = y./(obj.meanz{i}.forward(x,obj.coeffs{i}));
                        end
                        
                end
            end

            if nargout>1
                jj = 1;
                for i = 1:numel(dy)
                    nn = size(dy{i},2);
                    dY(:,jj:jj+nn-1) = cell2mat(dy(i));
                    jj=jj+nn;
                end
            end

        end

        function V = getHPs(obj)
            V = cell2mat(obj.coeffs);
        end

        function obj = setHPs(obj,V)

            nT = numel(obj.coeffs);

            for i = 1:nT
                nTs(i) = numel(obj.coeffs{i});
            end

            obj.coeffs = mat2cell(V(1:sum(nTs)),1,nTs);
            
        end

        function obj = plus(obj,M2)

            obj.meanz{end+1} = M2;
            obj.operations{end+1} = '+';
            obj.coeffs{end+1} = M2.coeffs{1};
        end

        function obj = subtract(obj,M2)

            obj.meanz{end+1} = M2;
            obj.operations{end+1} = '-';
            obj.coeffs{end+1} = M2.coeffs{1};
        end

        function obj = mtimes(obj,M2)

            obj.meanz{end+1} = M2;
            obj.operations{end+1} = '*';
            obj.coeffs{end+1} = M2.coeffs{1};

        end

        function obj = mrdivide(obj,M2)

            obj.meanz{end+1} = M2;
            obj.operations{end+1} = '/';
            obj.coeffs{end+1} = M2.coeffs{1};
        end
    end
end