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

        function [y] = eval(obj,x)

            nb = numel(obj.meanz);

            y = obj.meanz{1}.forward(x,obj.coeffs{1});

            for i = 2:nb
                switch obj.operations{i-1}
                    case '+'
                        
                        y = y + obj.meanz{i}.forward(x,obj.coeffs{i});

                    case '-'

                        y = y - obj.meanz{i}.forward(x,obj.coeffs{i});

                    case '*'

                        y = y.*obj.meanz{i}.forward(x,obj.coeffs{i});

                     case '/'
                         
                         y = y./(obj.meanz{i}.forward(x,obj.coeffs{i}));
                        
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