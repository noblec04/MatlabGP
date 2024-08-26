classdef Agent
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here

    properties
        state
        Q
    end

    methods
        function obj = Agent(Q,S)
            
            obj.Q = Q;
            obj.state = S;

        end

        function a = action(obj,state)
            
            a = obj.Q.eval(state);

        end

        function obj = train(obj,R)

            obj.Q = obj.Q.train(R);

        end
    end
end