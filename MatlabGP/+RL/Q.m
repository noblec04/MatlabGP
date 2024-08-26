classdef Q
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here

    properties
        actions
        states
        table

        alpha = 0.1
        gamma = 0.1
    end

    methods
        function obj = Q(actions,states)
            
            obj.actions = actions;
            obj.states = states;

            [A,~] = meshgrid(actions,states);

            obj.table = 0*A;

        end

        function a = eval(obj,state)
            
            is = find(obj.states == state);

            [~,ia] = max(obj.table(:,is));

            a = obj.actions(ia);

        end

        function obj = train(obj,a,s,R)

            is = find(obj.states == s);
            ia = find(obj.actions == a);

            obj.table(ia,is) = obj.table(ia,is) + obj.alpha*(R + obj.gamma*max(obj.table(:,is)) - obj.table(ia,is));

        end
    end
end