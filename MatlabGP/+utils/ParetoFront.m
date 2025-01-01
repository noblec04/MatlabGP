function [A] = ParetoFront(Y,maxmin)

if nargin==1
    %Bigger is better by default
    maxmin = 1;
end
Y = -1*maxmin*Y;

nY = size(Y,1);


A = zeros(nY,1);

for i = 1:nY
    
    A(i) = all(any(Y(1:i-1,:)>Y(i,:),2)) && all(any(Y(i+1:end,:)>Y(i,:),2));

end



end