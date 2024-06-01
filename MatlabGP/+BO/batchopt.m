function [x,R] = batchopt(FF,Z,q)

for i = 1:q
    [x(i,:),R(i)] = BO.argmin(FF,Z);

    [y,sig] = Z.eval(x(i,:));

    Z = Z.resolve(x(i,:),y-2*sqrt(abs(sig)));
end

end