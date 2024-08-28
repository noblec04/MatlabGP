function [B] = conv2D(A,F,b)

[m,~] = size(F);
[y, x] = size(A);
y = y - m + 1;
x = x - m + 1;
B = zeros(y,x,'like',F);
for i=1:y
    for j=1:x
        B(i,j) = sum(sum(A(i:i+m-1, j:j+m-1)*F)) + b;
    end
end

end