function [B] = conv1D(S,F,b)

[nF] = size(F,1);
[nS,nX] = size(S);

padded_S = zeros(nS, nX + 2 * (nF - 1),'like',S);
padded_S(:, nF:end - nF + 1) = S;

for i=1:nX
    B(:,i) = sum(padded_S(:,i:i+nF-1)*F,2) + b;
end

end