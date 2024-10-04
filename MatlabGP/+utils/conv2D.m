function [B] = conv2D(S,F,b)

[nFx,nFy] = size(F);
[nS,nX,nY] = size(S);

padded_S = zeros(nS, nX + 2 * (nFX - 1),nY + 2 * (nFy - 1),'like',S);
padded_S(:, nFx:end - nFx + 1,nFy:end - nFy + 1) = S;

for i=1:nX
    for j = 1:nY
        B(:,i,j) = sum(padded_S(:,i:i+nFx-1,j:j+nFy-1)*F,2) + b;
    end
end

end