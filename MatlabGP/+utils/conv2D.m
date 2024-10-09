function [B] = conv2D(S,F,b)

[nFx,nFy] = size(F);
[nS,nX,nY] = size(S);

padded_S = zeros(nS, nX + 2 * (nFx - 1),nY + 2 * (nFy - 1),'like',S);
padded_S(:, nFx:end - nFx + 1,nFy:end - nFy + 1) = S;

% Fs = zeros(nS,nFx,nFy,'like',F);
% 
% for i = 1:nS
%     Fs(i,:,:) = F;
% end

for k = 1:nS
for i=1:nX
    for j = 1:nY
        aa = squeeze(padded_S(k,i:i+nFx-1,j:j+nFy-1))*F;
        B(k,i,j) = sum(sum(aa,1),2) + b;
    end
end
end

end