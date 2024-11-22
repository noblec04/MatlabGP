function [Mp] = BlockPInv(A,B,C,D,Ainv)

if nargin<5

    K = A'*A + B'*B;

    Kinv = pinv(K);

    E = A'*D + B'*C;

    R = D - A*Kinv*E;

    S = C - B*Kinv*E;

    L = R'*R + S'*S;

    Linv = pinv(L);

    T = Kinv*E*(eye(size(L)) - Linv*L);

    F = Linv*R' + (eye(size(L)) - Linv*L)*(eye(size(L)) - T'*T)\((Kinv*E)'*Kinv*(A' - E*Linv*R'));

    H = Linv*S' + (eye(size(L)) - Linv*L)*(eye(size(L)) - T'*T)\((Kinv*E)'*Kinv*(B' - E*Linv*S'));

    Mp = [Kinv*(A' - E*F) Kinv*(B' - E*H);
        F                 H         ];

else

    H = pinv(C - B*Ainv*D);

    Mp = [Ainv + Ainv*D*H*B*Ainv -Ainv*D*H;
        -H*B*Ainv    H];
end
end