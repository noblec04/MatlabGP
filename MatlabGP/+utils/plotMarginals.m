function vp = plotMarginals(Z2,res,regress)

[~,ntm,~,tm0,tk0] = Z2.getHPs();

tmlb = 0*tm0 - 5;
tmub = 0*tm0 + 5;

tklb = 0*tk0 + 0.01;
tkub = 0*tk0 + 5;

tlb = [tmlb tklb];
tub = [tmub tkub];


[vp] = vbmc(@(x) Z2.LL(x,regress,ntm),[],tlb,tub,tlb,tub);

Xs = vbmc_rnd(vp,res);

cornerplot(Xs);

end