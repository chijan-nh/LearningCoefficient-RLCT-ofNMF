function [ lamNMFubd1,lamNMFubd2 ] =  lam_NMFubd(M,H,N,H_0)
    %calculation of upper bound of real log canonical threshold of NMF.
    %param M: number of rows (length of each column) in observed matrices
    %param N: number of columns (length of each row) in observed matrices
    %param H: innder dimmension of learning factorization
    %param H_0: true inner dimmension i.e. non-negative rank
    %Refs.
    %% lamNMFubd1 -> [Hayashi, 2017a]: Naoki Hayashi, Sumio Watanabe. "Upper Bound of Bayesian Generalization Error in Non-Negative Matrix Factorization", Neurocomputing, Volume 266C, 29 November 2017, pp.21-28. doi: 10.1016/j.neucom.2017.04.068. (2016/12/13 submitted. 2017/8/7 published on web).
    %% lamNMFubd2 -> [Hayashi, 2017b]: Naoki Hayashi, Sumio Watanabe. "Tighter Upper Bound of Real Log Canonical Threshold of Non-negative Matrix Factorization and its Application to Bayesian Inference." 2017 IEEE Symposium Series on Computational Intelligence (IEEE SSCI 2017), Honolulu, Hawaii, USA. Nov. 27 - Dec 1, 2017. (2017/11/28).
    %Note
    %%if H_0=0 or H=H_0=1 or H=H_0=2, then lamNMFubd2 gives the exact value of RLCT of NMF. %
    %%lamNMFubd2 is tighter than lamNMFubd1. %
    lamNMFubd1 =  1/2 * ((H-H_0)*min(M,N) + H_0*(M+N-1));
    if mod(H_0,2)==0
        lamNMFubd2 = 1/2 * ((H-H_0)*min(M,N) + H_0*(M+N-2));
    else
        lamNMFubd2 = 1/2 * ((H-H_0)*min(M,N) + H_0*(M+N-2))+1/2;
    end
end
