%%NaokiHAYASHI
%%NumericTestofNMF
clear
%% Preparation: parameters of experiment
G=100; %number of calc. generalization error (GE) to compute E[GE]
%true matrix parameter
ep = 0.01/3; %means "nearly equal 0" 
A_0=[1;3]; % true matrix
B_0=[5 1]; % true matrix
H=1; % learner factorization dimmension
C_0=A_0*B_0;
% Setting for making data
L=10; %consider compactset [0,L]^d as torus using modulo
n=200; %sample size
data_sig=0.1; %true std dev.
s = 0.1; %hyper parameter of prior
% Setting for MCMC (MH)
met_sig = 0.0023; %std dev. of proposed distribution
K=1000; %sample size for posterior
speriod=20; %period of sampling i.e. thin
burnin=20000; %burn in
tmpK=burnin + speriod*K %total iteration for 1 MCMC cycle
% Setting for calc. of GE
T=20000; %number of "test data" to calculate KL divergence of GE
%% Preparation2: define constants and functions for experiment
[M,N]=size(C_0);
H_0=1;
d=numel(C_0); %the dimmension of the model
model=@(X,A,B)(exp(-sum(sum((X-A*B).^2))./(2*data_sig^2))./sqrt(2*pi*data_sig^2)^d); %learning machine
truedist=@(X)(exp(-sum(sum((X-A_0*B_0).^2))./(2*data_sig^2))./sqrt(2*pi*data_sig^2)^d); %true distribution
%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Numerical Experiment %%
%%%%%%%%%%%%%%%%%%%%%%%%%%
gerrors=zeros(1,G); %gerrors(i) is  i-th GE
plug_errors=zeros(1,G); %i-th mean plugin (EAP) error
accept_prob=zeros(1,G); %i-th acceptance probability = number of accetpted update / total iteration in i-th MCMC
for g=1:1:G
    g
    %% maiking data
    Xn=C_0+data_sig*randn(M,N);
    for i=1:1:n
        Xi=C_0+data_sig*randn(M,N); % Xi ~ truedist N(X| C_0==A_0*B_0, data_sig^2)
        Xn(:,:,i)=Xi;
    end
    %%%%data set Xn just have been defined%%%%
    %%%%constant terms in the Hamiltonian%%%%
    D = sum(2*Xn,3); %sum of Xn for x=1:1:n
    E = sum(Xn.^2,3);
    %%%%%%%%%%%%%%%%%%%
    %% MCMC using MH %%
    %%%%%%%%%%%%%%%%%%%
    %%%% sample i.e. reproduced parameter matrices %%%%%%%%%%%%%
    %%%% {(Ak,Bk)} _k=1 ^K is generated from the posterior%%%%%%
    %%%%%%%%%%%%%%%%%%%
    %%%% Hamiltonian %%%%
    hamiltonian=@(A,B)(sum(sum((E-D.*(A*B)+n.*(A*B).^2)./(2*data_sig^2)))+sum(sum((s/2)*A.^2))+sum(sum((s/2)*B.^2)));
    %%%% acceptance ratio (not acceptance probability) %%%%
    accept_ratio=@(x,y,xx,yy)(min(1,exp(hamiltonian(x,y)-hamiltonian(xx,yy))));
    %%%% initialize %%%%
    %%%%%% naive and square case (depr)
    %A=ep*ones(M,H);%A_0;%horzcat(A_0,0.1*ones(M,H-H_0));
    %B=ep*ones(H,N);%B_0;%horzcat(B_0',0.1*ones(N,H-H_0))';
    %A=[1 ep ep;ep 1 ep;ep ep 1;ep ep ep];
    %B=[1 1 ep ep;1 ep 1 ep;ep 1 ep 1];
    A=horzcat(A_0,ep*ones(M,H-H_0));
    B=horzcat(B_0',ep*ones(N,H-H_0))';
    % A=1;
    % B=1;
    % Xn=Xn(:,:,1)-A*B;
    % for j=1:1:n
    %     Xn(:,:,j)=Xn(:,:,j)-A*B;
    % end
    %%%% to save sampling log %%%%
    hamil_memo = zeros(2,tmpK);
    ratio_memo = zeros(2,tmpK);
    accept_memo = zeros(2,tmpK);
    num_of_accept_prob=0;
    %%%%%% parameter
    allA=A;
    allB=B;
    %%%% candidate sampling before burn-in and thining %%%%
    for i=1:1:tmpK
        deltaA=met_sig*randn(M,H);
        deltaB=met_sig*randn(H,N);
        tmpA=A;
        tmpB=B;
        tmpAA = mod(tmpA + deltaA,L);
        tmpBB = mod(tmpB + deltaB,L);
        hamil_memo(:,i)=[i,hamiltonian(tmpA,tmpB)];
        r=accept_ratio(tmpA,tmpB,tmpAA,tmpBB);
        R = rand(1,1);
        ratio_memo(:,i)=[i,r];
        accept_memo(:,i)=[i,(r>=R).*r+(r<R).*(-0.1)];
        num_of_accept_prob=num_of_accept_prob + (r>=R)*1;
        %% in order to run MH correctly,
        %% the compact set [0,L]^d should be torus: (S^1) x ... x (S^1) (cartesian product of d circles whose radius is L/pi)
        A = mod(A + (r>=R).*deltaA,L);
        B = mod(B + (r>=R).*deltaB,L);
        allA(:,:,i)=A;
        allB(:,:,i)=B;
    end
    %% trace plot %%
    accept_prob(g)=num_of_accept_prob/tmpK;
    %% IF you want to get TRACE PLOT FIGURE using the following code,
    %% you should set G=1 to avoid figure flood
%     figure(1) %hamiltonian plot
%     plot(hamil_memo(1,:),hamil_memo(2,:))
%     figure(2)
%     plot(hamil_memo(1,:),hamil_memo(2,:))
%     xlim([burnin,tmpK]);
%     %ylim([-1,50]);
%     figure(3) %acceptance ratio plot
%     scatter(ratio_memo(1,:),ratio_memo(2,:),1,'blue')
%     xlim([burnin,tmpK]);
%     %ylim([-0.1,0.1])
%     hold on
%     scatter(accept_memo(1,:),accept_memo(2,:),1,'red')
%     xlim([burnin,tmpK]);
%     %ylim([-0.1,0.1]);
%     hold off
    %%%% burn-in and thinning %%%%
    sampleA=ones(M,H);
    sampleB=ones(H,N);
    sampleC=ones(M,N);
%     %preX=C_0;
%     zzz=zeros(1,K); %A(1,1)たち
    for k=1:1:K
        sampleA(:,:,k)=allA(:,:,burnin+speriod*k);
        %A_k=sampleA(:,:,k);
        sampleB(:,:,k)=allB(:,:,burnin+speriod*k);
        %B_k=sampleB(:,:,k);
        %zzz(k)=sampleA(1,1,k);
        sampleC(:,:,k)=sampleA(:,:,k)*sampleB(:,:,k);
    end
%      figure(3)
%      plot(zzz)
    %% For generalization error (GE)
    %%%% Calc. of GE %%%%
    %%%%%% define density arrays %%%%%%
    integralvarX=ones(M,N); %integral variable in KL div. of GE
    predicts=zeros(1,T); %densities of predictive pdf p*(X)
    truedists=zeros(1,T); %densities of true pdf q(X)
    logratios=zeros(1,T); %expected term of GE: log density ratio log(q(X)/p*(X))
    %%%%%%%% Bayesian GE %%%%%%%%
    models=zeros(1,K);
     for t=1:1:T
         integralvarX(:,:,t)=C_0 + data_sig*randn(M,N);
         for k=1:1:K
             models(k)=model(integralvarX(:,:,t),sampleA(:,:,k),sampleB(:,:,k));
         end
         predicts(t)=sum(models)/K;
         truedists(t)=truedist(integralvarX(:,:,t));
         logratios(t)=log(truedists(t)./predicts(t));
     end
    gerrors(g)=sum(logratios)/T;
    %%%%%%%% mean plugin GE using Frob. norm%%%%%%%%
    pred_C=mean(sampleA,3) * mean(sampleB,3);
    plug_error=norm(pred_C-C_0,'fro');
    plug_errors(g)=plug_error;
end
%% conclusion
figure(5);
histogram(gerrors,25)
mean_accept_prob=mean(accept_prob)
expected_gerror=sum(gerrors)/g %mean of GE: E[GE]
sem_gerror = std(gerrors)/sqrt(g) %SEM of GE
expected_plug_error=sum(plug_errors)/g %mean of mean plugin GE: E[||E_post[A]E_post[B]-A_0B_0||]
sem_plug_error=std(plug_errors)/sqrt(g) %sem of mean plugin GE
lambda=n*expected_gerror %the real log canonical threshold (RLCT) (or learning coefficient) calculated numerically
sem_lambda = n*sem_gerror %SEM of lambda
plug_lambda=n*expected_plug_error
sem_plug_lambda = n*sem_plug_error
[bound1,bound2] = lam_NMFubd(M,H,N,H_0) %theoretical upper bound of the RLCT in NMF by [Hayashi, 2017a] and [Hayashi, 2017b]
lamrrr=lam_rrr(M,H,N,rank(C_0)) %theoretical exact value of the RLCT of MF by [Aoyagi, 2005]
lamnmf=lam_rrr(M,H,N,H_0) %RLCT of MF formally use true non-negative rank instead of usual rank
beep;

% Refs.
%% [Aoyagi, 2005]: Miki Aoyagi. Sumio Watanabe. "Stochastic Complexities of Reduced Rank Regression in Bayesian Estimation", Neural Networks, 2005, No. 18, pp.924-933. 
%% [Hayashi, 2017a]: Naoki Hayashi, Sumio Watanabe. "Upper Bound of Bayesian Generalization Error in Non-Negative Matrix Factorization", Neurocomputing, Volume 266C, 29 November 2017, pp.21-28. doi: 10.1016/j.neucom.2017.04.068. (2016/12/13 submitted. 2017/8/7 published on web).
%% [Hayashi, 2017b]: Naoki Hayashi, Sumio Watanabe."Tighter Upper Bound of Real Log Canonical Threshold of Non-negative Matrix Factorization and its Application to Bayesian Inference." 2017 IEEE Symposium Series on Computational Intelligence (IEEE SSCI 2017), Honolulu, Hawaii, USA. Nov. 27 - Dec 1, 2017. (2017/11/28). 