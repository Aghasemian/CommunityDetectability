function [accuracy,ovl,nmi_spectral,eigs_1st_2nd,stab]=dsbm_temporal_spatial_dog3_finalAG(q,n,T,c,numvec,eta,epsilon,opts)
[As,conf_true_time,cin,cout,lambda,stab]=dsbm_gen_network_new(q,n,T,c,eta,epsilon);
%% ****************************************************************************************************
%% *** COPYRIGHT NOTICE *******************************************************************************
%% This is the main fucntion that simulates the dynamic nonbacktracking spectral algorithm in Physical Review X 6, 031005 (2016) 
%% (Preprint at arxiv:1506.06179). The goal is to estimate the latent labels
%% in a dynamic network with N nodes and T timeslots.
%% Authors: Amir Ghasemian, Pan Zhang, Aaron Clauset, Cristopher Moore, Leto Peel
%% Copyright (C) 2014 Amir Ghasemian, Pan Zhang
%% ****************************************************************************************************
%%@author: Amir Ghasemian, Pan Zhang 

%% regularized 4nT x 4nT form
fournT=4*n*T;
B=sparse([],[],[],fournT,fournT,0);
I=eye(n);
kvecs = sum(As(:,:,:),1);
omega=-1.0/q/(n-c)*(cin-cout);
for t=1:T %v^{in}
    a1 = (t-1)*n+1;
    a2 = t*n;
    %v^{in}
    b1 = (t-1)*n+1;
    b2 = t*n;
    B(a1:a2,b1:b2)= As(:,:,t)*lambda;
    %v^{out}
    b1 = n*T + (t-1)*n+1;
    b2 = n*T+t*n;
    B(a1:a2,b1:b2)=-1*lambda*I;
    %u^{in}
    b1 = 2*n*T+(t-1)*n+1;
    b2 = 2*n*T+t*n;
    B(a1:a2,b1:b2)= As(:,:,t)*lambda;
end
for t=1:T %v^{out}
    a1 = n*T+(t-1)*n+1;
    a2 = n*T+t*n;
    kvec = sum(As(:,:,t),1);
    dit = sparse(1:n,1:n,kvec,n,n,n);
    %v^{in}
    b1 = (t-1)*n+1;
    b2 = t*n;
    w=dit;
    w(dit>0) = dit(dit>0)- I(dit>0);
    B(a1:a2,b1:b2)= lambda*w;
    %u^{in}
    b1 = 2*n*T+(t-1)*n+1;
    b2 = 2*n*T+t*n;
    B(a1:a2,b1:b2)= lambda*dit;
end
for t=1:T %u^{in}
    a1 = 2*n*T+(t-1)*n+1;
    a2 = 2*n*T+t*n;
    %v^{in}
    for s=1:T
        if(s==t-1 || s==t+1)
            b1 = (s-1)*n+1;
            b2 = s*n;
            B(a1:a2,b1:b2)=eta*I;
        end
    end
    %u^{in}
    for s=1:T
        if(s==t-1 || s==t+1)
            b1 = 2*n*T+(s-1)*n+1;
            b2 = 2*n*T+s*n;
            B(a1:a2,b1:b2)=eta*I;
        end
    end
    %u^{out}
    b1 = 3*n*T+(t-1)*n+1;
    b2 = 3*n*T+t*n;
    B(a1:a2,b1:b2)= -1*eta*I;
end
for t=1:T %u^{out}
    a1 = 3*n*T+(t-1)*n+1;
    a2 = 3*n*T+t*n;
    if(T==1) % for T=1, u^{out} is zero
        break;
    end
    %v^{in}
    b1 = (t-1)*n+1;
    b2=t*n;
    if(t==1 || t== T)
        B(a1:a2,b1:b2)=eta*I;
    else
        B(a1:a2,b1:b2)=2*eta*I;
    end
    %u^{in}
    b1 = 2*n*T+(t-1)*n+1;
    b2 = 2*n*T+t*n;
    if(t~=1 && t~= T)
        B(a1:a2,b1:b2)=eta*I;
    end
end    
    

Bm=sparse([],[],[],n*T,fournT,0); % matrix to compute marginals
for t=1:T
    a1 = (t-1)*n+1;
    a2 = t*n;
    %v^{in}
    b1 = (t-1)*n+1;
    b2=t*n;
    Bm(a1:a2,b1:b2)= I;
    %u^{in}
    b1 = 2*n*T+(t-1)*n+1;
    b2=2*n*T+t*n;
    Bm(a1:a2,b1:b2)= I;
end


%% regularization part
xa=sparse([],[],[],fournT,T,0);
ya=sparse([],[],[],T,fournT,0);
xb=sparse([],[],[],fournT,T,0);
yb=sparse([],[],[],T,fournT,0);
xm=sparse([],[],[],n*T,T,0);
ym=sparse([],[],[],T,fournT,0);
for t=1:T
    kvec = sum(As(:,:,t),1);
    xa((t-1)*n+1:t*n,t)=kvec'*omega*lambda;
    xa(T*n+(t-1)*n+1:T*n+t*n,t)=kvec'*omega*lambda;
    if(t==1 || t== T)
        xa(3*T*n+(t-1)*n+1:3*T*n+t*n,t)=omega*eta;
    else
        xa(3*T*n+(t-1)*n+1:3*T*n+t*n,t)=2*omega*eta;
    end
    ya(t,(t-1)*n+1:t*n) = ones(1,n); 
    
    xb(2*T*n+(t-1)*n+1:2*T*n+t*n,t)=omega*eta;
    for s=1:T
        if(s==t-1 || s==t+1)
            yb(t,(s-1)*n+1:s*n) = ones(1,n); 
        end
    end
    
    xm((t-1)*n+1:t*n,t)=ones(n,1)*omega;
    ym(t,(t-1)*n+1:t*n) = ones(1,n); 
end

B=B-omega*n*c*speye(size(B));

%% spectrum part
[V, D,flag] = eigs(@(x)dnRk_2(x,B,T,xa,ya,xb,yb),fournT,2,'LM',opts);
eigs_1st_2nd=diag(D).';
V=V(:,1);

Vts=Bm*V;
for t=1:T
    Vts = Vts + xm(:,t)*(ym(t,:)*V);
end

accuracy=zeros(1,T);
nmi_spectral=zeros(1,T);

for t=1:T
    Vs2=Vts((t-1)*n+1:t*n,:);
    conf_infer1=Vs2;
    conf_infer2=Vs2;
    conf_infer1(Vs2<0)=1;
    conf_infer1(Vs2>0)=2;
    conf_infer1(Vs2==0) = randi([1 2]);
    conf_infer2(Vs2<0)=2;
    conf_infer2(Vs2>0)=1;
    conf_infer2(Vs2==0) = randi([1 2]);
    maxaccuracy=0;
    for num=1:numvec
        accuracy1=mean(conf_true_time(:,t)==conf_infer1(:,num));
        accuracy2=mean(conf_true_time(:,t)==conf_infer2(:,num));
        maxaccuracy=max([maxaccuracy,accuracy1,accuracy2]);
    end
    accuracy(1,t)=maxaccuracy;
    nmi_spectral(1,t)=nmi(conf_infer1,conf_true_time(:,t),q);

end
ovl=(accuracy-1/q)/(1-1/q);
    
