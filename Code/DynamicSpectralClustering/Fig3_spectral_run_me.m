
clear all
close all
rng shuffle

addpath(pwd)
q=2;
n=5000;
c=6;

eta_set=0.7;
epsilon_set=0.01:0.01:1;
size(epsilon_set)
size(eta_set)
T=4;
n_eta=length(eta_set);
n_epsilon=length(epsilon_set);
time_sol=zeros(n_epsilon,1);

%%
numvec=1;
rerun=0;
N_sim=1;

data=[];
opts_tol_opt=eps;
opts_tol_epsilon=opts_tol_opt*ones(n_epsilon,1);
for n_sim=1:N_sim
    for eta_ind=1
        eta=eta_set(eta_ind);
        for epsilon_ind=1:100
            display(n_sim)
            display(epsilon_ind)
            tic
            epsilon=epsilon_set(epsilon_ind);
            tr=1;
            opts.tol=opts_tol_epsilon(epsilon_ind);
            while tr==1
                try
                    opts.tol=opts_tol_epsilon(epsilon_ind);
                    [accuracy,ovl,nmi_spectral,eigs_1st_2nd,stab]=dsbm_temporal_spatial_dog3_finalAG(q,n,T,c,numvec,eta,epsilon,opts);
                    time_sol(epsilon_ind,1)=toc;
                    data=[data;[n T n_sim eta epsilon stab mean(accuracy) mean(ovl) mean(nmi_spectral) eigs_1st_2nd opts.tol time_sol(epsilon_ind,1)]];                    
                    tr=0;
                catch
                    display('error')
                    opts_tol_epsilon(epsilon_ind)=opts_tol_epsilon(epsilon_ind)*10000;
                end
            end
            save('PT_spectral_N5000T4_sim3_1_1_new1.mat','data');
        end
    end
end
save('PT_spectral_N5000T4_sim3_1_1_new1.mat','data');

accuracy_vec = [];
overlap_vec = [];
for epsilon_val = epsilon_set
    inds = find(data(:,5)==epsilon_val);
    accuracy_vec = [accuracy_vec; mean(data(inds,7))];
    overlap_vec = [overlap_vec; mean(data(inds,8))];
end

alpha = sqrt(c*((1-epsilon_set)./(1+epsilon_set)).^2);
alpha_c = sqrt(2)*(2+eta^2+eta^6+eta*sqrt(eta^8+2*eta^4+8*eta^2+5))^(-0.5);
alpha_over_alpha_c = alpha/alpha_c;
plot(alpha_over_alpha_c, overlap_vec,'-o','LineWidth',2,'color','#EDB120')
xlabel('\alpha/\alpha_c(T,\eta)')
ylabel('overlap')
ylim([0.0-0.01,1.0+0.04])

dlmwrite('overlap.txt',overlap_vec);
dlmwrite('accuracy.txt',accuracy_vec);
dlmwrite('alpha_over_alpha_c.txt',alpha_over_alpha_c');
