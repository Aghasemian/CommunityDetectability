%% This function will compute the normalized mutual information between two set of labels (inferred and original)
function nmi_score=nmi(label,label_orig,Q)
N=length(label);
label_aux=zeros(Q,N);
label_orig_aux=zeros(Q,N);

for i=1:N
    label_aux(label(i),i)=1;
    label_orig_aux(label_orig(i),i)=1;
end

CT=label_aux*label_orig_aux'; %Contingency Table

N_L=sum(label_aux,2); %Assigned Label
N_LO=sum(label_orig_aux,2); %Original Label
N_tot=N_L*N_LO';
H_L=-sum(N_L(N_L>0)/N.*log(N_L(N_L>0)/N)); %entropy of assigned labels
H_LO=-sum(N_LO(N_LO>0)/N.*log(N_LO(N_LO>0)/N)); %entropy of original labels

I_L_LO=sum(CT(CT>0)/N.*log(N*CT(CT>0)./N_tot(CT>0)));

nmi_score=2*I_L_LO/(H_L+H_LO);