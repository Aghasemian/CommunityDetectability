#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
%% ****************************************************************************************************
%% *** COPYRIGHT NOTICE *******************************************************************************
%% BPD_fbm : Belief Propagation using forward backward equations - matrix version
%% Copyright (C) 2018 Amir Ghasemian
%% This fucntion simulates the beleif propagation algorithm in Physical Review X 6, 031005 (2016) 
%% (Preprint at arxiv:1506.06179). The goal is to estimate the latent labels
%% in a dynamic network with N nodes and T timeslots. 
%% In summary the messages in message passing algorithm are updated in 3 phases:
%% The first phase is the forward direction updates from timeslot 1 to N and in the second phase the messages are updated 
%% in backward direction from N-1 to 1. Finally in last phase we have updates totally in forward, 
%% backward direction with 200 iterations. 
%% ****************************************************************************************************
@author: Amir Ghasemian
"""
from __future__ import division
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
from numpy import linalg as LA
import matplotlib.pyplot as plt
#-----------------

def gen_SBM_dynamic(N,K,T,epsilon_chosen,eta_chosen,c_ave):
    epsilon_vec = epsilon_chosen * np.ones((1,T))
    pa_vec = 1/K*np.ones((K,T))
    pab_vec = np.zeros((K,K,T))
    cin_vec = np.zeros((1,T))
    cout_vec = np.zeros((1,T))
    pin_vec = np.zeros((1,T))
    pout_vec = np.zeros((1,T))
    pab_given = np.zeros((K,K,T))
    for tt in range(T):
        cin_vec[0,tt] = K*c_ave/(1+(K-1)*epsilon_vec[0,tt])
        cout_vec[0,tt] = epsilon_vec[0,tt]*cin_vec[0,tt]
        pin_vec[0,tt] = cin_vec[0,tt]/N
        pout_vec[0,tt] = cout_vec[0,tt]/N
        pab_given[:,:,tt] = pout_vec[0,tt]*np.ones((K,K))+(pin_vec[0,tt] - pout_vec[0,tt])*np.eye(K)
    PTM = np.kron(np.ones((K,1)), (1-eta_chosen)*pa_vec[:,T-1].T)+eta_chosen*np.eye(K)
    As = np.zeros((N,N,T))
    edges = {}
    label_orig = np.zeros((N,T))
    nodes = range(N)
    fid = open("./whole_DynNet_N_"+str(N)+"_K_"+str(K)+"_T_"+str(T)+".txt",'w')
    fid.write("number of nodes:\n")
    fid.write(str(N)+"\n")
    fid.write("number of timeslots:\n")
    fid.write(str(T)+"\n")
    fid.write("edges:")
    
    for tt in range(T):
        fid.write("\nedges at time "+str(tt)+"\n")
        pin = cin_vec[0,tt]/N
        pout = cout_vec[0,tt]/N
        pab = pout*np.ones((K,K))+(pin - pout)*np.eye(K)
        if tt==0:
            type_indices = np.random.choice(K, N, p=pa_vec.T.tolist()[0],replace=True)
            type_indices = np.array([0]*(int(N/K)) + [1]*(int(N/K)))
        else:
            type_indices_aux = type_indices
            for itr1 in range(K):
                type_ = np.where(type_indices_aux==itr1)
                L_type= np.size(type_)
                type_transitions = np.random.choice(K, L_type, p=PTM[itr1,:],replace=True)
                type_indices[type_] = type_transitions
        
        A = np.zeros((N,N))
        for q1 in range(K):
            ind1 = np.where(type_indices==q1)[0]
            for q2 in range(q1,K):
                ind2 = np.where(type_indices==q2)[0]
                A[np.ix_(ind1,ind2)] = np.random.choice(2,[len(ind1),len(ind2)],p=[1-pab[q1,q2],pab[q1,q2]])
                if(q1!=q2):
                    A[np.ix_(ind2,ind1)] = A[ind1][:,ind2].T
        A = np.triu(A)+np.triu(A,1).T
        A = A*((np.eye(N)==0)*1)
        m = np.sum(np.sum(A))/2
        label_orig[:,tt] = type_indices
        A_edge_checker = np.triu(A)
        edges_row,edges_col = np.where(A_edge_checker==1)
        edges[tt] = np.vstack((edges_row,edges_col)).T
        As[:,:,tt] = A
#        fid.write("\n".join(str(row_) for row_ in edges[tt]))
        print(len(edges[tt]))
        for ee_id, ee in enumerate(edges[tt]):
            if ee_id < len(edges[tt])-1:
                fid.write(str(ee[0]) + " " + str(ee[1]) + "\n")
            else:
                fid.write(str(ee[0]) + " " + str(ee[1]))
    fid.write("\ntypes:\n")
    for tt_id,tt in enumerate(label_orig):
        if tt_id < len(label_orig)-1:
            fid.write(str(tt)+"\n")
        else:
            fid.write(str(tt))
    fid.close()    
    return cin_vec, cout_vec, pab_given, As, nodes, edges, label_orig

            

def hcalc(psi_real,pab,t,N,K):
    h = np.sum(np.dot(psi_real[:,:,t],pab[:,:,t]),axis=0)
    return h


    ###########
def node_init_prob(N,K):
    P = np.random.rand(N,K)
    Zi = LA.norm(P,1,axis=1)
    P = P/np.repeat(np.matrix(Zi).T, K, axis=1)
    return P


K = 2 #number of blocks
N =512 #number of nodes
T = 40 #number of timeslots
c_ave = 2 #average degree
Nsim = 1
epsilon_chosen_set = np.arange(11)/float(10)#cout/cin
#epsilon_chosen_set = np.arange(1,2)/float(10)
acuracy_BP_v2_T_fbend = np.zeros(epsilon_chosen_set.shape)
acuracy_label_BP_T_fbend = np.zeros(epsilon_chosen_set.shape)
nmi_BP_T_fbend = np.zeros(epsilon_chosen_set.shape)

for epsilon_id, epsilon_chosen in enumerate(epsilon_chosen_set):
    eta = 0.8
    eta_chosen = eta
    ########
    
    
    ########
    cin_vec, cout_vec, pab_given, As, nodes, edges, label_orig = gen_SBM_dynamic(N,K,T,epsilon_chosen,eta_chosen,c_ave) #label_orig is the original labels
    cin = cin_vec[:,0]
    cout = cout_vec[:,0]
    
    As_vec = np.zeros((N*N,T))
    for tt in range(T):
        As_vec[:,tt] = As[:,:,tt].flatten()
    
    
    ##### BP equations
    
    
    NT = N*T
    neighbors_m_tot = {}
    for tt in range(T):
        source,target = np.where(As[:,:,tt])
        neighbors_m_tot[tt] = {}
        for ii in range(N):
            neighbors_m_tot[tt][ii] = np.where(As[ii,:,tt]==1)[0]
        
    # clusters that are planted into the model
    K = 2
    check_convergence1 = 0
    iter_run = 0 
    ITER_NUM = 0
    fb_number = 0 
    fb_end = 1
    pa = np.zeros((K,T))
    pab = np.zeros((K,K,T))
    cab = np.zeros((K,K,T))
    perturb_value = 0.05
    RT_sec_fb1 = np.zeros((fb_end+1,T))
    RT_iter_fb1 = np.zeros((fb_end+1,T))
    acuracy_BP_v2_T_fb1 = np.zeros((fb_end+1,1))
    acuracy_label_BP_T_fb1 = np.zeros((fb_end+1,1))
    nmi_BP_T_fb1 = np.zeros((fb_end+1,1))
    
    acuracy_BP_v2_T_fb1_check = np.zeros((fb_end+1,1))
    acuracy_label_BP_T_fb1_check = np.zeros((fb_end+1,1))
    nmi_BP_T_fb1_check = np.zeros((fb_end+1,1))
    while (iter_run <= ITER_NUM and check_convergence1==0):
    #    check_convergence1 = 1
        pa_init = 1/K*np.ones((K))
        for tt in range(T):
            pa[:,tt] = pa_init
            pab[:,:,tt] = pab_given[:,:,tt]
        cab[:,:,0] = pab[:,:,0]*N
        
        psi_real_0 = node_init_prob(N,K)
        psi_real = np.zeros((N,K,T))
        psi_m = np.zeros((N,N,K,T))
        psi_t_pt = np.zeros((N,K,T-1))
        psi_t_nt = np.zeros((N,K,T-1))
        h_t = np.zeros((K,T))
        label_inferred = np.zeros((N,T)) #inferred labels
        if fb_number == 0: #forward equations (the first iteration)
            t_set = range(T)
        elif fb_number > 0 and fb_number%2 == 0: #forward
            t_set = range(1,T)
        elif fb_number%2 == 1: #backward
            t_set = range(T-2,-1,-1)
        print("wait ...")
        print("initial forward message passing")
        for tt in t_set:
            if fb_number==0:
                if tt==0:
                    psi_real[:,:,tt] = psi_real_0.copy()
                    indices_row,indices_col = np.where(As[:,:,tt])
                    L_indices = len(indices_row)
                    psi_m_0 = node_init_prob(L_indices,K)
                    Zi = np.sum(psi_m_0,axis=1)
                    psi_m_0 = psi_m_0/np.repeat(np.matrix(Zi),K,axis=1)
                    psi_m[indices_row,indices_col,:,tt] = psi_m_0
                elif tt==(T-1):
                    psi_m_aux = (As[:,:,tt]>0)*(As[:,:,tt-1]>0)
                    
                    psi_m[:,:,:,tt] = psi_m[:,:,:,tt-1]*np.repeat(psi_m_aux[:,:,np.newaxis],K,axis=2)
                    psi_m_aux = (As[:,:,tt]>0)*(As[:,:,tt-1]==0)
                    psi_m[:,:,:,tt] = psi_m[:,:,:,tt-1] + np.repeat(psi_real[:,np.newaxis,:,tt-1],N,axis=1)*np.repeat(psi_m_aux[:,:,np.newaxis],K,axis=2)                    
                    
                    perturb_vec_sum_partial = np.zeros((N,N))
                    perturb_vec = perturb_value*np.random.choice([-1,1],size=(N,N,K-1))*np.repeat(As[:,:,np.newaxis,tt],K-1,axis=2)
                    psi_m[:,:,0:K-1,tt] += perturb_vec
                    perturb_vec_sum_partial = np.sum(perturb_vec,axis=2)
                    psi_m[:,:,K-1,tt] -= perturb_vec_sum_partial
                    zero_indices_row,zero_indices_col,zero_indices_depth = np.where(psi_m[:,:,0:K-1,tt]<0)
                    psi_m[zero_indices_row,zero_indices_col,zero_indices_depth,tt] = 0
                    one_indices_row,one_indices_col,one_indices_depth = np.where(psi_m[:,:,0:K-1,tt]>1)
                    psi_m[one_indices_row,one_indices_col,one_indices_depth,tt] = 1
                    psi_m_sum_partial = np.sum(psi_m[:,:,0:K-1,tt],axis=2)
                    psi_m[:,:,K-1,tt] = np.ones(psi_m_sum_partial.shape) - psi_m_sum_partial                    
                    psi_real[:,:,tt] = psi_real[:,:,tt-1].copy()
                    perturb_vec = perturb_value*np.random.choice([-1,1],size=(N,K-1))
                    perturb_vec = np.concatenate((perturb_vec,-np.sum(perturb_vec,axis=1)[np.newaxis].T),axis=1)
                    psi_real[:,:,tt] += perturb_vec
                    zero_indices_row,zero_indices_col = np.where(psi_real[:,0:K-1,tt]<0)
                    psi_real[zero_indices_row,zero_indices_col,tt] = 0
                    one_indices_row,one_indices_col = np.where(psi_real[:,0:K-1,tt]>1)
                    psi_real[one_indices_row,one_indices_col,tt] = 1
                    psi_real_sum_partial = np.sum(psi_real[:,0:K-1,tt],axis=1)
                    psi_real[:,K-1,tt] = np.ones(psi_real_sum_partial.shape) - psi_real_sum_partial                        
                else:
                    psi_m_aux = (As[:,:,tt]>0)*(As[:,:,tt-1]>0)
                    psi_m[:,:,:,tt] = psi_m[:,:,:,tt-1]*np.repeat(psi_m_aux[:,:,np.newaxis],K,axis=2)
                    psi_m_aux = (As[:,:,tt]>0)*(As[:,:,tt-1]==0)
                    psi_m[:,:,:,tt] = psi_m[:,:,:,tt-1] + np.repeat(psi_real[:,np.newaxis,:,tt-1],N,axis=1)*np.repeat(psi_m_aux[:,:,np.newaxis],K,axis=2)
                    perturb_vec_sum_partial = np.zeros((N,N))
                    perturb_vec = perturb_value*np.random.choice([-1,1],size=(N,N,K-1))*np.repeat(As[:,:,np.newaxis,tt],K-1,axis=2)
                    psi_m[:,:,0:K-1,tt] += perturb_vec
                    perturb_vec_sum_partial = np.sum(perturb_vec,axis=2)
                    psi_m[:,:,K-1,tt] -= perturb_vec_sum_partial
                    
                    zero_indices_row,zero_indices_col,zero_indices_depth = np.where(psi_m[:,:,0:K-1,tt]<0)
                    psi_m[zero_indices_row,zero_indices_col,zero_indices_depth,tt] = 0
                    one_indices_row,one_indices_col,one_indices_depth = np.where(psi_m[:,:,0:K-1,tt]>1)
                    psi_m[one_indices_row,one_indices_col,one_indices_depth,tt] = 1
                    psi_m_sum_partial = np.sum(psi_m[:,:,0:K-1,tt],axis=2)
                    psi_m[:,:,K-1,tt] = np.ones(psi_m_sum_partial.shape) - psi_m_sum_partial
                    psi_real[:,:,tt] = psi_real[:,:,tt-1].copy()
                    perturb_vec = perturb_value*np.random.choice([-1,1],size=(N,K-1))
                    perturb_vec = np.concatenate((perturb_vec,-np.sum(perturb_vec,axis=1)[np.newaxis].T),axis=1)
                    psi_real[:,:,tt] += perturb_vec
                    zero_indices_row,zero_indices_col = np.where(psi_real[:,0:K-1,tt]<0)
                    psi_real[zero_indices_row,zero_indices_col,tt] = 0
                    one_indices_row,one_indices_col = np.where(psi_real[:,0:K-1,tt]>1)
                    psi_real[one_indices_row,one_indices_col,tt] = 1
                    psi_real_sum_partial = np.sum(psi_real[:,0:K-1,tt],axis=1)
                    psi_real[:,K-1,tt] = np.ones(psi_real_sum_partial.shape) - psi_real_sum_partial
            h_t[:,tt] = hcalc(psi_real,pab,tt,N,K)
            
            #define bethe free energy and overlap variables
            time_conv = 10
            #convergence criterion
            crit_infer = 1e-4
            #learning process
            #initialize psi_real for each node i in three time slots envolved
            for t_iter in range(time_conv):
                maxdiffm = -100
                ranseq = range(N)
                for nn in range(N):
                    i = ranseq[nn]
                    neighbors_m = neighbors_m_tot[tt][i]
                    if len(neighbors_m)==0:
                        print(len(neighbors_m))
                    psi_real_temp = psi_real[:,:,tt].copy()
                    if tt==(T-1):
                        psi_real_aux1 = pa[:,tt]*np.exp(-h_t[:,tt])
                        L_neighbors_m = len(neighbors_m)
                        psi_aux = psi_m[neighbors_m,i,0:K,tt].copy()
                        
                        psi_neighs_t_aux = np.dot(psi_aux,pab[:,:,tt])
                        psi_real_aux2 = psi_real_aux1*np.prod(psi_neighs_t_aux, axis=0)
                        psi_real_aux = psi_real_aux2.copy()
                        psi_t_pt_aux = np.zeros((K,1))
                        
                        weight_aux = np.repeat(((1-eta)*pa[:,tt])[np.newaxis].T,K,axis=1)
                        weight_aux += eta*np.eye(K)
                        psi_t_pt_aux = np.dot(psi_t_pt[i,:,tt-1],weight_aux)
                        psi_real_aux = psi_real_aux*psi_t_pt_aux                            
                        Zi_tot = np.sum(psi_real_aux)
                        psi_real_aux3 = psi_real_aux/Zi_tot
                        psi_real[i,:,tt] = psi_real_aux3.T
                        psi_m_new = np.zeros((L_neighbors_m,K))
                        for neighs_ind in range(L_neighbors_m):
                            for kk in range(K):
                                if psi_neighs_t_aux[neighs_ind,kk]<np.spacing(1):
                                    psi_m_new[neighs_ind,kk] = psi_real_aux1[kk]*np.prod(psi_neighs_t_aux[range(neighs_ind)+range(neighs_ind+1,psi_neighs_t_aux.shape[0]),kk],axis=0)
                                    psi_m_new[neighs_ind,kk] = psi_m_new[neighs_ind,kk]*psi_t_pt_aux[kk]
                                else:
                                    psi_m_new[neighs_ind,kk] = psi_real_aux[kk]/psi_neighs_t_aux[neighs_ind,kk]
                        Zij = np.sum(psi_m_new,axis=1)
                        psi_m_new = psi_m_new/np.repeat(np.matrix(Zij).T, K, axis=1)  #flag 
                        
                        psi_new_m = np.zeros((K,N))
                        psi_new_m[:,neighbors_m] = psi_m_new[:,:].T
                        
                        Zi_T_nT = np.sum(psi_real_aux2)
                        psi_T_nT_new = psi_real_aux2/Zi_T_nT
                        
                        mydiff = np.abs(psi_t_nt[i,:,tt-1]-psi_T_nT_new)
                        if len(mydiff) != 0:
                            mymaxdiff = np.max(mydiff)

                        if(mymaxdiff>maxdiffm):
                            maxdiffm = mymaxdiff
                        psi_t_nt[i,:,tt-1] = psi_T_nT_new.T  
                    else:
                        psi_real_aux1 = pa[:,tt]*np.exp(-h_t[:,tt])
                        L_neighbors_m = len(neighbors_m)
                        psi_aux = psi_m[neighbors_m,i,0:K,tt].copy()
                        psi_neighs_t_aux = np.dot(psi_aux,pab[:,:,tt]) #flag
                        psi_real_aux2 = psi_real_aux1*np.prod(psi_neighs_t_aux, axis=0)
                        psi_real_aux = psi_real_aux2.copy()
                        psi_t_pt_aux = np.zeros((K,1))
                        
                        weight_aux = np.repeat(((1-eta)*pa[:,tt])[np.newaxis].T,K,axis=1)
                        weight_aux += eta*np.eye(K)
                        if tt==0:
                            pass
                        else:
                            psi_t_pt_aux = np.dot(psi_t_pt[i,:,tt-1],weight_aux)
                            psi_real_aux = psi_real_aux*psi_t_pt_aux
                            
                        Zi_tot = np.sum(psi_real_aux)
                        psi_real_aux3 = psi_real_aux/Zi_tot
                        psi_real[i,:,tt] = psi_real_aux3.T
                        
                        psi_m_new = np.zeros((L_neighbors_m,K))
                        for neighs_ind in range(L_neighbors_m):
                            for kk in range(K):
                                if psi_neighs_t_aux[neighs_ind,kk]<np.spacing(1):
                                    psi_m_new[neighs_ind,kk] = psi_real_aux1[kk]*np.prod(psi_neighs_t_aux[range(neighs_ind)+range(neighs_ind+1,psi_neighs_t_aux.shape[0]),kk],axis=0)
                                    if tt==0:
                                        print(1)
                                    else:  
                                        psi_m_new[neighs_ind,kk] = psi_m_new[neighs_ind,kk]*psi_t_pt_aux[kk]
                                else:
                                    psi_m_new[neighs_ind,kk] = psi_real_aux[kk]/psi_neighs_t_aux[neighs_ind,kk]
                                    
                        Zij = np.sum(psi_m_new,axis=1)
                        psi_m_new = psi_m_new/np.repeat(np.matrix(Zij).T, K, axis=1)  #flag 
                        psi_new_m = np.zeros((K,N))
                        psi_new_m[:,neighbors_m] = psi_m_new[:,:].T

                        psi_t_pt_new = psi_real_aux3.copy()
                        Zi = np.sum(psi_t_pt_new)
                        psi_t_pt_new = psi_t_pt_new/Zi

                        
                        mydiff = np.abs(psi_t_pt[i,:,tt]-psi_t_pt_new)
                        if len(mydiff) != 0:
                            mymaxdiff = np.max(mydiff)

                        if(mymaxdiff>maxdiffm):
                            maxdiffm = mymaxdiff
                        psi_t_pt[i,:,tt] = psi_t_pt_new.T 
                    
                    
                    mydiff_aux = np.abs(psi_m[i,neighbors_m,:,tt]-psi_new_m[:,neighbors_m].T)
                    if len(mydiff_aux) != 0:
                        mymaxdiff = np.max(mydiff_aux)
                    if (mymaxdiff>maxdiffm):
                        maxdiffm=mymaxdiff
                        print("difference")
                        print(maxdiffm)
                    psi_m[i,neighbors_m,:,tt] = psi_new_m[:,neighbors_m].T
                        
                    for kk in range(K):
                        psi_real_aux = psi_real_temp[i,:]
                        h_t[kk,tt] = h_t[kk,tt]-np.dot(pab[:,kk,tt],psi_real_aux)
                    for kk in range(K):
                        psi_real_aux = psi_real[i,:,tt]
                        h_t[kk,tt] = h_t[kk,tt]+np.dot(pab[:,kk,tt],psi_real_aux)
                    
                if(maxdiffm<crit_infer):
                    bp_last_conv_time = t_iter
                    bp_last_diff = maxdiffm
                    check_convergence1 = 1
                    break
                # inferring the group assignment
            for ii in range(N):
                Ind_max_psi_real = np.argmax(psi_real[ii,:,tt])
                max_psi_real = psi_real[ii,Ind_max_psi_real,tt]
                label_inferred[ii,tt] = Ind_max_psi_real
            if tt==(T-1):
                S = label_inferred.flatten()
                label_orig_tot = label_orig.flatten()
                acuracy_label_BP_T_fb1_check[fb_number,0] = max(np.sum(np.sum(S==label_orig_tot)),np.sum(np.sum(S!=label_orig_tot)))/NT
                acuracy_BP_v2_T_fb1_check[fb_number,0] = (acuracy_label_BP_T_fb1_check[fb_number,0]-1/K)/(1-1/K)
                nmi_BP_T_fb1_check[fb_number,0] = normalized_mutual_info_score(S,label_orig_tot)
            if tt==(T-1):
                label_new = label_inferred
                for tt in range(T):
                    indices_l1_orig = np.where(label_orig[:,tt]==1)[0]
                    L_indices_l1_orig = len(indices_l1_orig)
                    num_l1 = np.sum(label_new[indices_l1_orig,tt]==1)
                    num_l2 = L_indices_l1_orig-num_l1
                    if num_l1>num_l2:
                        pass
                    else:
                        label_new[:,tt] = label_new[:,tt]+1
                        two_indices=np.where(label_new[:,tt]==2)
                        label_new[two_indices,tt]=0
                S = label_inferred.flatten()
                label_orig_tot = label_orig.flatten()
                acuracy_label_BP = max(np.sum(np.sum(S==label_orig_tot)),np.sum(np.sum(S!=label_orig_tot)))/NT
                acuracy_BP_v2 = (acuracy_label_BP-1/K)/(1-1/K)
                nmi_BP = normalized_mutual_info_score(S,label_orig_tot)
            if tt==(T-1):
                acuracy_BP_v2_T_fb1[fb_number,0] = acuracy_BP_v2
                acuracy_label_BP_T_fb1[fb_number,0] = acuracy_label_BP
                nmi_BP_T_fb1[fb_number,0] = nmi_BP
            if iter_run==ITER_NUM:
                check_convergence1 = 1
            if (np.sum(label_inferred[:,tt]==np.ones((1,N)))==N or np.sum(label_inferred[:,tt]==np.zeros((1,N)))==N) and tt==0 and fb_number==0:
                print(tt)
                print(fb_number)
                if iter_run==ITER_NUM:
                    pass
                elif iter_run<ITER_NUM:
                    check_convergence1=0
                    iter_run = iter_run+1
                    break
                    print('broken loop cause of trivial convergence')
    
    #forward and backward loop    
    for fb_number in range(1,fb_end+1):
        if fb_number == 0: #forward equations (the first iteration)
            t_set = range(T)
        elif fb_number > 0 and fb_number%2 == 0: #forward equations
            t_set = range(1,T)
        elif fb_number%2 == 1: #backward equations
            t_set = range(T-2,-1,-1)
        print("wait ...")
        print("initial backward message passing")
        for tt in t_set:
            if fb_number==1:
                if tt==0:
                    perturb_vec_sum_partial = np.zeros((N,N))
                    perturb_vec = perturb_value*np.random.choice([-1,1],size=(N,N,K-1))*np.repeat(As[:,:,np.newaxis,tt],K-1,axis=2)
                    psi_m[:,:,0:K-1,tt] += perturb_vec
                    perturb_vec_sum_partial = np.sum(perturb_vec,axis=2)
                    psi_m[:,:,K-1,tt] -= perturb_vec_sum_partial

                    zero_indices_row,zero_indices_col,zero_indices_depth = np.where(psi_m[:,:,0:K-1,tt]<0)
                    psi_m[zero_indices_row,zero_indices_col,zero_indices_depth,tt] = 0
                    one_indices_row,one_indices_col,one_indices_depth = np.where(psi_m[:,:,0:K-1,tt]>1)
                    psi_m[one_indices_row,one_indices_col,one_indices_depth,tt] = 1
                    psi_m_sum_partial = np.sum(psi_m[:,:,0:K-1,tt],axis=2)
                    psi_m[:,:,K-1,tt] = np.ones(psi_m_sum_partial.shape) - psi_m_sum_partial

                    perturb_vec = perturb_value*np.random.choice([-1,1],size=(N,K-1))
                    perturb_vec = np.concatenate((perturb_vec,-np.sum(perturb_vec,axis=1)[np.newaxis].T),axis=1)
                    psi_real[:,:,tt] += perturb_vec
                    
                    zero_indices_row,zero_indices_col = np.where(psi_real[:,0:K-1,tt]<0)
                    psi_real[zero_indices_row,zero_indices_col,tt] = 0
                    one_indices_row,one_indices_col = np.where(psi_real[:,0:K-1,tt]>1)
                    psi_real[one_indices_row,one_indices_col,tt] = 1
                    psi_real_sum_partial = np.sum(psi_real[:,0:K-1,tt],axis=1)
                    psi_real[:,K-1,tt] = np.ones(psi_real_sum_partial.shape) - psi_real_sum_partial

                else:
                    
                    perturb_vec_sum_partial = np.zeros((N,N))
                    perturb_vec = perturb_value*np.random.choice([-1,1],size=(N,N,K-1))*np.repeat(As[:,:,np.newaxis,tt],K-1,axis=2)
                    psi_m[:,:,0:K-1,tt] += perturb_vec
                    perturb_vec_sum_partial = np.sum(perturb_vec,axis=2)
                    psi_m[:,:,K-1,tt] -= perturb_vec_sum_partial
                    
                    zero_indices_row,zero_indices_col,zero_indices_depth = np.where(psi_m[:,:,0:K-1,tt]<0)
                    psi_m[zero_indices_row,zero_indices_col,zero_indices_depth,tt] = 0
                    one_indices_row,one_indices_col,one_indices_depth = np.where(psi_m[:,:,0:K-1,tt]>1)
                    psi_m[one_indices_row,one_indices_col,one_indices_depth,tt] = 1
                    psi_m_sum_partial = np.sum(psi_m[:,:,0:K-1,tt],axis=2)
                    psi_m[:,:,K-1,tt] = np.ones(psi_m_sum_partial.shape) - psi_m_sum_partial

                    
                    perturb_vec = perturb_value*np.random.choice([-1,1],size=(N,K-1))
                    perturb_vec = np.concatenate((perturb_vec,-np.sum(perturb_vec,axis=1)[np.newaxis].T),axis=1)
                    psi_real[:,:,tt] += perturb_vec
                    
                    zero_indices_row,zero_indices_col = np.where(psi_real[:,0:K-1,tt]<0)
                    psi_real[zero_indices_row,zero_indices_col,tt] = 0
                    one_indices_row,one_indices_col = np.where(psi_real[:,0:K-1,tt]>1)
                    psi_real[one_indices_row,one_indices_col,tt] = 1
                    psi_real_sum_partial = np.sum(psi_real[:,0:K-1,tt],axis=1)
                    psi_real[:,K-1,tt] = np.ones(psi_real_sum_partial.shape) - psi_real_sum_partial
            
            h_t[:,tt] = hcalc(psi_real,pab,tt,N,K)
            
            #define bethe free energy and overlap variables
            time_conv = 10
            #convergence criterion
            crit_infer = 1e-4  #bp_conv_crit or conv_crit
            #learning process
            #initialize psi_real for each node i in three time slots envolved
            for t_iter in range(time_conv):
                maxdiffm = -100
                ranseq = range(N)
                for nn in range(N):
                    i = ranseq[nn]
                    neighbors_m = neighbors_m_tot[tt][i]
                    psi_real_temp = psi_real[:,:,tt].copy()
                    if fb_number%2==0:
                        if tt==(T-1):
                            ##
                            psi_real_aux1 = pa[:,tt]*np.exp(-h_t[:,tt])
                            L_neighbors_m = len(neighbors_m)
                            psi_aux = psi_m[neighbors_m,i,0:K,tt].copy()
                            psi_neighs_t_aux = np.dot(psi_aux,pab[:,:,tt])
                            psi_real_aux2 = psi_real_aux1*np.prod(psi_neighs_t_aux, axis=0)

                            psi_real_aux = psi_real_aux2.copy()
                            psi_t_pt_aux = np.zeros((K,1))
                            
                            weight_aux = np.repeat(((1-eta)*pa[:,tt])[np.newaxis].T,K,axis=1)
                            weight_aux += eta*np.eye(K)
                            psi_t_pt_aux = np.dot(psi_t_pt[i,:,tt-1],weight_aux)
                            psi_real_aux = psi_real_aux*psi_t_pt_aux
                                
                            Zi_tot = np.sum(psi_real_aux)
                            psi_real_aux3 = psi_real_aux/Zi_tot
                            psi_real[i,:,tt] = psi_real_aux3.T
                            psi_m_new = np.zeros((L_neighbors_m,K))
                            for neighs_ind in range(L_neighbors_m):
                                for kk in range(K):
                                    if psi_neighs_t_aux[neighs_ind,kk]<np.spacing(1):
                                        psi_m_new[neighs_ind,kk] = psi_real_aux1[kk]*np.prod(psi_neighs_t_aux[range(neighs_ind)+range(neighs_ind+1,psi_neighs_t_aux.shape[0]),kk],axis=0)
                                        psi_m_new[neighs_ind,kk] = psi_m_new[neighs_ind,kk]*psi_t_pt_aux[kk]
                                    else:
                                        psi_m_new[neighs_ind,kk] = psi_real_aux[kk]/psi_neighs_t_aux[neighs_ind,kk]
                            Zij = np.sum(psi_m_new,axis=1)
                            psi_m_new = psi_m_new/np.repeat(np.matrix(Zij).T, K, axis=1)  #flag 
                            
                            psi_new_m = np.zeros((K,N))
                            psi_new_m[:,neighbors_m] = psi_m_new[:,:].T

                            Zi_T_nT = np.sum(psi_real_aux2)
                            psi_T_nT_new = psi_real_aux2/Zi_T_nT
                            
                            mydiff = np.abs(psi_t_nt[i,:,tt-1]-psi_T_nT_new)
                            if len(mydiff) != 0:
                               mymaxdiff = np.max(mydiff)
                            

                            if(mymaxdiff>maxdiffm):
                                maxdiffm = mymaxdiff           

                            psi_t_nt[i,:,tt-1] = psi_T_nT_new.T 
                            ##
                        else:

                            psi_real_aux1 = pa[:,tt]*np.exp(-h_t[:,tt])
                            L_neighbors_m = len(neighbors_m)

                            psi_aux = psi_m[neighbors_m,i,0:K,tt].copy()
                            psi_neighs_t_aux = np.dot(psi_aux,pab[:,:,tt]) #flag
                            psi_real_aux2 = psi_real_aux1*np.prod(psi_neighs_t_aux, axis=0)

                            psi_real_aux = psi_real_aux2.copy()
                            psi_t_nt_aux = np.zeros((K,1))
                            psi_t_pt_aux = np.zeros((K,1))
                            
                            weight_aux = np.repeat(((1-eta)*pa[:,tt])[np.newaxis].T,K,axis=1)
                            weight_aux += eta*np.eye(K)
                            psi_t_nt_aux = np.dot(psi_t_nt[i,:,tt],weight_aux)
                            psi_real_aux = psi_real_aux*psi_t_nt_aux
                            if tt==0:
                                pass
                            else:
                                psi_t_pt_aux = np.dot(psi_t_pt[i,:,tt-1],weight_aux)
                                psi_real_aux = psi_real_aux*psi_t_pt_aux
                                
                            Zi_tot = np.sum(psi_real_aux)
                            psi_real_aux3 = psi_real_aux/Zi_tot
                            psi_real[i,:,tt] = psi_real_aux3.T
                            psi_m_new = np.zeros((L_neighbors_m,K))
                            for neighs_ind in range(L_neighbors_m):
                                for kk in range(K):
                                    if psi_neighs_t_aux[neighs_ind,kk]<np.spacing(1):
                                        psi_m_new[neighs_ind,kk] = psi_real_aux1[kk]*np.prod(psi_neighs_t_aux[range(neighs_ind)+range(neighs_ind+1,psi_neighs_t_aux.shape[0]),kk],axis=0)
                                        psi_m_new[neighs_ind,kk] = psi_m_new[neighs_ind,kk]*psi_t_nt_aux[kk]
                                        if tt==0:
                                            print(1)
                                        else:  
                                            psi_m_new[neighs_ind,kk] = psi_m_new[neighs_ind,kk]*psi_t_pt_aux[kk]
                                    else:
                                        psi_m_new[neighs_ind,kk] = psi_real_aux[kk]/psi_neighs_t_aux[neighs_ind,kk]
                                        
                            Zij = np.sum(psi_m_new,axis=1)
                            psi_m_new = psi_m_new/np.repeat(np.matrix(Zij).T, K, axis=1)  #flag 
                            psi_new_m = lil_matrix((K,N))
                            for kk in range(K):
                                psi_new_m[kk,neighbors_m] = psi_m_new[:,kk].T
                            psi_t_pt_new = np.zeros((K,1))
                            for kk in range(K):
                                if psi_t_nt_aux[kk]<np.spacing(1):
                                    psi_t_pt_new[kk] = psi_real_aux2[kk]*psi_t_pt_aux[kk]
                                else:
                                    psi_t_pt_new[kk] = psi_real_aux3[kk]/psi_t_nt_aux[kk]
                            
                            Zi = np.sum(psi_t_pt_new)
                            psi_t_pt_new = psi_t_pt_new/Zi
                            mydiff = np.abs(psi_t_pt[i,:,tt]-psi_t_pt_new)
                            if len(mydiff) != 0:
                               mymaxdiff = np.max(mydiff)

                            if(mymaxdiff>maxdiffm):
                                maxdiffm = mymaxdiff           

                            psi_t_pt[i,:,tt] = psi_t_pt_new.T
                    elif fb_number%2==1:
                        if tt==0:
                            psi_real_aux1 = pa[:,tt]*np.exp(-h_t[:,tt])
                            L_neighbors_m = len(neighbors_m)
                            psi_aux = psi_m[neighbors_m,i,0:K,tt].copy()
                            psi_neighs_t_aux = np.dot(psi_aux,pab[:,:,tt])
                            psi_real_aux2 = psi_real_aux1*np.prod(psi_neighs_t_aux, axis=0)

                            psi_real_aux = psi_real_aux2.copy()
                            psi_t_nt_aux = np.zeros((K,1))
                            
                            weight_aux = np.repeat(((1-eta)*pa[:,tt])[np.newaxis].T,K,axis=1)
                            weight_aux += eta*np.eye(K)
                            psi_t_nt_aux = np.dot(psi_t_nt[i,:,tt],weight_aux)
                            psi_real_aux = psi_real_aux*psi_t_nt_aux
                                
                            Zi_tot = np.sum(psi_real_aux)
                            psi_real_aux3 = psi_real_aux/Zi_tot
                            psi_real[i,:,tt] = psi_real_aux3.T
                            psi_m_new = np.zeros((L_neighbors_m,K))
                            for neighs_ind in range(L_neighbors_m):
                                for kk in range(K):
                                    if psi_neighs_t_aux[neighs_ind,kk]<np.spacing(1):
                                        psi_m_new[neighs_ind,kk] = psi_real_aux1[kk]*np.prod(psi_neighs_t_aux[range(neighs_ind)+range(neighs_ind+1,psi_neighs_t_aux.shape[0]),kk],axis=0)
                                        psi_m_new[neighs_ind,kk] = psi_m_new[neighs_ind,kk]*psi_t_nt_aux[kk]
                                    else:

                                        psi_m_new[neighs_ind,kk] = psi_real_aux[kk]/psi_neighs_t_aux[neighs_ind,kk]
                            Zij = np.sum(psi_m_new,axis=1)
                            psi_m_new = psi_m_new/np.repeat(np.matrix(Zij).T, K, axis=1)  #flag 
                            
                            psi_new_m = np.zeros((K,N))
                            psi_new_m[:,neighbors_m] = psi_m_new[:,:].T

                            Zi_T_nT = np.sum(psi_real_aux2)
                            psi_1_p1_new = psi_real_aux2/Zi_T_nT
                            
                            mydiff = np.abs(psi_t_pt[i,:,tt]-psi_1_p1_new)

                            if len(mydiff) != 0:
                               mymaxdiff = np.max(mydiff)
                            if(mymaxdiff>maxdiffm):
                                maxdiffm = mymaxdiff              
                            psi_t_pt[i,:,tt] = psi_1_p1_new.T 
                        else:
                            psi_real_aux1 = pa[:,tt]*np.exp(-h_t[:,tt])
                            L_neighbors_m = len(neighbors_m)
                            psi_aux = psi_m[neighbors_m,i,0:K,tt].copy()
                            psi_neighs_t_aux = np.dot(psi_aux,pab[:,:,tt]) #flag
                            psi_real_aux2 = psi_real_aux1*np.prod(psi_neighs_t_aux, axis=0)
                            psi_real_aux = psi_real_aux2.copy()
                            psi_t_nt_aux = np.zeros((K,1))
                            psi_t_pt_aux = np.zeros((K,1))
                            
                            weight_aux = np.repeat(((1-eta)*pa[:,tt])[np.newaxis].T,K,axis=1)
                            weight_aux += eta*np.eye(K)
                            
                            if tt==(T-1):
                                pass
                            else:
                                psi_t_nt_aux = np.dot(psi_t_nt[i,:,tt],weight_aux)
                                psi_real_aux = psi_real_aux*psi_t_nt_aux
                            psi_t_pt_aux = np.dot(psi_t_pt[i,:,tt-1],weight_aux)
                            psi_real_aux = psi_real_aux*psi_t_pt_aux
                            
                                
                            Zi_tot = np.sum(psi_real_aux)
                            psi_real_aux3 = psi_real_aux/Zi_tot
                            psi_real[i,:,tt] = psi_real_aux3.T
                            psi_m_new = np.zeros((L_neighbors_m,K))
                            for neighs_ind in range(L_neighbors_m):
                                for kk in range(K):
                                    if psi_neighs_t_aux[neighs_ind,kk]<np.spacing(1):
                                        psi_m_new[neighs_ind,kk] = psi_real_aux1[kk]*np.prod(psi_neighs_t_aux[range(neighs_ind)+range(neighs_ind+1,psi_neighs_t_aux.shape[0]),kk],axis=0)
                                        if tt==(T-1):
                                            pass
                                        else:  
                                            psi_m_new[neighs_ind,kk] = psi_m_new[neighs_ind,kk]*psi_t_nt_aux[kk]
                                        psi_m_new[neighs_ind,kk] = psi_m_new[neighs_ind,kk]*psi_t_pt_aux[kk]
                                    else:
                                        psi_m_new[neighs_ind,kk] = psi_real_aux[kk]/psi_neighs_t_aux[neighs_ind,kk]
                                        
                            Zij = np.sum(psi_m_new,axis=1)
                            psi_m_new = psi_m_new/np.repeat(np.matrix(Zij).T, K, axis=1)  #flag 
                            
                            psi_new_m = np.zeros((K,N))
                            psi_new_m[:,neighbors_m] = psi_m_new[:,:].T

                            
                            psi_t_nt_new = np.zeros((K,1))
                            for kk in range(K):
                                if psi_t_pt_aux[kk]<np.spacing(1):
                                    psi_t_nt_new[kk] = psi_real_aux2[kk]*psi_t_nt_aux[kk]
                                else:
                                    psi_t_nt_new[kk] = psi_real_aux3[kk]/psi_t_pt_aux[kk]
                                    
                            Zi = np.sum(psi_t_nt_new)
                            psi_t_nt_new = psi_t_nt_new/Zi
                            
                            mydiff = np.abs(psi_t_nt[i,:,tt-1]-psi_t_nt_new)
                            if len(mydiff) != 0:
                               mymaxdiff = np.max(mydiff)

                            if(mymaxdiff>maxdiffm):
                                maxdiffm = mymaxdiff            

                            psi_t_nt[i,:,tt-1] = psi_t_nt_new.T 
                     
                    mydiff_aux = np.abs(psi_m[i,neighbors_m,:,tt]-psi_new_m[:,neighbors_m].T)
                    if len(mydiff_aux) != 0:
                        mymaxdiff = np.max(mydiff_aux)
                    if (mymaxdiff>maxdiffm):
                        maxdiffm=mymaxdiff
                        print("difference")
                        print(maxdiffm)

                    psi_m[i,neighbors_m,:,tt] = psi_new_m[:,neighbors_m].T

                    for kk in range(K):
                        psi_real_aux = psi_real_temp[i,:]
                        h_t[kk,tt] = h_t[kk,tt]-np.dot(pab[:,kk,tt],psi_real_aux)
                    for kk in range(K):
                        psi_real_aux = psi_real[i,:,tt]
                        h_t[kk,tt] = h_t[kk,tt]+np.dot(pab[:,kk,tt],psi_real_aux) #flag multiplication of matrices in python
                if(maxdiffm<crit_infer):
                    bp_last_conv_time = t_iter
                    bp_last_diff = maxdiffm
                    check_convergence1 = 1
                    break
            for ii in range(N):
                Ind_max_psi_real = np.argmax(psi_real[ii,:,tt])
                max_psi_real = psi_real[ii,Ind_max_psi_real,tt]
                label_inferred[ii,tt] = Ind_max_psi_real
            
        S = label_inferred.flatten()
        label_orig_tot = label_orig.flatten()
        acuracy_label_BP_T_fb1_check[fb_number,0] = max(np.sum(np.sum(S==label_orig_tot)),np.sum(np.sum(S!=label_orig_tot)))/NT
        acuracy_BP_v2_T_fb1_check[fb_number,0] = (acuracy_label_BP_T_fb1_check[fb_number,0]-1/K)/(1-1/K)
        nmi_BP_T_fb1_check[fb_number,0] = normalized_mutual_info_score(S,label_orig_tot)
        
        label_new = label_inferred
        for tt in range(T):
            indices_l1_orig = np.where(label_orig[:,tt]==1)[0]
            L_indices_l1_orig = len(indices_l1_orig)
            num_l1 = np.sum(label_new[indices_l1_orig,tt]==1)
            num_l2 = L_indices_l1_orig-num_l1
            if num_l1>num_l2:
                pass
            else:
                label_new[:,tt] = label_new[:,tt]+1
                two_indices=np.where(label_new[:,tt]==2)
                label_new[two_indices,tt]=0
        S = label_inferred.flatten()
        label_orig_tot = label_orig.flatten()
        acuracy_label_BP = max(np.sum(np.sum(S==label_orig_tot)),np.sum(np.sum(S!=label_orig_tot)))/NT
        acuracy_BP_v2 = (acuracy_label_BP-1/K)/(1-1/K)
        nmi_BP = normalized_mutual_info_score(S,label_orig_tot)
        
        acuracy_BP_v2_T_fb1[fb_number,0] = acuracy_BP_v2
        acuracy_label_BP_T_fb1[fb_number,0] = acuracy_label_BP
        nmi_BP_T_fb1[fb_number,0] = nmi_BP
        
    
    
    # final loop
    # convergence criterion
    time_conv2 = 200
    crit_infer = 1e-6 #bp_conv_crit or conv_crit
    
    ff = fb_number+1
    for t_iter in range(time_conv2+1):
        print("wait till 200 iteration or till convergence")
        print("#"+str(t_iter))
        maxdiffm  = -100
        if ff%2 == 0:
            ranseq = range(NT)
        else:
            ranseq = range(NT-1,-1,-1)
        for nn in range(NT):
            iii = ranseq[nn]
            tt=int(np.floor(iii/N))
            i = iii%N
            neighbors_m = neighbors_m_tot[tt][i]
            psi_real_temp = psi_real[:,:,tt].copy()
            
            psi_real_aux1 = pa[:,tt]*np.exp(-h_t[:,tt])
            L_neighbors_m = len(neighbors_m)
            psi_aux = psi_m[neighbors_m,i,0:K,tt].copy()
            psi_neighs_t_aux = np.dot(psi_aux,pab[:,:,tt]) #flag
            psi_real_aux2 = psi_real_aux1*np.prod(psi_neighs_t_aux, axis=0)
            psi_real_aux = psi_real_aux2.copy()
            psi_t_nt_aux = np.zeros((K,1))
            psi_t_pt_aux = np.zeros((K,1))
            
            weight_aux = np.repeat(((1-eta)*pa[:,tt])[np.newaxis].T,K,axis=1)
            weight_aux += eta*np.eye(K)
            
            if tt==(T-1):
                pass
            else:
                psi_t_nt_aux = np.dot(psi_t_nt[i,:,tt],weight_aux)
                psi_real_aux = psi_real_aux*psi_t_nt_aux
            if tt==0:
                pass
            else:
                psi_t_pt_aux = np.dot(psi_t_pt[i,:,tt-1],weight_aux)
                psi_real_aux = psi_real_aux*psi_t_pt_aux
                
            Zi_tot = np.sum(psi_real_aux)
            psi_real_aux3 = psi_real_aux/Zi_tot
            psi_real[i,:,tt] = psi_real_aux3.T
            
            psi_m_new = np.zeros((L_neighbors_m,K))
            for neighs_ind in range(L_neighbors_m):
                for kk in range(K):
                    if psi_neighs_t_aux[neighs_ind,kk]<np.spacing(1):
                        psi_m_new[neighs_ind,kk] = psi_real_aux1[kk]*np.prod(psi_neighs_t_aux[range(neighs_ind)+range(neighs_ind+1,psi_neighs_t_aux.shape[0]),kk],axis=0)
                        if tt==(T-1):
                            pass
                        else:  
                            psi_m_new[neighs_ind,kk] = psi_m_new[neighs_ind,kk]*psi_t_nt_aux[kk]
                        if tt==0:
                            pass
                        else:
                            psi_m_new[neighs_ind,kk] = psi_m_new[neighs_ind,kk]*psi_t_pt_aux[kk]
                    else:
                        psi_m_new[neighs_ind,kk] = psi_real_aux[kk]/psi_neighs_t_aux[neighs_ind,kk]
                        
            Zij = np.sum(psi_m_new,axis=1)
            psi_m_new = psi_m_new/np.repeat(np.matrix(Zij).T, K, axis=1)  #flag 
            
            psi_new_m = np.zeros((K,N))
            psi_new_m[:,neighbors_m] = psi_m_new[:,:].T
            
            psi_t_nt_new = np.zeros((K,1))
            psi_t_pt_new = np.zeros((K,1))
            if tt==0:
                psi_t_pt_new = psi_real_aux2
            else:
                for kk in range(K):
                    if psi_t_nt_aux[kk]<np.spacing(1):
                        psi_t_pt_new[kk] = psi_real_aux2[kk]*psi_t_pt_aux[kk]
                    else:
                        psi_t_pt_new[kk] = psi_real_aux3[kk]/psi_t_nt_aux[kk]
            if tt==(T-1):
                psi_t_nt_new = psi_real_aux2
            else:
                for kk in range(K):
                    if psi_t_pt_aux[kk]<np.spacing(1):
                        psi_t_nt_new[kk] = psi_real_aux2[kk]*psi_t_nt_aux[kk]
                    else:
                        psi_t_nt_new[kk] = psi_real_aux3[kk]/psi_t_pt_aux[kk]
            
            Zi_t_nt = np.sum(psi_t_nt_new)
            psi_t_nt_new = psi_t_nt_new/Zi_t_nt
            Zi_t_pt = np.sum(psi_t_pt_new)
            psi_t_pt_new = psi_t_pt_new/Zi_t_pt
            if tt==(T-1):
                pass
            else:
                mydiff = np.abs(psi_t_pt[i,:,tt]-psi_t_pt_new)
                if len(mydiff) != 0:
                    mymaxdiff = np.max(mydiff)
                if(mymaxdiff>maxdiffm):
                    maxdiffm = mymaxdiff
#                    print("maxdiffm1 = ", maxdiffm)
                    
                psi_t_pt[i,:,tt] = psi_t_pt_new.T 
                
                
            if tt==0:
                pass
            else:
                mydiff = np.abs(psi_t_nt[i,:,tt-1]-psi_t_nt_new)
                if len(mydiff) != 0:
                    mymaxdiff = np.max(mydiff)
                if(mymaxdiff>maxdiffm):
                     maxdiffm = mymaxdiff
#                     print("maxdiffm2 = ", maxdiffm)
                psi_t_nt[i,:,tt-1] = psi_t_nt_new.T 
            mydiff_aux = np.abs(psi_m[i,neighbors_m,:,tt]-psi_new_m[:,neighbors_m].T)
            if len(mydiff_aux) != 0:
                mymaxdiff = np.max(mydiff_aux)
            if (mymaxdiff>maxdiffm):
                maxdiffm=mymaxdiff
                print("difference")
                print(maxdiffm)    
            psi_m[i,neighbors_m,:,tt] = psi_new_m[:,neighbors_m].T
            for kk in range(K):
                psi_real_aux = psi_real_temp[i,:]
                h_t[kk,tt] = h_t[kk,tt]-np.dot(pab[:,kk,tt],psi_real_aux)
            for kk in range(K):
                psi_real_aux = psi_real[i,:,tt]
                h_t[kk,tt] = h_t[kk,tt]+np.dot(pab[:,kk,tt],psi_real_aux)
        
        if(maxdiffm<crit_infer):
            bp_last_conv_time = t_iter
            bp_last_diff = maxdiffm
            check_convergence1 = 1
            break
    
    for tt in range(T):    
        for ii in range(N):
            Ind_max_psi_real = np.argmax(psi_real[ii,:,tt])
            max_psi_real = psi_real[ii,Ind_max_psi_real,tt]
            label_inferred[ii,tt] = Ind_max_psi_real
            
    S = label_inferred.flatten()
    label_orig_tot = label_orig.flatten()
    acuracy_label_BP_T_check = max(np.sum(np.sum(S==label_orig_tot)),np.sum(np.sum(S!=label_orig_tot)))/NT
    acuracy_BP_v2_T_check = (acuracy_label_BP_T_check-1/K)/(1-1/K)
    nmi_BP_check = normalized_mutual_info_score(S,label_orig_tot)
    
    nmi_BP_t = np.zeros((T,1))
    acuracy_label_BP_t = np.zeros((T,1))
    acuracy_BP_v2_t = np.zeros((T,1))
    for tt in range(T):
        acuracy_label_BP_t[tt] = max(np.sum(np.sum(label_inferred[:,tt]==label_orig[:,tt])),np.sum(np.sum(label_inferred[:,tt]!=label_orig[:,tt])))/N
        acuracy_BP_v2_t[tt] = (acuracy_label_BP_t[tt]-1/K)/(1-1/K)
        nmi_BP_t[tt] = normalized_mutual_info_score(label_inferred[:,tt],label_orig[:,tt])
    nmi_BP_ave = np.mean(nmi_BP_t)
    acuracy_label_BP_ave=np.mean(acuracy_label_BP_t)
    acuracy_BP_v2_ave=np.mean(acuracy_BP_v2_t)
    
    #label_new = label_inferred
    for tt in range(T):
        indices_l1_orig = np.where(label_orig[:,tt]==1)[0]
        L_indices_l1_orig = len(indices_l1_orig)
        num_l1 = np.sum(label_inferred[indices_l1_orig,tt]==1)
        num_l2 = L_indices_l1_orig-num_l1
        if num_l1>num_l2:
            pass
        else:
            label_inferred[:,tt] = label_inferred[:,tt]+1
            two_indices=np.where(label_inferred[:,tt]==2)
            label_inferred[two_indices,tt]=0
    S = label_inferred.flatten()
    label_orig_tot = label_orig.flatten()
    acuracy_label_BP = max(np.sum(np.sum(S==label_orig_tot)),np.sum(np.sum(S!=label_orig_tot)))/NT
    acuracy_BP_v2 = (acuracy_label_BP-1/K)/(1-1/K)
    nmi_BP = normalized_mutual_info_score(S,label_orig_tot)
    
    acuracy_BP_v2_T_fbend[epsilon_id] = acuracy_BP_v2
    acuracy_label_BP_T_fbend[epsilon_id] = acuracy_label_BP
    nmi_BP_T_fbend[epsilon_id] = nmi_BP
    
    
    print('finished')
    for tt in range(T):
        if np.sum(label_inferred[:,tt]==np.ones((N,1)))==N or np.sum(label_inferred[:,tt]==np.zeros((N,1)))==N:
            print('alert of trivial fixed point')
    print('accuracy of the labels inferred : '+ str(acuracy_BP_v2_T_fbend[epsilon_id]))
    print('overlap : '+ str(acuracy_BP_v2_T_fbend[epsilon_id]))
    print('NMI (normalized mutual information) : '+ str(nmi_BP_T_fbend[epsilon_id]))

plt.figure(figsize=(1 * 8, 6))
plt.plot(epsilon_chosen_set[1:],acuracy_BP_v2_T_fbend[1:],'r--',marker='s')
plt.plot([0.5,0.5],[-0.2,1.2],c='r')
plt.title('overlap versus $\epsilon$ for ' + r'$\bar{c}=2$')
plt.xlabel('community structure strength $\epsilon$')
plt.ylabel('overlap')
plt.savefig('overlap_vs_eps_PRX_N512_T40_etap8_c2.pdf')
plt.show()