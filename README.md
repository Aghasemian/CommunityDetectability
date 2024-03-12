# Community Detectability

<p align="justify">This page is a companion for the <a href="https://journals.aps.org/prx/abstract/10.1103/PhysRevX.6.031005" target="_blank">paper</a> 	
  
> <b>Amir Ghasemian</b>, Pan Zhang, Aaron Clauset, and Cristopher Moore, and Leto Peel
> <br><b><a href="https://journals.aps.org/prx/abstract/10.1103/PhysRevX.6.031005" target="_blank">Detectability thresholds and optimal algorithms for community structure in dynamic networks</a>, Phys. Rev. X 6, 031005 (2016). </b>
on dynamic community detection.</p>


<p align="justify">The detection of communities within a dynamic network is a common means for obtaining a coarse-grained view of a complex system and for investigating its underlying processes. While a number of methods have been proposed in the machine learning and physics literature, we lack a theoretical analysis of their strengths and weaknesses, or of the ultimate limits on when communities can be detected. In this project, we study the fundamental limits of detecting community structure in dynamic networks. Specifically, we analyze the limits of detectability for a dynamic stochastic block model where nodes change their community memberships over time, but where edges are generated independently at each time step. Using the cavity method, we derive a precise detectability threshold as a function of the rate of change and the strength of the communities. Below this sharp threshold, we claim that no efficient algorithm can identify the communities better than chance. We then give two algorithms that are optimal in the sense that they succeed all the way down to this threshold. The first uses belief propagation, which gives asymptotically optimal accuracy, and the second is a fast spectral clustering algorithm, based on linearizing the belief propagation equations. These results (see Fig. below) extend our understanding of the limits of community detection in an important direction, and introduce new mathematical tools for similar extensions to networks with other types of auxiliary information.</p>

<p align="center">
<img src ="./Images/Fig_overlap_epsilon_eta_hres_v1.jpg" width=700><br>
<b>The overlap for (<em>top</em>) belief propagation and (<em>bottom</em>) our spectral algorithm. Please refer to the paper, Fig. 3 for more details.</b>
</p>


### Download the code:
<p align="left">
<a href="./Code/DynamicBeliefPropagation/DBP_AG.py">Dynamic Belief Propagation</a>.</p>
<p align="left">
<a href="./Code/DynamicSpectralClustering/dsbm_temporal_spatial_dog3_finalAG.m">Dynamic Nonbacktracking Spectral Clustering</a>.</p>


### How to cite this work:
<p>If you use this code or data in your research, please cite it as follows:</p>
<pre>
@article{ghasemian2016detectability,
  title = {Detectability thresholds and optimal algorithms for community structure in dynamic networks},
  author = {Ghasemian, Amir and Zhang, Pan and Clauset, Aaron and Moore, Cristopher and Peel, Leto},
  journal = {Physical Review X},
  volume = {6},
  number = {3},
  pages = {031005},
  year = {2016},
  publisher = {APS},
}
</pre>
