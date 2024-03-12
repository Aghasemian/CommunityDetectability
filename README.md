# Community Detectability

<p align="justify">This page is a companion for the <a href="https://journals.aps.org/prx/abstract/10.1103/PhysRevX.6.031005" target="_blank">paper</a> 	
  
> <b>Amir Ghasemian</b>, Pan Zhang, Aaron Clauset, and Cristopher Moore, and Leto Peel
> <br><b>"Detectability thresholds and optimal algorithms for community structure in dynamic networks", Phys. Rev. X 6, 031005 [<a href="https://journals.aps.org/prx/abstract/10.1103/PhysRevX.6.031005" target="_blank"></a>] </b>

on dynamic community detection.</p>


<p align="justify">The detection of communities within a dynamic network is a common means for obtaining a coarse-grained view of a complex system and for investigating its underlying processes. While a number of methods have been proposed in the machine learning and physics literature, we lack a theoretical analysis of their strengths and weaknesses, or of the ultimate limits on when communities can be detected. In this project, we study the fundamental limits of detecting community structure in dynamic networks. Specifically, we analyze the limits of detectability for a dynamic stochastic block model where nodes change their community memberships over time, but where edges are generated independently at each time step. Using the cavity method, we derive a precise detectability threshold as a function of the rate of change and the strength of the communities. Below this sharp threshold, we claim that no efficient algorithm can identify the communities better than chance. We then give two algorithms that are optimal in the sense that they succeed all the way down to this threshold. The first uses belief propagation, which gives asymptotically optimal accuracy, and the second is a fast spectral clustering algorithm, based on linearizing the belief propagation equations. These results (see Fig. below) extend our understanding of the limits of community detection in an important direction, and introduce new mathematical tools for similar extensions to networks with other types of auxiliary information.</p>

<p>
The detectability limit:
\begin{equation}
\label{eq:threshold}
c \lambda^2 = \frac{1-\eta^2}{1+\eta^2} 
\quad \text{or} \quad 
|c_{\textrm{in}} - c_{\textrm{out}}| = k \sqrt{c \,\frac{1-\eta^2}{1+\eta^2}} \, , 
\end{equation}
</p>

<div class="row">
    <div class="col-12 mt-3 mt-md-0">
        {% include figure.html path="Images/Fig_overlap_epsilon_eta_hres_v1.jpg" title="example image" class="img-fluid rounded z-depth-1 custom-image-size" %}
        <figcaption class="figure-caption justified-caption">
            The overlap for (<em>top</em>) belief propagation and (<em>bottom</em>) our spectral algorithm. The detectability transition in Eq. \eqref{eq:threshold} for $T=\infty$ is shown as a solid line. The dashed curve shows the detectability transition for $T=40$; the magenta curve shows the transition for $T=\infty$. Each point shows the average over 100 dynamic networks generated by our model with $n=512$, $T=40$, $k=2$ groups, and average degree $c=16$. The overlap here is calculated by averaging the maximum overlap at each time slot over all permutations. This maximization step implies that the expected overlap in the undetectable region is $O(n^{-1/2})$, and this produces a small deviation away from overlap = 0 in our numerical experiments.
        </figcaption>
    </div>
</div>

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
