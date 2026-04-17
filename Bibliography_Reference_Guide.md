# Bibliography Reference Guide — Thesis Papers

> **Purpose:** This document provides the full bibliographical reference and a summary paragraph for each paper in the thesis collection. Use it to identify where to cite each source in your undergraduate thesis and for bibliography compliance.

---

## Table of Contents — Thesis Bibliography [1]–[23]

1. [Collective Motion & Swarming — Foundational Models](#1-collective-motion--swarming--foundational-models) — [17], [8]
2. [Data-Driven Analysis of Collective Motion](#2-data-driven-analysis-of-collective-motion) — [2]
3. [Data-Driven Equation Discovery & Balance Models](#3-data-driven-equation-discovery--balance-models) — [12], [13], [6], [19]
4. [Model Order Reduction for Transport-Dominated Systems](#4-model-order-reduction-for-transport-dominated-systems) — [15], [10], [3], [14], [20]
5. [Symmetry Reduction Methods](#5-symmetry-reduction-methods) — [5], [4], [16]
6. [Reduced Order Modeling & Multiscale Methods](#6-reduced-order-modeling--multiscale-methods) — [1], [21]
7. [Numerical Methods for Stiff PDEs](#7-numerical-methods-for-stiff-pdes) — [7], [9]
8. [Statistical & Optimization Methods](#8-statistical--optimization-methods) — [11], [18], [22], [23]
9. [Additional Reference Papers (in folder, not in thesis bibliography)](#9-additional-reference-papers)

---

## 1. Collective Motion & Swarming — Foundational Models

### [17] Vicsek, T., Czirók, A., Ben-Jacob, E., Cohen, I., & Shochet, O. (1995)
**"Novel Type of Phase Transition in a System of Self-Driven Particles."**
*Physical Review Letters*, 75(6), 1226–1229.
DOI: [10.1103/PhysRevLett.75.1226](https://doi.org/10.1103/PhysRevLett.75.1226)

This is the foundational paper that introduced the Vicsek model, a minimal agent-based model for collective motion. In the model, particles move at constant speed and at each time step adopt the average direction of their neighbors plus a random perturbation. The paper presents numerical evidence of a kinetic phase transition from disordered motion (zero average velocity) to ordered collective transport via spontaneous symmetry breaking of rotational symmetry. The transition is continuous, with the average velocity scaling as $(η_c - η)^β$ with $β ≈ 0.45$. This work is the canonical reference for self-propelled particle models and established the framework upon which much of the collective motion literature is built. **Cite when:** introducing the Vicsek model, discussing phase transitions in active matter, or referencing the foundational framework for self-propelled particle systems.

---

### [8] D'Orsogna, M. R., Chuang, Y. L., Bertozzi, A. L., & Chayes, L. S. (2006)
**"Self-Propelled Particles with Soft-Core Interactions: Patterns, Stability, and Collapse."**
*Physical Review Letters*, 96(10), 104302.
DOI: [10.1103/PhysRevLett.96.104302](https://doi.org/10.1103/PhysRevLett.96.104302)

This paper models self-propelled biological or artificial agents interacting through pairwise attractive–repulsive forces described by a generalized Morse potential. For the first time, the authors predict the stability and morphology of collective organization starting from the shape of the two-body interaction potential, using fundamental principles from statistical mechanics. They present a coherent theory for all possible phases of collective motion (e.g., crystal-like swarms, localized vortices, collapse). This work provides a systematic framework for predicting whether a swarm will collapse or expand as particle number increases. **Cite when:** introducing the D'Orsogna swarming model, discussing attractive–repulsive interaction potentials, or analyzing pattern stability in self-propelled particle systems.

---

## 2. Data-Driven Analysis of Collective Motion

### [2] Bhaskar, D., Manhart, A., Milzman, J., Nardini, J. T., Storey, K. M., Topaz, C. M., & Ziegelmeier, L. (2019)
**"Analyzing Collective Motion with Machine Learning and Topology."**
*Chaos*, 29(12), 123125.
DOI: [10.1063/1.5125493](https://doi.org/10.1063/1.5125493)

This paper applies topological data analysis (TDA) and machine learning to study the D'Orsogna et al. model of collective motion. The authors compare two types of input features for classifying emergent behaviors (flocking, milling, etc.) and recovering model parameters: (1) traditional time series of order parameters and (2) topological summaries based on time-varying persistent homology across multiple scales. For both unsupervised and supervised machine learning tasks, the topological approach outperforms the one based on traditional order parameters, and does so without requiring prior knowledge of the expected patterns. **Cite when:** discussing topological data analysis of collective motion, machine learning classification of swarming patterns, or the use of persistent homology in dynamical systems.

---

## 3. Data-Driven Equation Discovery & Balance Models

### [12] Messenger, D. A. & Bortz, D. M. (2021)
**"Weak SINDy for Partial Differential Equations."**
*Journal of Computational Physics*, 443, 110525.
DOI: [10.1016/j.jcp.2021.110525](https://doi.org/10.1016/j.jcp.2021.110525)

This paper extends the Weak SINDy (WSINDy) framework to the setting of partial differential equations. By replacing pointwise derivative approximations with a convolutional weak formulation, the method achieves machine-precision recovery of PDE coefficients from noise-free data and robust identification even under large noise (signal-to-noise ratios approaching one). The algorithm exploits the separability of test functions and uses the Fast Fourier Transform for efficient computation, achieving worst-case complexity $O(N^{D+1}\log(N))$ for datasets with $N$ points in $D+1$ dimensions. The paper also introduces a learning algorithm for the thresholding parameter in sequential-thresholding least-squares and a scale-invariance technique for poorly-scaled datasets. **Cite when:** introducing the WSINDy algorithm for PDEs, discussing weak-form system identification, or referencing robust sparse regression for equation discovery from noisy data.

---

### [13] Minor, E., Bhatt, H., Bortz, D. M., & Messenger, D. A. (2025)
**"Multi-Step Thresholded Least Squares with Dominant Balance for Sparse PDE Discovery."**
Preprint, 2025.

This paper introduces a dominant-balance variant of sequential thresholded least squares for sparse PDE discovery. The multi-step thresholding procedure refines the identification of active terms in governing equations by incorporating the concept of dominant balance — the idea that in different regions of parameter or physical space, different subsets of terms in a PDE may dominate the dynamics. This approach improves the accuracy and robustness of sparse regression methods for system identification, particularly when the governing PDE exhibits scale separation or regime-dependent behavior. **Cite when:** discussing refinements to sparse regression thresholding procedures, dominant balance analysis in PDE discovery, or multi-step algorithms extending WSINDy/SINDy-type methods.

---

### [6] Callaham, J. L., Koch, J. V., Brunton, B. W., Kutz, J. N., & Brunton, S. L. (2021)
**"Learning Dominant Physical Processes with Data-Driven Balance Models."**
*Nature Communications*, 12(1), 1016.
DOI: [10.1038/s41467-021-21331-z](https://doi.org/10.1038/s41467-021-21331-z)

This paper develops a data-driven framework that uses unsupervised learning to automatically identify dominant balance regimes in complex physical systems — regions of space-time where specific subsets of terms in the governing equations dominate the dynamics. The method combines sparse regression to identify locally active terms with clustering to delineate spatially and temporally coherent dominant balance regions, without requiring prior knowledge of the balance structure. Demonstrations on turbulent boundary layers, ocean circulation, and other fluid systems show that the approach successfully recovers known dominant balances and can reveal new physical insights. **Cite when:** discussing data-driven identification of dominant physical processes, unsupervised learning for regime identification in PDEs, or the concept of dominant balance in the context of sparse equation discovery.

---

### [19] Brunton, S. L., Proctor, J. L., & Kutz, J. N. (2016)
**"Discovering Governing Equations from Data by Sparse Identification of Nonlinear Dynamical Systems."**
*Proceedings of the National Academy of Sciences*, 113(15), 3932–3937.
DOI: [10.1073/pnas.1517384113](https://doi.org/10.1073/pnas.1517384113)

This paper introduces the Sparse Identification of Nonlinear Dynamics (SINDy) framework, which discovers governing equations directly from measurement data by leveraging the fact that most physical systems have dynamics with only a few active terms in a high-dimensional function space. SINDy constructs a library of candidate nonlinear functions of the state variables and uses sparse regression (sequentially thresholded least squares) to identify the fewest terms needed to accurately capture the dynamics. The method is demonstrated on the Lorenz system, a mean-field model for vortex shedding behind a cylinder, and other canonical nonlinear systems, recovering the correct governing equations even with moderate noise. SINDy is fast, interpretable, and provides a data-driven alternative to first-principles modeling. **Cite when:** introducing the SINDy framework for data-driven equation discovery, motivating sparse regression for system identification, or providing background for the WSINDy extension to PDEs.

---

## 4. Model Order Reduction for Transport-Dominated Systems

### [15] Reiss, J., Schulze, P., Sesterhenn, J., & Mehrmann, V. (2018)
**"The Shifted Proper Orthogonal Decomposition: A Mode Decomposition for Multiple Transport Phenomena."**
*SIAM Journal on Scientific Computing*, 40(3), A1322–A1344.
arXiv: [1512.01985](https://arxiv.org/abs/1512.01985)

This paper introduces the shifted proper orthogonal decomposition (sPOD), which extends classical POD by introducing time-dependent shifts of the snapshot matrix to handle transport-dominated phenomena. Classical POD struggles with traveling waves and coherent structures because such solutions do not lie in a low-dimensional linear subspace. The sPOD determines multiple transport velocities and separates the contributions of distinct moving structures, achieving low-rank decompositions where standard POD fails. The method is demonstrated on one- and two-dimensional test cases, showing clear superiority over standard POD for transport-dominated systems. **Cite when:** introducing the sPOD for transport-dominated problems, discussing limitations of classical POD, or referencing mode decomposition methods for traveling waves.

---

### [10] Krah, P., Baumgärtner, A., Żórawski, B., & Reiss, J. (2024)
**"A Robust Shifted Proper Orthogonal Decomposition: Insights from Proximal Methods."**
Preprint, 2024.

This paper extends the sPOD framework by developing robust proximal optimization algorithms for decomposing flows with multiple transports. The methods optimize co-moving fields directly and penalize their nuclear norm to promote low-rank structure, while adding a robustness term to handle interpolation errors and noise. The proximal algorithms are benchmarked against synthetic data and demonstrated on incompressible and reactive flows, showing improved separation of multiple transport phenomena compared to prior sPOD methods. The resulting methodology provides the same interpretability as standard POD for the individual co-moving fields. **Cite when:** discussing robust extensions of the sPOD, nuclear norm penalization in mode decomposition, or proximal optimization algorithms for transport-dominated flow decomposition.

---

### [3] Black, F., Schulze, P., & Unger, B. (2020)
**"Projection-Based Model Reduction with Dynamically Transformed Modes."**
*ESAIM: Mathematical Modelling and Numerical Analysis*, 54(6), 2011–2043.
DOI: [10.1051/m2an/2020046](https://doi.org/10.1051/m2an/2020046)

This paper proposes a model reduction framework for transport phenomena using time-dependent transformation operators that generalize the moving finite element method (MFEM) to arbitrary basis functions. The reduced model projects the original evolution equation onto a nonlinear manifold (rather than a linear subspace), enabling low-dimensional approximations with small errors where classical linear MOR fails. The paper provides an a-posteriori error bound based on residual minimization, shows a connection to the method of freezing (symmetry reduction), and analyzes the problem of finding optimal basis functions from full-order solution data. **Cite when:** discussing nonlinear model reduction for transport phenomena, projection onto nonlinear manifolds, or the connection between model reduction and symmetry reduction (method of freezing).

---

### [14] Peherstorfer, B. (2022)
**"Breaking the Kolmogorov Barrier with Nonlinear Model Reduction."**
*Notices of the American Mathematical Society*, 69(5), 725–733.
DOI: [10.1090/noti2475](https://doi.org/10.1090/noti2475)

This expository article reviews the fundamental challenge that transport-dominated problems pose for model reduction: hyperbolic equations, conservation laws, and traveling structures induce solution manifolds with slowly decaying Kolmogorov $n$-widths, meaning low-dimensional linear subspaces provide poor approximations (the "Kolmogorov barrier"). The paper surveys recent advances in nonlinear model reduction that overcome this barrier by approximating solutions on nonlinear manifolds rather than linear subspaces, using approaches such as transport-based methods, autoencoders, and composition-based techniques. **Cite when:** motivating the need for nonlinear model reduction, explaining the Kolmogorov barrier, or providing a high-level overview of transport-adapted MOR approaches.

---

### [20] Berkooz, G., Holmes, P., & Lumley, J. L. (1993)
**"The Proper Orthogonal Decomposition in the Analysis of Turbulent Flows."**
*Annual Review of Fluid Mechanics*, 25(1), 539–575.
DOI: [10.1146/annurev.fl.25.010193.002543](https://doi.org/10.1146/annurev.fl.25.010193.002543)

This seminal review paper provides the mathematical foundations and physical motivation for the Proper Orthogonal Decomposition (POD) — also known as the Karhunen–Loève decomposition or principal component analysis — in the context of turbulent flows. The authors present the theory of POD as an optimal linear basis for representing an ensemble of functions, prove its optimality in the energy (L²) norm, and survey applications to wall-bounded turbulence, free shear flows, and convection. The paper also discusses the Galerkin projection of the Navier–Stokes equations onto POD modes to derive low-dimensional dynamical systems and the relationship between symmetry, homogeneity, and the structure of the POD eigenfunctions. **Cite when:** introducing POD / SVD-based dimensionality reduction, motivating the use of POD for snapshot data, or referencing the theoretical foundations of proper orthogonal decomposition.

---

## 5. Symmetry Reduction Methods

### [5] Budanur, N. B., Cvitanović, P., Davidchack, R. L., & Siminos, E. (2015)
**"Reduction of Continuous Symmetries of Chaotic Flows by the Method of Slices."**
*Communications in Nonlinear Science and Numerical Simulation*, 29(1–3), 30–47.
DOI: [10.1016/j.cnsns.2015.04.022](https://doi.org/10.1016/j.cnsns.2015.04.022)

This paper presents the method of slices for continuous symmetry reduction of chaotic dynamical systems and demonstrates its application to high-dimensional problems. The authors show how to define a slice by minimizing the distance to a template, how to cover the reduced state space with multiple slices (one per dynamically prominent unstable pattern), and how to handle the singular crossings of inflection hyperplanes via tessellation. The approach is applied to the Kuramoto–Sivashinsky equation and other systems with continuous symmetry, enabling the systematic computation of relative periodic orbits. **Cite when:** introducing the method of slices for symmetry reduction, discussing how to quotient out continuous symmetries in chaotic/turbulent systems, or referencing tessellation strategies for global symmetry reduction.

---

### [4] Budanur, N. B. & Cvitanović, P. (2015)
**"Periodic Orbit Analysis of a System with Continuous Symmetry — A Tutorial."**
*Chaos*, 25(7), 073112.
DOI: [10.1063/1.4923742](https://doi.org/10.1063/1.4923742)

This tutorial paper illustrates different symmetry-reduction techniques on a 4-dimensional model with SO(2) symmetry and chaotic dynamics. The authors demonstrate that relative equilibria are conveniently found using symmetry-invariant polynomial bases, while the method of slices is preferable for analyzing chaotic dynamics in high dimensions. A Poincaré section on the slice reduces the flow to a unimodal map, enabling systematic determination of all relative periodic orbits and their symbolic dynamics. The paper then presents cycle averaging formulas for systems with continuous symmetry. **Cite when:** providing a pedagogical reference for symmetry reduction techniques, discussing the method of slices in practice, or referencing periodic orbit theory in systems with continuous symmetry.

---

### [16] Sonday, B. E., Singer, A., & Kevrekidis, I. G. (2013)
**"Noisy Dynamic Simulations in the Presence of Symmetry: Data Alignment and Model Reduction."**
*Annals of Applied Statistics*, 7(3), 1535–1557.
DOI: [10.1214/13-AOAS627](https://doi.org/10.1214/13-AOAS627)

This paper demonstrates eigenvector-based techniques for quotienting out symmetry degrees of freedom from noisy trajectory snapshots. Illustrative examples include the Kuramoto–Sivashinsky equation with periodic boundary conditions and a stochastic simulation of nematic liquid crystals modeled through a nonlinear Smoluchowski equation on a sphere. The authors show that "vector diffusion maps" can combine symmetry removal and dimensionality reduction in a single formulation — a useful first step toward data mining symmetry-adjusted snapshot ensembles for low-dimensional parametrizations. **Cite when:** discussing data-driven symmetry reduction from noisy simulations, alignment of trajectory snapshots with intrinsic symmetries, or the use of diffusion maps for combined symmetry reduction and dimensionality reduction.

---

## 6. Reduced Order Modeling & Multiscale Methods

### [1] Vargas Alvarez, H., Patsatzis, D. G., Russo, L., Kevrekidis, I., & Siettos, C. (2025)
**"Next Generation Equation-Free Multiscale Modelling of Crowd Dynamics via Machine Learning."**
arXiv: [2508.03926](https://arxiv.org/abs/2508.03926)

This paper proposes a combined manifold learning and machine learning approach to learn discrete evolution operators for emergent crowd dynamics in latent spaces from high-fidelity agent-based simulations. The four-stage pipeline uses: (1) kernel density estimation to derive continuous density fields from discrete pedestrian positions, (2) POD-based manifold learning for latent space construction, (3) LSTMs and multivariate autoregressive models for the reduced dynamics, and (4) mass-conserving reconstruction in the full space. Demonstrated on the Social Force Model in a corridor with obstacle, the approach creates an effective solution operator for the unavailable macroscopic PDE, enabling fast and accurate simulation of crowd dynamics. **Cite when:** discussing equation-free multiscale methods, machine learning surrogates for agent-based models, or manifold learning for crowd dynamics.

---

### [21] Hochreiter, S. & Schmidhuber, J. (1997)
**"Long Short-Term Memory."**
*Neural Computation*, 9(8), 1735–1780.
DOI: [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)

This paper introduces the Long Short-Term Memory (LSTM) recurrent neural network architecture, designed to overcome the vanishing gradient problem that prevents standard recurrent networks from learning long-range temporal dependencies. The key innovation is a memory cell with input, output, and forget gates that regulate information flow, allowing the network to store and access information over arbitrarily long time intervals. The authors demonstrate that LSTMs can learn to bridge time lags of more than 1000 discrete steps, far exceeding the capabilities of conventional RNNs, BPTT, and RTRL. LSTM has since become one of the most widely used architectures for sequence modeling, time-series prediction, and temporal dynamics learning. **Cite when:** introducing the LSTM architecture used in the ROM forecasting pipeline, discussing recurrent neural networks for temporal dynamics, or referencing the gated memory mechanism for learning long-range dependencies in latent-space evolution.

---

## 7. Numerical Methods for Stiff PDEs

### [7] Cox, S. M. & Matthews, P. C. (2002)
**"Exponential Time Differencing for Stiff Systems."**
*Journal of Computational Physics*, 176(2), 430–455.
DOI: [10.1006/jcph.2002.6995](https://doi.org/10.1006/jcph.2002.6995)

This paper develops exponential time differencing (ETD) methods for stiff systems of ODEs arising from spectral discretization of PDEs. The authors present second- and higher-order schemes, introduce Runge–Kutta variants (ETDRK), and extend the method to systems with nondiagonal linear parts. The key idea is to integrate the linear (stiff) part exactly and treat the nonlinear part with appropriate quadrature. Testing against integrating factor and linearly implicit methods on both dissipative and dispersive PDEs demonstrates superior accuracy. **Cite when:** introducing exponential time differencing methods, discussing numerical integration of stiff ODEs/PDEs, or referencing the original ETDRK scheme.

---

### [9] Kassam, A.-K. & Trefethen, L. N. (2005)
**"Fourth-Order Time-Stepping for Stiff PDEs."**
*SIAM Journal on Scientific Computing*, 26(4), 1214–1233.
DOI: [10.1137/S1064827502410633](https://doi.org/10.1137/S1064827502410633)

This paper presents a modification of the ETDRK4 scheme (from Cox & Matthews, 2002) that solves numerical instability issues in the original formulation and generalizes it to nondiagonal linear operators. The authors compare the modified ETD scheme against implicit-explicit (IMEX), integrating factor, time-splitting, and Fornberg–Driscoll "sliders" methods on the KdV, Kuramoto–Sivashinsky, Burgers, and Allen–Cahn equations in 1D. The modified ETDRK4 is found to be the best method for these applications with fixed time steps. Short MATLAB implementations are provided. **Cite when:** referencing the stabilized ETDRK4 time-stepping scheme, comparing numerical methods for stiff PDEs, or citing the practical MATLAB implementation.

---

## 8. Statistical & Optimization Methods

### [11] Meinshausen, N. & Bühlmann, P. (2010)
**"Stability Selection."**
*Journal of the Royal Statistical Society: Series B (Statistical Methodology)*, 72(4), 417–473.
arXiv: [0809.2932](https://arxiv.org/abs/0809.2932)

This paper introduces stability selection, a general method based on subsampling combined with high-dimensional selection algorithms (e.g., Lasso, graphical models, clustering). The approach provides finite-sample control for error rates of false discoveries and a transparent principle for choosing regularization parameters in structure estimation. The authors prove that for the randomized Lasso, stability selection achieves variable selection consistency even when the conditions needed for consistency of the original Lasso are violated. Demonstrations cover variable selection and Gaussian graphical modeling with real and simulated data. **Cite when:** discussing sparsity-promoting methods, subsampling-based model selection, false discovery control in high-dimensional settings, or the stability selection procedure used within WSINDy-type algorithms.

---

### [18] Wasserman, L. (2004)
**"All of Statistics: A Concise Course in Statistical Inference."**
*Springer Texts in Statistics*. Springer, New York.
ISBN: 978-0-387-40272-7

This textbook provides a broad, concise introduction to statistical inference covering probability, classical and Bayesian estimation, hypothesis testing, regression, nonparametric methods, and statistical learning. It is widely used as a graduate-level reference in statistics and machine learning. Topics particularly relevant to data-driven modeling include maximum likelihood estimation, bootstrap methods, regularization, and model selection criteria. **Cite when:** referencing foundational statistical concepts (estimation, hypothesis testing, convergence), providing a general statistics reference for methodology used in your thesis, or supporting theoretical claims about statistical inference.

---

### [22] Gavish, M. & Donoho, D. L. (2014)
**"The Optimal Hard Threshold for Singular Values is $4/\sqrt{3}$."**
*IEEE Transactions on Information Theory*, 60(8), 5040–5053.
DOI: [10.1109/TIT.2014.2323359](https://doi.org/10.1109/TIT.2014.2323359)

This paper derives the asymptotically optimal hard threshold for singular value truncation when the data matrix is a low-rank signal corrupted by i.i.d. Gaussian noise. When the noise level $\sigma$ is known, the optimal threshold is $(4/\sqrt{3})\,\sigma\sqrt{n}$ (for an $m \times n$ matrix with $m/n \to \beta$), and a closed-form expression involving the Marčenko–Pastur distribution is given for general aspect ratios. The authors also provide an optimal threshold for the case when $\sigma$ is unknown, based on the median singular value. This result provides a principled, parameter-free criterion for choosing the truncation rank in SVD/POD-based dimensionality reduction, replacing heuristics such as the "energy" or "elbow" criterion. **Cite when:** justifying the SVD truncation rank in the POD pipeline, referencing the optimal hard singular value threshold, or discussing principled rank selection for noisy data.

---

### [23] Zheng, P., Askham, T., Brunton, S. L., Kutz, J. N., & Aravkin, A. Y. (2019)
**"A Unified Framework for Sparse Relaxed Regularized Regression: SR3."**
*IEEE Access*, 7, 1404–1423.
DOI: [10.1109/ACCESS.2018.2886528](https://doi.org/10.1109/ACCESS.2018.2886528)

This paper proposes the Sparse Relaxed Regularized Regression (SR3) framework, a broad approach for solving regularized regression problems that introduces an auxiliary variable to split sparsity and accuracy requirements. The relaxation has three key advantages: (1) solutions are superior with respect to errors, false positives, and conditioning; (2) it enables extremely fast algorithms for both convex and nonconvex formulations; and (3) it applies to composite regularizers essential for total variation and tight-frame compressed sensing. The authors provide rigorous convergence theory and demonstrate SR3 on compressed sensing, LASSO, matrix completion, TV regularization, and group sparsity problems, uniformly outperforming state-of-the-art methods. SR3 underlies the sparse regression solver used in the WSINDy pipeline for PDE coefficient identification. **Cite when:** introducing the SR3 sparse regression framework, discussing the optimization backend of WSINDy/SINDy-type methods, or referencing relaxed formulations for sparsity-promoting regression.

---

## 9. Additional Reference Papers

> *These papers are in the thesis folder but are not in the current thesis bibliography [1]–[18]. They remain here for reference in case you add them later.*

---

### Grégoire, G. & Chaté, H. (2004)
**"Onset of Collective and Cohesive Motion."**
*Physical Review Letters*, 92(2), 025702.
DOI: [10.1103/PhysRevLett.92.025702](https://doi.org/10.1103/PhysRevLett.92.025702)

This paper re-examines the phase transition to collective motion in the Vicsek model and related systems with cohesion. The authors demonstrate that in two spatial dimensions the transition to ordered collective motion is always *discontinuous* (first-order), contradicting the continuous (second-order) transition originally reported by Vicsek et al. They also show that cohesion is always lost near the onset of collective motion due to the interplay of density, velocity, and shape fluctuations. **Cite when:** discussing the nature (first-order vs. second-order) of the phase transition to collective motion.

---

### Toner, J., Tu, Y., & Ramaswamy, S. (2005)
**"Hydrodynamics and Phases of Flocks."**
*Annals of Physics*, 318(1), 170–244.
DOI: [10.1016/j.aop.2005.04.011](https://doi.org/10.1016/j.aop.2005.04.011)

Comprehensive review covering the hydrodynamic theory of flocking. Classifies flock phases by symmetry (ferromagnetic, nematic) and develops long-wavelength equations for each. Shows that ordered flocks can exist in 2D in defiance of the Mermin–Wagner theorem. **Cite when:** introducing the continuum/hydrodynamic description of flocking or the Toner–Tu theory.

---

### Ginelli, F. (2016)
**"The Physics of the Vicsek Model."**
*European Physical Journal Special Topics*, 225(11–12), 2099–2117.
DOI: [10.1140/epjst/e2016-60066-8](https://doi.org/10.1140/epjst/e2016-60066-8)

Pedagogical lecture notes reviewing the Vicsek model, its algorithmic implementation, universality class, and connection to the Toner–Tu hydrodynamic theory. **Cite when:** providing a detailed reference for the Vicsek model's properties or its universality class.

---

### Wang, Y., Turci, F., Kadge, A., Hammond, I. S., Sherpa, Q., & Rowell, C. P. (2022)
**"Dominating Length Scales of Zebrafish Collective Behaviour."**
*PLOS Computational Biology*, 18(1), e1009697.
DOI: [10.1371/journal.pcbi.1009697](https://doi.org/10.1371/journal.pcbi.1009697)

3D study of fifty zebrafish identifying two dominant physical length scales (persistence length and nearest-neighbor distance) that capture behavioral transitions in collective motion. **Cite when:** referencing experimental data on zebrafish collective behavior or validating self-propelled particle models.

---

### Wong, I. Y. & Sandstede, B.
**"Cell Migration Manifolds."**
Brown University, unpublished research proposal.

Research proposal outlining a program to study collective cell migration on curved manifolds using TDA and manifold learning. **Cite when:** motivating TDA applied to cell biology. *Note: confirm citation format with your advisor.*

---

### Messenger, D. A. & Bortz, D. M. (2022)
**"Learning Mean-Field Equations from Particle Data Using WSINDy."**
*Physica D: Nonlinear Phenomena*, 439, 133406.
DOI: [10.1016/j.physd.2022.133406](https://doi.org/10.1016/j.physd.2022.133406)

Develops a weak-form sparse identification method for interacting particle systems combining mean-field theory with WSINDy. Converges at rate $O(N^{-1/2})$ for large particle numbers. **Cite when:** discussing data-driven identification of interacting particle systems or mean-field WSINDy.

---

### Minor, S., Elderd, B. D., Van Allen, B., Bortz, D. M., & Dukic, V. (2025)
**"Weak Form Learning for Mean-Field Partial Differential Equations: An Application to Insect Movement."**
arXiv: [2510.07786](https://arxiv.org/abs/2510.07786)

Extends weak-form equation learning with kernel density estimation to learn Fokker–Planck equations for fall armyworm movement from sparse experimental data. **Cite when:** discussing WSINDy applications to biological data or Fokker–Planck equation learning.

---

### Minor, S., Messenger, D. A., Dukic, V., & Bortz, D. M. (2025)
**"Learning Physically Interpretable Atmospheric Models from Data with WSINDy."**
arXiv: [2501.00738](https://arxiv.org/abs/2501.00738)

Uses WSINDy to discover interpretable PDEs governing atmospheric phenomena from simulated and ECMWF assimilated data. **Cite when:** discussing interpretable data-driven atmospheric modeling or WSINDy for geophysical fluids.

---

### Duchi, J., Shalev-Shwartz, S., Singer, Y., & Chandra, T. (2008)
**"Efficient Projections onto the ℓ₁-Ball for Learning in High Dimensions."**
*Proceedings of the 25th International Conference on Machine Learning (ICML)*, Helsinki, Finland, 272–279.

Efficient $O(n)$ algorithms for $\ell_1$-ball projection, enabling practical sparse optimization. **Cite when:** discussing $\ell_1$-regularization or computational methods underlying sparse regression.

---

### Lu, F., Mou, C., Liu, H., & Iliescu, T. (2022)
**"Stochastic Data-Driven Variational Multiscale Reduced Order Models."**
arXiv: [2209.02739](https://arxiv.org/abs/2209.02739)

Proposes a robust stochastic ROM closure (S-ROM) from multiple trajectories with random initial conditions, achieving space-time reduction with uncertainty quantification. **Cite when:** discussing stochastic closures for ROMs or data-driven variational multiscale methods.

---

### Froehlich, S. & Cvitanović, P. (2012)
**"Reduction of Continuous Symmetries of Chaotic Flows by the Method of Slices."**
*Communications in Nonlinear Science and Numerical Simulation*, 17(5), 2074–2084.
DOI: [10.1016/j.cnsns.2011.07.007](https://doi.org/10.1016/j.cnsns.2011.07.007)

Earlier work on the method of slices for SO(2) symmetry reduction, illustrated on the complex Lorenz equations. **Cite when:** referencing the original method of slices formulation.

---

## Quick Citation Look-Up Table — Thesis Bibliography [1]–[23]

| Ref | Short Key | Full Reference | Use For |
|-----|-----------|---------------|---------|
| [1] | Vargas Alvarez et al. (2025) | arXiv:2508.03926 | Equation-free crowd dynamics, ML + POD |
| [2] | Bhaskar et al. (2019) | Chaos 29, 123125 | TDA + ML for classifying collective motion |
| [3] | Black et al. (2020) | ESAIM: M2AN 54(6), 2011 | Projection MOR with transformed modes |
| [4] | Budanur & Cvitanović (2015) | Chaos 25, 073112 | Periodic orbit analysis with continuous symmetry |
| [5] | Budanur et al. (2015) | Commun. Nonlinear Sci. 29, 30 | Method of slices for symmetry reduction |
| [6] | Callaham et al. (2021) | Nature Comms. 12, 1016 | Data-driven dominant balance models |
| [7] | Cox & Matthews (2002) | J. Comput. Phys. 176, 430 | ETD methods for stiff systems |
| [8] | D'Orsogna et al. (2006) | PRL 96, 104302 | Swarming with Morse potential, pattern stability |
| [9] | Kassam & Trefethen (2005) | SIAM J. Sci. Comput. 26, 1214 | ETDRK4 (stabilized fourth-order ETD) |
| [10] | Krah et al. (2024) | Preprint | Robust sPOD with proximal methods |
| [11] | Meinshausen & Bühlmann (2010) | JRSS-B 72(4), 417 | Stability selection |
| [12] | Messenger & Bortz (2021) | J. Comput. Phys. 443, 110525 | WSINDy for PDEs |
| [13] | Minor, Bhatt et al. (2025) | Preprint | Multi-step thresholded LS, dominant balance |
| [14] | Peherstorfer (2022) | Notices AMS 69(5), 725 | Kolmogorov barrier & nonlinear MOR |
| [15] | Reiss et al. (2018) | SIAM J. Sci. Comput. 40(3), A1322 | Original shifted POD (sPOD) |
| [16] | Sonday et al. (2013) | Ann. Appl. Stat. 7(3), 1535 | Noisy symmetry reduction, diffusion maps |
| [17] | Vicsek et al. (1995) | PRL 75(6), 1226 | Original Vicsek model |
| [18] | Wasserman (2004) | Springer | All of Statistics — general stats reference |
| [19] | Brunton et al. (2016) | PNAS 113(15), 3932 | SINDy — sparse identification of dynamics |
| [20] | Berkooz et al. (1993) | Annu. Rev. Fluid Mech. 25, 539 | POD foundations for turbulent flows |
| [21] | Hochreiter & Schmidhuber (1997) | Neural Comput. 9(8), 1735 | LSTM architecture for sequence modeling |
| [22] | Gavish & Donoho (2014) | IEEE Trans. Info. Theory 60(8), 5040 | Optimal hard SVD threshold ($4/\sqrt{3}$) |
| [23] | Zheng et al. (2019) | IEEE Access 7, 1404 | SR3 sparse relaxed regression framework |
