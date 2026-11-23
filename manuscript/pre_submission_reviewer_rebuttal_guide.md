# Pre-Submission Reviewer Objection & Rebuttal Matrix
**Manuscript:** CoDe-CBF: Distribution-Free Safety Filtering for Decentralized Multi-UAV Swarms under Non-Stationary Aerodynamic Disturbances and Variable Communication Age-of-Information  
**Target Venue:** IEEE Robotics and Automation Letters (RA-L) / Elsevier Robotics and Autonomous Systems (RAS)

---

### Objection 1: Why not use an Extended State Observer (ESO) or Disturbance Observer CBF (DO-CBF) instead of Conformal Inference?
* **Reviewer Concern:** Disturbance observers are well-established in quadrotor flight control. Why introduce statistical conformal inference?
* **Author Rebuttal & Evidence:**
  1. ESOs require high observer gains to track rapid wind shear, which amplifies sensor noise and induces phase lag during abrupt step gusts. In close-proximity flight (1m separation), phase lag of just 100 ms allows collision before the observer converges.
  2. DO-CBFs provide asymptotic stability under bounded disturbance derivatives, but offer **zero probabilistic coverage guarantees** under sudden stochastic downwash spikes.
  3. CoDe-CBF's Adaptive Conformal Inference directly quantifies the true non-stationary residual distribution online and guarantees finite-sample coverage (Theorem 1) without requiring observer gain tuning or bounded derivative assumptions.

---

### Objection 2: Conformal Prediction requires exchangeability; multi-UAV flight trajectories are non-stationary and non-exchangeable.
* **Reviewer Concern:** Standard conformal prediction relies on the exchangeability assumption ($z_i \sim 	ext{i.i.d. } P$), which is strictly violated by wind gusts, rotor downwash, and trajectory curvature.
* **Author Rebuttal & Evidence:**
  1. We explicitly state in Section I and Section II that **offline split-conformal prediction fails** precisely because exchangeability is violated.
  2. CoDe-CBF does **not** assume exchangeability. We utilize **Adaptive Conformal Inference (ACI)**, which updates the nominal quantile coverage level $lpha_t$ via online stochastic gradient tracking on past miscoverage errors.
  3. Theorem 1 rigorously proves that for **any arbitrary, non-stationary disturbance process**, the long-run empirical safety violation rate converges to $lpha_{	ext{target}}$ at rate $\mathcal{O}(1/T)$ almost surely.

---

### Objection 3: How can a decentralized quadratic program guarantee real-time 50 Hz execution on resource-constrained drone compute?
* **Reviewer Concern:** Solving optimization problems onboard edge microprocessors can cause jitter and deadline overruns.
* **Author Rebuttal & Evidence:**
  1. The CoDe-CBF QP is strictly formulated over **4 decision variables** ($u_{i,x}, u_{i,y}, u_{i,z}, \delta_i$) with $K+4$ linear affine constraints (where $K$ is the number of neighbors, typically $\le 5$).
  2. Solved via OSQP with C-level warm starting, the median solve time across 1.15 million iterations in our 1,000-seed benchmark was **0.55 ms** in Python and **< 0.05 ms** in C++.
  3. Against the 20 ms control loop period ($\Delta t = 0.02	ext{ s}$ at 50 Hz), CoDe-CBF consumes less than $3\%$ of the available tick budget, requiring zero GPU acceleration and $< 0.1	ext{ MB}$ RAM footprint.

---

### Objection 4: Why couple communication Age-of-Information (AoI) with physical aerodynamic residuals?
* **Reviewer Concern:** Why not simply treat communication delay as an independent network QoS layer?
* **Author Rebuttal & Evidence:**
  1. In multi-UAV swarms, physical safety barriers are evaluated against estimated neighbor states. Under delayed broadcasts, position uncertainty grows as $	ilde{p}_j(t) \sim 	au_{ij}(t) \Delta t \cdot v_j$.
  2. Decoupled controllers evaluate barriers using stale positions without margin inflation, leading to immediate boundary breach when bursts of packets drop.
  3. CoDe-CBF's score normalization $s_t = rac{r_t}{1 + \lambda_{	ext{aoi}} 	au_{ij} \Delta t}$ makes the nonconformity score scale-invariant across communication bursts and proactively inflates the effective margin $\hat{q}_t$ during packet drop events, preventing collisions before stale states lead to failure.

---

### Objection 5: Is recursive feasibility mathematically guaranteed during severe disturbance shocks?
* **Reviewer Concern:** Control Barrier Functions subject to hard acceleration bounds can become infeasible during sudden external disturbances.
* **Author Rebuttal & Evidence:**
  1. Theorem 2 provides a complete proof of recursive feasibility. By introducing a strictly penalized quadratic slack variable $ho_{	ext{slack}} \delta_i^2$ ($ho_{	ext{slack}} = 2000$), the feasible set is guaranteed non-empty for all state space configurations.
  2. We prove via KKT stationarity that $\delta_i^* = 0$ strictly holds whenever nominal control authority suffices, while bounded slack smoothly absorbs physical shocks during extreme unmodeled disturbances without crashing the solver.
