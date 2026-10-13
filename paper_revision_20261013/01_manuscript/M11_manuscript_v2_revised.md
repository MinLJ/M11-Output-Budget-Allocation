---
title: "Learning Prefix Marginal Utility for Resource-Aware Output Budget Allocation in Intelligent Vehicle Perception"
author:
  - "Yuyi Zhu, Yang Luo*, and Yi Chen**"
---

<div class="frontmatter-note">

**Author affiliation.** Industrial Intelligent Agent Research Center, School of Intelligent Manufacturing Ecosystem, XJTLU Entrepreneur College (Taicang), Xi'an Jiaotong-Liverpool University, Suzhou 215123, China. *Yang Luo and Yi Chen are the corresponding authors. Correspondence: Yi.Chen02@xjtlu.edu.cn.

**Abstract—**A fixed detector can produce more candidate records than a downstream interface is configured to retain. Under a shared output budget, the remaining decision is how many detections to keep for each input. Detector confidence alone does not express the additional object coverage obtained by extending a retained prefix. Two properties make the decision difficult. The supervision signal must be inferred from the detector's own outputs, because object annotations are unavailable when the allocation is made. The shared budget also couples the decisions across images, so prefix lengths have to be chosen jointly instead of per image. The allocation problem is studied here with the detector and the candidate order fixed. The method learns prefix-matching increments at ten localization thresholds, combines them into class-weighted utility curves, and selects prefix lengths by exact dynamic programming. On a BDD100K-derived confirmation set, it improves Coverage by 0.0792 objects per image and the class-weighted utility by 0.1100 over score-based allocation, averaged over mean budgets of 10, 15, and 20 records per image. Separately fitted allocators for YOLOv8n and Faster R-CNN also improve coverage and standard detection metrics on their respective fixed pre-NMS candidate streams. On COCO2017 Road8, the frozen estimator degrades; target-domain fitting improves on frozen transfer and yields a small weighted-utility gain alongside a core-budget coverage deficit against the score baseline. These results support learning the value of prefix extensions for constrained output selection, while identifying domain dependence and additional allocation cost as practical limits.

**Index Terms—**Object detection, output budget allocation, marginal utility, resource-aware perception, intelligent vehicle perception.

</div>

# I. Introduction

Object detection does not end with generating candidate boxes: a perception system must also decide which records to retain for subsequent processing. Consider an output interface in which several inputs share a fixed number of detection slots. Once the detector has produced their candidates, assigning more slots to one image leaves fewer for the others. Equal quotas are simple, but they ignore differences in the usefulness of additional detections. This paper studies how to distribute this shared output capacity when the detector and the candidate order within each image are fixed.

Figure 1 contrasts equal and adaptive prefix allocations. A sparse scene may require only a short prefix to cover its objects, whereas a crowded scene may benefit from retaining further candidates. Confidence-based allocation can already assign different quotas across images; its limitation is therefore not a lack of adaptivity. The issue is what each additional record contributes. A high-confidence candidate may duplicate an object covered by the preceding prefix, while a lower-confidence candidate in another image may cover a new object. The quantity needed for allocation is the marginal utility of extending the prefix, which depends on the next candidate and the detections already retained.

![Fig. 1](figures/fig1.png)

**Fig. 1.** Uniform versus adaptive allocation of a shared output budget. Equal quotas waste slots on duplicates in sparse scenes and miss useful objects in crowded ones, whereas scene-aware prefix lengths reallocate the same budget. Schematic; values are prescribed for illustration.


Existing approaches address several related resource decisions. Efficient detector architectures [1]–[7] reduce the cost of prediction, dense and region-based alternatives set their candidate sets through fixed anchor grids or refined proposals [8]–[11], and dynamic inference varies computation across inputs [12]–[14]. Adaptive proposal-count methods [15] determine how many hypotheses a detection head should process. The decision studied here follows candidate generation: for each image $i$ in a group $I_g$, the method chooses a retained length $K_i$ such that $\sum_{i \in I_g} K_i = B_g$. Each output remains an original-score prefix. This setting permits direct comparisons between raw scores, calibrated scores [16], [17], and learned prefix values under the same output budget.

Estimating these values presents two linked challenges. The true gain of a prefix extension depends on object annotations, which are unavailable during allocation. It must be inferred from the detector's existing outputs. In addition, the shared budget couples the decisions across images. A useful low-ranked candidate may follow several records with little additional value, so obtaining its gain requires retaining the intervening prefix. The marginal values need not decrease with rank, and selecting the largest currently available gain greedily can overlook a better combination of prefix lengths.

Prefix values are learned from matching increments. During training, each extension is labeled by the increase in same-class maximum matching between retained candidates and annotations at ten localization thresholds. Unlike a target that only records whether a candidate can match an object, this supervision measures whether its addition increases the coverage of the current prefix. A compact multilayer perceptron predicts the ten increments from candidate attributes, preceding-prefix statistics, image-level summaries, and detector-native features. Calibrated probabilities are combined with class weights and accumulated into image-level utility curves. Exact dynamic programming then chooses the prefix lengths jointly under the shared budget. The design separates utility estimation from allocation, allowing the predictor to be fitted to a detector's feature interface while retaining the same optimization problem.

Experiments evaluate both the allocation gains and the estimator's transfer behavior. On a BDD100K-derived confirmation set [18], learned allocation improves Coverage by 0.0792 objects per image and QUALITY by 0.1100 over score-based allocation, averaged over output budgets of 10, 15, and 20 records per image. Controlled comparisons examine prefix supervision, feature inputs, and the exact solver. Separately fitted allocators for YOLOv8n and Faster R-CNN also improve coverage, AP, and AR100 on their respective fixed pre-NMS candidate streams. On COCO2017 Road8 [19], frozen transfer instead degrades; target-domain fitting improves on the transferred model but retains a core-budget coverage–quality trade-off against score-based allocation. These comparisons distinguish the benefits of learning prefix increments from the ability of a fitted estimator to transfer across data distributions.

This paper contributes three things. First, output-budget allocation is defined as the joint selection of score-ordered detection prefixes under a group budget, and utility estimation is separated from the allocation decision (Sections III-A to III-C). Second, the predictor is supervised with same-class maximum-matching increments rather than candidate matchability, so that a candidate is credited only when its addition increases the number of covered objects (Section III-B). Third, the joint selection is solved exactly by dynamic programming, and the cost of doing so is measured against a greedy solver (Sections III-D and IV-D).

# II. Related Work

**Adaptive computation and candidate counts.** Efficient detection methods adjust the computation used to produce candidate predictions [1]–[7]. RT-DETR [1] combines an efficient hybrid encoder with query selection, while Salience DETR [20] restricts transformer encoding to queries selected through hierarchical salience filtering. PaQ-DETR [21] generates image-specific queries from shared latent patterns and uses quality-aware assignment to balance their supervision. Dense and two-stage detector families set the candidate set through different mechanisms: single-shot detectors score a fixed grid of anchors [8]–[10], whereas region-based detectors refine a proposal set [11], [22]. Dynamic inference adapts how much computation an input receives [12]–[14].

Proposal-count decisions are the closest existing problem to output-count allocation. He and Li [15] estimate foreground-region counts and use them to determine how many proposals a two-stage detection head processes; Guan et al. [23] select a per-image adaptive proposal number from proposal statistics before region-based detection. Both formulations recognize that a single proposal count can be excessive for sparse images and insufficient for crowded ones. The difference from the present work is one of stage and resource: those methods decide how many hypotheses a detector should *produce*, which changes candidate generation and the computation spent on it. This paper decides how many already-produced records to *keep* under a group-level budget shared across images, with candidate generation, candidate order, and detector computation all fixed. The two decisions are complementary, and the post-detector decision remains open even when a proposal-count policy is optimal.

**Candidate value and confidence.** Candidate selection depends on both prediction reliability and redundancy. Soft-NMS [24] reduces detection scores as a function of overlap with selected boxes, changing the ordering used during suppression. Learned non-maximum suppression [25] trains a network to decide which detections to keep, but again operates inside suppression rather than under a shared budget. Oksuz et al. [26] study scene-level uncertainty, detection calibration, and robustness to domain shift within the self-aware object detection task. Kuzucu et al. [16] examine pitfalls in detector-calibration evaluation and develop detection-specific Platt scaling and isotonic regression with localization-aware targets; modern neural networks more generally are known to be miscalibrated [17]. Their results establish post-hoc calibration as a substantive alternative to raw detector confidence. Calibrated values are therefore included in the fixed-prefix allocation comparisons. The relevant distinction is the prediction target: localization-aware calibration estimates detection quality under its matching protocol, whereas the supervision used here measures the change in matching cardinality when a specified prefix is extended. An individually matchable candidate can have zero prefix increment when earlier records already provide the same object coverage. The comparison tests whether learning this increment improves budget allocation beyond improving the confidence values themselves.

**Resource-aware collaborative perception.** Communication-constrained perception also assigns capacity according to the information available across inputs. Where2comm [27] uses spatial confidence maps to select perceptually relevant regions for transmission and adjusts the selected regions to available bandwidth. WhisperNet [28] uses sender-side saliency metadata to construct a receiver-side request plan, allocating feature contributions across agents before fusion. Communication-graph and attention-based collaboration methods decide what each agent transmits to the others [29]–[31]. These methods connect information selection with a shared resource constraint. The allocation unit here is a detection record, and the available decisions are prefix lengths of already generated candidate streams. This yields a discrete allocation problem whose learned values account for the detections retained within each prefix.

# III. Proposed Method

## A. Fixed-Prefix Allocation

For image $i$, a fixed detector and candidate extraction procedure produce an ordered sequence $C_i = (c_{i1}, \ldots, c_{iN_i})$. Each record contains a predicted class, a bounding box, and a detector score. Candidates follow descending score order, and retaining $K_i$ records returns $R_i(K_i) = (c_{i1}, \ldots, c_{iK_i})$. The budget counts candidate records: different class hypotheses from the same detector source remain separate records.

Let $I_g$ contain $n$ images sharing $B_g$ output slots. Each image retains a mandatory prefix of length $L$ and at most $H_i \le N_i$ records. The feasible allocations of (1) are

$$K_g = \Big\{ K \in \mathbb{Z}^n : L \le K_i \le H_i,\ i \in I_g,\ \sum_{i \in I_g} K_i = B_g \Big\}. \qquad (1)$$

The budget is feasible when $nL \le B_g \le \sum_{i \in I_g} H_i$. The reference configuration uses $N_i = 100$, $L = 5$, and $H_i = 50$. The retained lengths are chosen jointly within each group.

Prefix utility is measured by the objects it covers across localization thresholds. Let $m^\tau_{i,a}(k)$ be the maximum number of one-to-one matches between class-$a$ records in $R_i(k)$ and class-$a$ annotations, with an edge permitted when intersection over union (IoU) is at least $\tau$. For thresholds $T = \{0.50 + 0.05j : j = 0, \ldots, 9\}$ and positive class weights $w_a$, the utility of (2) is

$$U_i(k) = \frac{1}{|T|} \sum_{\tau \in T} \sum_{a} w_a m^\tau_{i,a}(k). \qquad (2)$$

The marginal utility is $\Delta_i(k) = U_i(k) - U_i(k-1)$. The ideal allocation maximizes $\sum_{i \in I_g} U_i(K_i)$ over $K_g$. Because these utilities require annotations, their increments are learned from training data and predicted curves are constructed for inference. Figure 2 summarizes this learning and allocation process.

![Fig. 2](figures/fig2.png)

**Fig. 2.** Prefix-increment learning and budget allocation. Training labels each optional position by same-class maximum-matching increments at ten IoU thresholds, and inference aggregates calibrated predictions into prefix utility curves whose lengths are chosen jointly by exact dynamic programming.


## B. Learning Prefix Increments

**Matching-based supervision.** The training target measures what changes when the retained prefix grows by one record. Write $m^\tau_i(k) = \sum_a m^\tau_{i,a}(k)$. For each optional position $k = L+1, \ldots, H_i$, the target at threshold $\tau$ is

$$y^\tau_{ik} = m^\tau_i(k) - m^\tau_i(k-1) \in \{0, 1\}. \qquad (3)$$

Adding one record can increase the maximum matching cardinality by at most one. The candidate order is scanned separately at each threshold, and augmenting paths update the matching, allowing previously retained records to be reassigned. A candidate that overlaps a valid object can receive a zero target when the preceding prefix already covers that object. It receives a positive target only when its addition increases the number of matched objects.

This distinction aligns supervision with the prefix-length decision. Figure 3 illustrates how repeated high-confidence hypotheses and lower-scored candidates covering new objects can favor different allocations. In the reference configuration, labels are constructed for $k = 6, \ldots, 50$; the first five records participate in every matching state. The targets describe fixed prefix extensions and can be reused across output budgets.

![Fig. 3](figures/fig3.png)

**Fig. 3.** High confidence is not high marginal utility. With a shared budget of six records across the two images, allocation by predicted marginal utility matches six labeled objects versus four under score-based allocation. Schematic; values are prescribed for illustration.


**Prefix-conditioned features.** Each position is represented by $x_{ik}$, combining candidate attributes, statistics of ranks $1, \ldots, k-1$, and an image-level summary of the available Top100 stream. Candidate attributes include score, rank, predicted class, box geometry, and detector-native information. Prefix statistics capture the preceding scores and same-class spatial relations, including overlap and center distance. The image summary describes score distributions, class proportions, and box sizes. Together, these features provide context for estimating whether another record adds coverage.

The RT-DETR reference implementation uses 58 candidate, 12 prefix, and 20 image-context features, giving a 90-dimensional input. Native information comprises Road8 logits and the final-decoder query embedding, gathered by image and query identity. PCA projects the embedding to 32 dimensions before feature concatenation and standardization. PCA and continuous-feature scaling use the fitting subset; class indicators and the prefix-presence flag retain their binary values. Detector-specific feature sources and fitting configurations are reported in Section IV.

**Prediction and calibration.** A multilayer perceptron returns ten logits $z_{ik} = f_\theta(x_{ik})$. Its layer widths are $90 \to 128 \to 64 \to 10$, with LayerNorm after the first linear layer and GELU after each hidden transformation. For fitting positions $D_{\text{fit}}$, training minimizes the loss of (4),

$$\mathcal{L} = \frac{1}{|D_{\text{fit}}|\,|T|} \sum_{(i,k) \in D_{\text{fit}}} \sum_{\tau \in T} \ell_{\text{BCE}}\big(z^\tau_{ik}, y^\tau_{ik}\big), \qquad (4)$$

where $\ell_{\text{BCE}}$ is binary cross-entropy with logits. Positions and threshold outputs receive equal loss weights. Model selection uses binary cross-entropy on an early-stopping subset. A separate calibration subset fits one positive temperature $T_{\text{cal}}$ per model by minimizing the same loss. This temperature is shared by the ten outputs, giving $\hat{p}^\tau_{ik} = \sigma(z^\tau_{ik}/T_{\text{cal}})$, where $\sigma$ is the sigmoid function.

## C. Quality-Aware Utility Estimation

The ten predictions describe additional object coverage at different localization thresholds. Their mean credits extensions that increase matching cardinality across those thresholds. Class weights specify the relative value assigned to each object category. With fitting-set object counts $n_a$ and $n_{\max} = \max_a n_a$, the weights of (5) are set to

$$\alpha_a = \min\left(4, \sqrt{\frac{n_{\max}}{\max(n_a, 1)}}\right), \quad
Z = \frac{\sum_a n_a \alpha_a}{\sum_a n_a}, \quad
w_a = \frac{\alpha_a}{Z}. \qquad (5)$$

The inverse-frequency factor increases the relative weight of less frequent classes, the cap limits its magnitude, and $Z$ normalizes the object-frequency-weighted mean to one. These weights enter utility aggregation and the QUALITY metric; training retains the unweighted loss in (4).

Adding $c_{ik}$ changes only the matching component of its predicted class $a_{ik}$. The true and estimated marginal utilities of (6) are therefore

$$\Delta_i(k) = \frac{w_{a_{ik}}}{|T|} \sum_{\tau \in T} y^\tau_{ik}, \qquad
\hat{\Delta}_i(k) = \frac{w_{a_{ik}}}{|T|} \sum_{\tau \in T} \hat{p}^\tau_{ik}. \qquad (6)$$

The configuration that combines class weighting with multi-threshold averaging is the reference method of this paper, written descriptively as class-weighted multi-threshold prefix-increment utility and labeled **M11**. The label follows the two-bit naming used for the aggregation study in Section IV-D. The first bit marks class weighting and the second marks multi-threshold averaging. M00 (neither), M10 (weighting only), and M01 (threshold averaging only) are the remaining combinations of the same predictor. Averaging follows probability calibration. The resulting prefix curves define the inference objective of (7):

$$\hat{U}_i(K) = \sum_{k = L+1}^{K} \hat{\Delta}_i(k), \qquad
\hat{K} = \arg\max_{K \in K_g} \sum_{i \in I_g} \hat{U}_i(K_i). \qquad (7)$$

Setting $\hat{U}_i(L) = 0$ removes the base-prefix contribution shared by every feasible allocation. The first $L$ records are still retained and evaluated. Predicted utilities determine the prefix lengths, while the original detector scores remain attached to the selected records.

## D. Exact Budget Allocation

The predicted curves are nondecreasing because their increments are nonnegative, but those increments need not decrease with rank. Reaching a valuable later candidate may require retaining several less valuable records first. Consequently, repeatedly choosing the largest currently available increment can overlook a better combination of prefixes. Complete prefix lengths are optimized with dynamic programming.

Index the group's images in ascending image-identity order as $1, \ldots, n$. Let $F_i(b)$ be the maximum predicted utility obtained by assigning exactly $b$ records to the first $i$ images. Initialize $F_0(0) = 0$ and every unreachable state to $-\infty$. The recurrence in (8) is

$$F_i(b) = \max_{K \in \{L, \ldots, \min(H_i, b)\}} \Big\{ F_{i-1}(b - K) + \hat{U}_i(K) \Big\}. \qquad (8)$$

Each transition chooses one complete prefix for image $i$. Backtracking from $F_n(B_g)$ recovers the retained lengths. Objectives are compared in float64; exact ties select the lexicographically smallest complete length vector in the fixed image order.

One forward table supports all requested budgets up to the largest total capacity $C$. Let $W = \max_i(H_i - L + 1)$ and let $q$ be the number of requested budgets. Transitions cost $O(nCW)$, lexicographic state ranking adds $O(nC \log C)$, and backtracking costs $O(qn)$. Stored predecessors, rolling states, and input curves require $O(nC + C + nW)$ space. The same predicted curves are reused across budgets; only the jointly selected prefix lengths change.

# IV. Experiments

## A. Experimental Setup

**Datasets.** The primary evaluation uses the BDD100K-derived protocol [18] adopted for this project, referred to as LargeClean-v1 (LC-v1). LC-v1 is a project-defined split and evaluation protocol rather than an official BDD100K split name; its roles are defined on BDD100K official-train identities after a frozen exclusion list, drawing on the 68,074 eligible identities that remain after the 1,926-identity historical exclusion, and they must not be confused with the official train/val/test naming. LC-v1 contains 10,000 training images, divided into 8,000 fitting, 1,000 early-stopping, and 1,000 calibration images, together with separate 2,000-image DEV and TEST sets. DEV supports method development and additional comparisons; TEST evaluates the frozen M11 configuration. Procedural information on the one recorded deviation in TEST annotation access is provided in Appendix A. The eight Road8 classes are person, bicycle, car, motorcycle, bus, train, truck, and traffic light. DEV and TEST contain 29,966 and 29,704 valid objects, respectively; images without valid objects remain in the evaluation.

For cross-dataset evaluation, all 5,000 COCO2017 validation images are used, restricted to the same eight categories [19]. They contain 14,897 valid non-crowd Road8 objects; the 1,940 images without such objects also participate in allocation. Target-domain fitting divides train2017 into 94,629 fitting images and 11,829 images for each of early stopping and calibration. The reference detector's published configuration uses COCO training and validation data; this experiment evaluates the allocator on its fixed outputs. The YOLO and Faster R-CNN branches use the BDD training and DEV images. Their detector fitting and feature interfaces are specified below; YOLO has its own image-level role assignments and allocation groups.

**Candidates and implementation.** RT-DETRv2-R18VD [2] is fixed throughout the reference experiments. Inputs are resized to 640 × 640 without letterboxing. The final decoder supplies 300 queries, each with eight Road8 class hypotheses; allocation consumes their highest-scoring 100 records without NMS or score thresholding. The predictor and feature construction of Section III are used, with PCA and scaling fitted on the fitting subset. Both the LC-v1-trained and the COCO-fitted RT-DETR allocators use AdamW with a constant learning rate of $10^{-3}$, weight decay $10^{-4}$, batches of 4,096 prefix positions, and a maximum of 30 epochs. Early stopping uses patience five on (4), and a separate calibration subset fits $T_{\text{cal}} \in [0.25, 4]$. Three models are trained independently for each learned configuration and evaluated separately before averaging their metrics.

**Budgets and baselines.** Unless varied explicitly, each group contains $n = 40$ images, $L = 5$, $H_i = 50$, and $B_g = n\overline{K}$, where $\overline{K} \in \{10, 15, 20, 30, 40\}$. Groups follow deterministic image-identity orderings. Core summaries average budgets 10, 15, and 20 equally. Fixed-K (S_FIXED) assigns $K_i = \overline{K}$; Score (S_ADAPT) maximizes the sum of retained detector scores. DEV comparisons also include class-weighted scores (Class-score), global temperature calibration, and localization-aware class-specific Platt calibration [16]. Platt uses calibrated values in prefix allocation; Weighted Platt multiplies them by the same fitting-set class weights as M11. The maps are fitted on FIT and applied to all available records. All methods within a comparison share the candidate stream, prefix bounds, group membership, and exact output budget.

**Metrics and uncertainty.** Coverage is the same-class maximum matching count at IoU 0.50, averaged over images. Coverage recall divides the total by valid object support. QUALITY averages (2) over images. AP, AP50, AP75, and AR100 use original detector scores and are computed on the complete evaluation set. COCO matching utilities use valid non-crowd objects; AP/AR retain the evaluator's crowd and area handling. Tables report AP/AR in percent and their differences in percentage points. Reported AP and AR values are low in absolute terms because they are computed on the raw pre-NMS candidate stream used for allocation, which retains duplicate and low-score hypotheses. These values are comparable across methods within a comparison, because all methods share the same candidate stream, prefix bounds, and budget; they are not comparable to published AP values, which are computed after non-maximum suppression. For Coverage and QUALITY, paired bootstrap resamples complete allocation groups 5,000 times after averaging the seed-level differences within each group. TEST retains its prespecified 97.5% confidence intervals; the development studies report 95% intervals. AP/AR are full-set point estimates. Every learned entry is the arithmetic mean over the three independently trained and separately evaluated models; where a ± value is given, it is the standard deviation across those three model evaluations, and no spread is defined for the deterministic baselines. Seed spreads for the ablation table are reported in Table V; for the remaining tables the per-run values are held in the frozen evaluation records of each arm.

## B. Main Results

**Confirmation on BDD.** Table I(a) evaluates M11 against Score on TEST. Across the core budgets, M11 increases Coverage by 0.0792 objects per image and QUALITY by 0.1100, with 97.5% confidence intervals of [0.0621, 0.0964] and [0.0948, 0.1254]. The candidates and their ordering are identical; the improvement comes from changing the retained lengths across images. All three models have positive core gains. At $\overline{K} = 10$, AP increases by 0.885 percentage points and AR100 by 1.921 points. These gains narrow at larger budgets (Fig. 4). At $\overline{K} = 40$, Coverage, QUALITY, and AP remain higher, but AR100 decreases by 0.780 points. Class-wise results also differ: core coverage recall increases by 3.42 points for person and 5.57 for bus, while car and motorcycle decrease by 0.23 and 0.51 points. Motorcycle and train have only 88 and six TEST objects, respectively.

![Fig. 4](figures/fig4.png)

**Fig. 4.** Gain over score-based allocation (percentage points) on TEST as a function of the average output budget $\overline{K}$. Detection gains shrink as the budget grows, and AR100 turns negative at $\overline{K} = 40$.


**Comparison with calibrated scores.** On DEV, Score already improves core Coverage over Fixed-K by 0.5665 objects per image (Table I(b)). M11 adds a further 0.0934. A global temperature and the budget-matched score-threshold implementation reproduce Score selections, whereas class-specific calibration changes allocations. Against Platt, M11 gains 0.0291 Coverage/image and 0.1511 QUALITY/image; against Weighted Platt, the gains are 0.2868 and 0.1120, with 95% intervals of [0.2633, 0.3090] and [0.0979, 0.1260]. Thus the learned increments improve both allocation metrics beyond these score-based value estimates. The comparison retains class-level trade-offs: bicycle and motorcycle lose 3.55 and 3.16 coverage-recall points relative to Weighted Platt. These Platt baselines adapt the source method to fixed-prefix budgeting: calibrated values are evaluated without its final threshold-based removal. At $\overline{K} = 40$, approximately 70.5% of their selected records lie below the fitting threshold. The calibration comparison therefore includes extrapolation into this low-score region.

**TABLE I.** BDD100K-based results with RT-DETRv2-R18VD. Coverage and QUALITY are per-image counts; AP/AR100 are percentages. M11 averages three independent models; the ± value is the standard deviation across those three model evaluations, and no spread is defined for the deterministic baselines.

| $\overline{K}$ | Method | Coverage | QUALITY | AP (%) | AR100 (%) |
|---:|---|---:|---:|---:|---:|
| **(a) TEST: frozen-method confirmation** | | | | | |
| 10 | Score | 6.8600 | 4.0963 | 15.83 | 21.55 |
| 10 | M11 | 6.9228 | 4.2211 | 16.71 | 23.47 |
| 15 | Score | 8.1515 | 4.7009 | 17.62 | 26.15 |
| 15 | M11 | 8.2245 | 4.8117 | 18.02 | 27.35 |
| 20 | Score | 8.8465 | 5.0398 | 18.20 | 28.29 |
| 20 | M11 | 8.9483 | 5.1341 | 18.44 | 29.26 |
| 30 | Score | 9.6540 | 5.4114 | 18.70 | 30.85 |
| 30 | M11 | 9.7692 | 5.4797 | 18.85 | 31.66 |
| 40 | Score | 10.1195 | 5.6232 | 18.93 | 33.32 |
| 40 | M11 | 10.1685 | 5.6518 | 18.97 | 32.54 |
| **(b) DEV: core-budget means** | | | | | |
| Core | Fixed-K | 7.4563 | 4.4028 | 15.62 | 21.76 |
| Core | Score | 8.0228 | 4.7480 | 16.56 | 23.83 |
| Core | Class-score | 7.4395 | 4.6441 | 16.78 | 25.46 |
| Core | Platt | 8.0872 | 4.7154 | 16.40 | 23.14 |
| Core | Weighted Platt | 7.8295 | 4.7545 | 16.88 | 25.22 |
| Core | M11 | 8.1163 ± 0.0123 | 4.8665 ± 0.0011 | 17.03 ± 0.02 | 25.27 ± 0.11 |

## C. Cross-Dataset and Cross-Detector Evaluation

**Frozen transfer and target-domain fitting.** The frozen LC-v1-trained allocator transfers the BDD-trained predictor, PCA, scaler, temperatures, and class weights to COCO. COCO-fitted re-estimates the predictor, preprocessing, and temperature from the COCO training roles while retaining the detector, class weights, and optimizer recipe. The two routes use the same 125 allocation groups. Target-domain fitting was conducted after the frozen-transfer results were observed.

Table II reports all five budgets. Score improves on Fixed-K by 0.3172 Coverage/image and 0.3502 QUALITY/image across the core budgets. The frozen LC-v1-trained allocator instead falls below Score in both metrics at every budget. Its core differences are −0.0181 Coverage/image (95% CI [−0.0214, −0.0150]) and −0.0178 QUALITY/image ([−0.0219, −0.0138]); AP and AR100 decrease by 0.273 and 0.883 percentage points. Across the core budgets, the three-model seed spread is small: the frozen configuration gives 2.6389 ± 0.0032 Coverage and 3.5304 ± 0.0025 QUALITY per image.

Target-domain fitting raises all four of these metrics over frozen transfer at each budget. Core Coverage improves by 0.0142 ([0.0110, 0.0175]) and QUALITY by 0.0234 ([0.0189, 0.0281]). Core AP reaches 51.247% and AR100 reaches 64.283%, compared with 50.984% and 63.479% for the frozen transfer. This recovery retains the original evaluation weights, so the improvement is not attributable to a change in the QUALITY definition.

Against Score, COCO-fitted increases core QUALITY by 0.0056 per image (95% CI [0.0030, 0.0083]) but reduces Coverage by 0.0040 ([−0.0063, −0.0017]). Core AP and AR100 remain lower by 0.011 and 0.078 points. The QUALITY gain is approximately 0.159% relative to Score, indicating a small weighted-utility improvement rather than a broad increase in detection quality. At $\overline{K} = 30$ and 40, the Coverage differences are +0.00013 and zero per image, while the QUALITY gains narrow to 0.00166 and 0.00047. Thus the core-budget trade-off does not imply a Coverage deficit at every budget. Across the core budgets the COCO-fitted three-model spread is 2.6530 ± 0.0007 Coverage and 3.5538 ± 0.0008 QUALITY per image.

Post-hoc analysis of frozen transfer finds that images without valid Road8 objects receive 0.7788 more slots on average than under Score, while images with valid objects lose 0.4938 slots. This allocation pattern accompanies the transfer loss; target-domain fitting improves the measured results without changing the candidate stream or budget.

**TABLE II.** COCO2017 Road8 results on all 5,000 validation images. Coverage and QUALITY are per image; AP, AP50, AP75, and AR100 are percentages. Each learned route averages three independently evaluated models, and the detector and evaluation weights are shared; core-budget seed standard deviations are given in the text.

| $\overline{K}$ | Method | Coverage | QUALITY | AP | AP50 | AP75 | AR100 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 10 | Fixed-K | 2.1398 | 2.9797 | 45.462 | 61.170 | 49.186 | 52.065 |
| 10 | Score | 2.5884 | 3.4597 | 50.841 | 70.947 | 54.360 | 62.323 |
| 10 | LC-v1 frozen | 2.5661 | 3.4378 | 50.349 | 70.135 | 53.858 | 61.072 |
| 10 | COCO-fitted | 2.5850 | 3.4694 | 50.828 | 70.755 | 54.436 | 62.316 |
| 15 | Fixed-K | 2.3784 | 3.2395 | 48.174 | 65.872 | 51.883 | 56.868 |
| 15 | Score | 2.6782 | 3.5736 | 51.407 | 71.758 | 54.969 | 64.982 |
| 15 | LC-v1 frozen | 2.6623 | 3.5583 | 51.207 | 71.474 | 54.715 | 64.185 |
| 15 | COCO-fitted | 2.6731 | 3.5781 | 51.381 | 71.689 | 54.971 | 64.770 |
| 20 | Fixed-K | 2.5012 | 3.3748 | 49.672 | 68.623 | 53.381 | 59.815 |
| 20 | Score | 2.7044 | 3.6112 | 51.525 | 71.911 | 55.137 | 65.779 |
| 20 | LC-v1 frozen | 2.6881 | 3.5952 | 51.397 | 71.736 | 54.967 | 65.179 |
| 20 | COCO-fitted | 2.7009 | 3.6140 | 51.531 | 71.905 | 55.149 | 65.764 |
| 30 | Fixed-K | 2.6156 | 3.5096 | 50.858 | 70.675 | 54.513 | 63.054 |
| 30 | Score | 2.7106 | 3.6269 | 51.556 | 71.934 | 55.182 | 66.029 |
| 30 | LC-v1 frozen | 2.7073 | 3.6220 | 51.526 | 71.896 | 55.155 | 65.831 |
| 30 | COCO-fitted | 2.7107 | 3.6285 | 51.557 | 71.934 | 55.187 | 66.038 |
| 40 | Fixed-K | 2.6752 | 3.5848 | 51.320 | 71.422 | 54.992 | 64.914 |
| 40 | Score | 2.7118 | 3.6296 | 51.557 | 71.943 | 55.188 | 66.058 |
| 40 | LC-v1 frozen | 2.7109 | 3.6280 | 51.553 | 71.946 | 55.179 | 66.005 |
| 40 | COCO-fitted | 2.7118 | 3.6301 | 51.558 | 71.944 | 55.188 | 66.066 |

**YOLOv8n instantiation.** A separate allocator is fitted to YOLOv8n [32], using 640 × 640 letterboxed inputs and a fixed pre-NMS Top100 stream. Each of 8,400 prediction locations contributes eight Road8 class hypotheses. Its native vector consists of four box coordinates and 80 class scores; fitting-only PCA supplies 32 components for the 90-dimensional input. The branch uses new role assignments, groups, and class weights, and refits PCA, scaling, the predictor, and temperature. Its three models use AdamW with learning rate $10^{-3}$, weight decay $10^{-4}$, batches of 4,096 positions, and a maximum of 30 epochs.

Table III reports gains over both Fixed-K and Score at every budget. The core differences from Score are 0.7370 Coverage/image and 0.4850 QUALITY/image, with 95% intervals of [0.7070, 0.7670] and [0.4564, 0.5133]. AP and AR100 increase by 0.717 and 3.097 points. The largest relative Coverage improvement is 36.5% at $\overline{K} = 10$; the largest absolute improvement is 0.8030 per image at $\overline{K} = 15$. This extends the evaluation to a dense candidate source and a separately fitted detector-specific representation.

A reference applies class-wise NMS at IoU 0.70 to the frozen prefixes, without refilling discarded slots. At $\overline{K} = 10$, M11 seed 830101 achieves AP of 7.56%, compared with 5.64% for Score. The corresponding post-NMS output counts are 2.9950 and 2.1055 per image. The reference preserves the AP advantage after suppression, with the shared budget defined before NMS.

**TABLE III.** YOLOv8n on BDD DEV with fixed pre-NMS Top100 candidates. Cov. and Q. are per image; AP/AR100 are percentages. Each M11 entry averages three independently trained and separately evaluated models; per-run values for this branch are not reproduced in this manuscript.

| $\overline{K}$ | Method | Cov. | Q. | AP | AR100 |
|---:|---|---:|---:|---:|---:|
| 10 | Fixed-K | 1.8600 | 1.2352 | 2.05 | 5.84 |
| 10 | Score | 1.7995 | 1.2534 | 2.04 | 5.99 |
| 10 | M11 | 2.4560 | 1.7447 | 2.83 | 8.84 |
| 15 | Fixed-K | 2.4460 | 1.5885 | 2.28 | 7.35 |
| 15 | Score | 2.3920 | 1.6505 | 2.28 | 7.58 |
| 15 | M11 | 3.1950 | 2.1757 | 3.06 | 11.03 |
| 20 | Fixed-K | 2.9595 | 1.8929 | 2.44 | 8.71 |
| 20 | Score | 3.0200 | 2.0371 | 2.51 | 9.31 |
| 20 | M11 | 3.7715 | 2.4756 | 3.10 | 12.30 |
| 30 | Fixed-K | 3.8135 | 2.3698 | 2.73 | 10.95 |
| 30 | Score | 4.1065 | 2.6450 | 2.88 | 12.41 |
| 30 | M11 | 4.5970 | 2.8584 | 3.11 | 13.73 |
| 40 | Fixed-K | 4.5745 | 2.7891 | 2.95 | 12.95 |
| 40 | Score | 4.8450 | 2.9832 | 3.04 | 14.07 |
| 40 | M11 | 5.0363 | 3.0482 | 3.11 | 14.38 |

**Faster R-CNN instantiation.** A two-stage Faster R-CNN [22] with a ResNet-18 backbone and feature pyramid network (FPN) [33] is also evaluated. The backbone is initialized from ImageNet weights. Detector training uses 9,000 BDD images assigned to fitting and calibration; the other 1,000 early-stopping images select the detector checkpoint. Inputs preserve their aspect ratio with the longer side resized to 640 pixels. After detector fitting, its weights are held fixed for allocation. The Top100 records are retained after RoI classification, class-specific box regression, clipping, and background removal, before final RoI score filtering and NMS. Scores are class probabilities from a nine-way softmax including background.

The allocator uses 1,024-dimensional RoI box-head features projected to 32 components and the eight foreground logits within the 90-dimensional feature schema. PCA, scaling, class weights, the predictor, and temperature are fitted for this branch using the allocator's 8,000/1,000/1,000 roles and three independent seeds, 730101–730103.

Table IV shows that M11 improves all four reported metrics over both Score and Fixed-K at every budget. Across the core budgets, the gains over Score are 1.1582 Coverage/image and 0.7588 QUALITY/image, with 95% confidence intervals of [1.1129, 1.2052] and [0.7172, 0.8033]. AP and AR100 increase by 1.973 and 4.702 percentage points. The corresponding M11 core means with seed spread are 3.8233 ± 0.0029 Coverage and 2.2976 ± 0.0014 QUALITY per image. Against Fixed-K, the improvements are 0.9167 Coverage/image, 0.6234 QUALITY/image, 1.619 AP points, and 3.740 AR100 points.

Score has lower Coverage than Fixed-K at budgets 10–30, whereas learned allocation improves on both. Thus variable quotas alone do not explain the gain in this branch. The largest Coverage improvement over Score occurs at $\overline{K} = 20$ (1.4163 objects per image), then narrows to 0.5845 at $\overline{K} = 40$.

**TABLE IV.** Faster R-CNN (ResNet-18-FPN) on BDD DEV, pre-NMS Top100. Cov./Q. are per image; AP/AR100 are percentages. M11 averages three models with seeds 730101–730103; the core-budget seed standard deviations are given in the text.

| $\overline{K}$ | Method | Cov. | Q. | AP | AR100 |
|---:|---|---:|---:|---:|---:|
| 10 | Fixed-K | 2.3635 | 1.3602 | 1.63 | 3.65 |
| 10 | Score | 2.2290 | 1.2824 | 1.40 | 3.13 |
| 10 | M11 | 3.0815 | 1.8826 | 3.32 | 7.01 |
| 15 | Fixed-K | 2.9120 | 1.6735 | 1.90 | 4.80 |
| 15 | Score | 2.6675 | 1.5453 | 1.62 | 3.93 |
| 15 | M11 | 3.8732 | 2.3402 | 3.69 | 8.89 |
| 20 | Fixed-K | 3.4445 | 1.9886 | 2.31 | 6.15 |
| 20 | Score | 3.0990 | 1.7885 | 1.76 | 4.65 |
| 20 | M11 | 4.5153 | 2.6699 | 3.68 | 9.91 |
| 30 | Fixed-K | 4.3610 | 2.4962 | 2.66 | 8.01 |
| 30 | Score | 4.0770 | 2.3678 | 2.28 | 6.78 |
| 30 | M11 | 5.3672 | 3.0876 | 3.60 | 10.99 |
| 40 | Fixed-K | 5.1520 | 2.9396 | 3.05 | 9.91 |
| 40 | Score | 5.2180 | 3.0093 | 3.07 | 9.73 |
| 40 | M11 | 5.8025 | 3.2893 | 3.45 | 11.52 |

## D. Ablation and Grouping Robustness

**Utility aggregation.** Table V compares three alternative aggregations of the same ten-output predictor with full M11. M00 uses the IoU-0.50 probability, M10 adds class weighting, and M01 averages all thresholds without class weights; the bit meanings follow the naming defined in Section III-C. M11 combines weighting and threshold averaging. It obtains the highest QUALITY, AP, and AR100 among these four policies, while M00 retains more Coverage. For example, M00 attains 8.2541 Coverage/image versus 8.1163 for M11, but lower QUALITY (4.7957 versus 4.8665). The aggregation choices select a different coverage–quality trade-off from the same predictions.

This difference is systematic rather than incidental: Coverage counts matches at IoU 0.50 only, whereas QUALITY averages ten localization thresholds. M00 therefore optimizes the reported Coverage metric directly and gains +0.2313 over the Score baseline of 8.0228, while M11 gains +0.0935 on that metric but improves QUALITY, AP, and AR100 (Table V). M11 is retained as the reference configuration because the allocator studied here is intended for interfaces in which a retained record must remain useful at stricter localization tolerances, as in downstream association or grouping. Optimizing a single threshold would also make the allocation sensitive to the choice of that threshold. M00 is reported as a co-equal configuration of the same predictor rather than an inferior variant: an interface judged only by IoU-0.50 coverage counting should prefer M00, and an interface judged by localization-quality-sensitive records should prefer M11. Both are available from one trained model by changing the aggregation rule alone, so the choice is a deployment decision rather than a retraining decision.

**Prefix supervision.** The matchability control retains the full inputs, architecture, and training recipe, but labels a candidate positive whenever it can match a same-class object at the given threshold. It does not account for objects already covered by the prefix. Prefix-increment supervision raises core Coverage by 0.1867 and QUALITY by 0.0619 relative to this control; the 95% intervals are [0.1702, 0.2038] and [0.0533, 0.0704]. This controlled comparison supports measuring the contribution of the extension rather than only the matchability of the added candidate.

**Feature inputs.** No embedding masks the 32 projected components and their native-vector norm; No prefix statistics masks the 12 explicit preceding-prefix features. Both are retrained with the remaining inputs and network dimensions unchanged. Removing the embedding block lowers core Coverage by 0.0218, with a 95% interval for the full-model advantage of [0.0111, 0.0326]. The prefix-statistics effect is smaller and unresolved: the difference is 0.0042 with interval [−0.0050, 0.0133]. Rank and image summaries remain available in this control, so it tests the incremental value of the explicit prefix block.

**Exact allocation.** Next-slot greedy reuses the M11 predictions, starts every image at $L = 5$, and repeatedly selects the highest-valued currently available extension. Exact DP raises core Coverage by 0.5616 and QUALITY by 0.3713, with intervals of [0.5099, 0.6128] and [0.3340, 0.4057]. It also attains a higher predicted objective in all 750 group–seed–budget conditions. The solver therefore changes retained-object quality even when its input predictions are unchanged.

The two ablations also quantify their relative contribution, and the ordering is worth stating plainly. With the learned predictions held fixed, replacing exact DP by next-slot greedy reduces core Coverage from 8.1163 to 7.5547, which is 0.5616 below the Score baseline's core value of 8.0228. Replacing prefix-increment supervision by the matchability target, with the exact solver retained, reduces it to 7.9296, also slightly below the baseline. The solver therefore contributes about three times the supervision target: +0.5616 versus +0.1867 core Coverage. Neither component is sufficient alone at this budget, and the failure of the greedy solver on the same predictions is direct evidence that joint prefix selection, rather than a stronger value estimate, is what separates the proposed configuration from score-based allocation.

**Grouping.** Allocation is repeated on the same DEV images using three deterministic permutations and group sizes $n \in \{10, 20, 40, 80\}$. Across the twelve configurations, core gains over Score range from 0.0783 to 0.1039 Coverage/image and from 0.1011 to 0.1258 QUALITY/image; all corresponding 95% interval lower bounds are positive. Group membership changes the gain magnitude but not its direction in this study. At $n = 1$, the exact budget fixes each prefix and the methods coincide.

**TABLE V.** BDD DEV ablations. All entries are absolute core-budget means with common evaluation weights; the ± value is the standard deviation across the three independently trained and separately evaluated models, and no spread is defined for the deterministic baseline. Cov./Q. are per image; AP/AR100 are percentages.

| Policy / variant | Cov. | Q. | AP | AR100 |
|---|---:|---:|---:|---:|
| M11 (full) | 8.1163 ± 0.0123 | 4.8665 ± 0.0011 | 17.03 ± 0.02 | 25.27 ± 0.11 |
| *(Alternative aggregation, same trained models)* | | | | |
| M00: IoU 0.50 | 8.2541 ± 0.0014 | 4.7957 ± 0.0038 | 16.54 ± 0.05 | 23.46 ± 0.12 |
| M10: weighted 0.50 | 8.1763 ± 0.0026 | 4.8515 ± 0.0025 | 16.93 ± 0.02 | 24.80 ± 0.07 |
| M01: multi-IoU | 8.2192 ± 0.0077 | 4.8230 ± 0.0027 | 16.70 ± 0.04 | 24.04 ± 0.16 |
| *(Retrained feature and supervision controls)* | | | | |
| No embedding | 8.0945 ± 0.0071 | 4.8512 ± 0.0011 | 17.00 ± 0.00 | 25.20 ± 0.04 |
| No prefix statistics | 8.1121 ± 0.0173 | 4.8615 ± 0.0008 | 16.99 ± 0.01 | 25.11 ± 0.06 |
| Matchability target | 7.9296 ± 0.0116 | 4.8046 ± 0.0033 | 16.96 ± 0.01 | 25.14 ± 0.04 |
| *(Alternative solver, same predictions)* | | | | |
| Next-slot greedy | 7.5547 ± 0.0030 | 4.4952 ± 0.0004 | 16.01 ± 0.01 | 22.71 ± 0.01 |
| Score baseline (deterministic) | 8.0228 | 4.7480 | 16.56 | 23.83 |

## E. Computational Cost

Profiling starts with candidate/native tensors resident in host memory and ends with quotas and selected record IDs. It includes feature construction, PCA/scaling, one predictor, temperature scaling, utility aggregation, and allocation. Detector execution, data loading, and evaluation are outside this boundary. Measurements use an RTX 5080 Laptop GPU with PyTorch 2.11.0 and CUDA 12.8. The group-size profile uses 400 fitting images, three warm-up runs and ten measurements per group, with $\overline{K} = 15$ and seed 530101.

At $n = 40$, M11 requires a median of 140.76 ms per group, compared with 7.92 ms for Score (Table VI). The allocation stage is therefore 17.8 times slower than the score baseline for the same output decision, or an amortized 3.519 ms/image on top of the detector it does not modify.

For the core budgets, that cost buys +0.0792 covered objects per image on TEST, which is the only comparable-unit figure available: approximately 44 ms of allocator time per additional covered object per image at the reported median. No downstream measurement — end-to-end latency, tracking quality, or task success — was performed, so the trade-off cannot yet be expressed in application-level terms, and the ratio above should be read as the price of the retained-detection gain rather than as a deployment recommendation.

The cost is also not fixed: in a separate matched ablation profile on five 40-image groups, full M11 takes 136.56 ms, No embedding 130.50 ms, No prefix statistics 63.51 ms, and Next-slot greedy 113.99 ms. These correspond to reductions of 4.44%, 53.49%, and 16.53% within that profile. Removing prefix statistics offers the largest measured saving, whereas the greedy solver incurs the larger quality loss in Table V. The unresolved accuracy difference for the prefix block therefore remains relevant to this cost trade-off. A cheaper representation of the prefix statistics is the most plausible next step toward reducing the 17.8× ratio.

**TABLE VI.** Allocation-stage cost at $\overline{K} = 15$, $n = 40$. Median and p95 are milliseconds per group.

| Method | Median | p95 | ms/image |
|---|---:|---:|---:|
| Score | 7.92 | 10.49 | 0.198 |
| Platt | 32.35 | 39.97 | 0.809 |
| Weighted Platt | 32.43 | 39.28 | 0.811 |
| M11 | 140.76 | 207.34 | 3.519 |

# V. Discussion

**Solver and target: which part carries the gain.** Two ablations clarify why prefix allocation benefits from more than candidate reliability. Replacing prefix-increment supervision with matchability reduces retained-object quality, while substituting next-slot greedy for DP loses quality with predictions unchanged (Table V). The magnitudes, however, are not equal, and the evidence assigns more of the gain to the solver: with identical predictions, exact DP adds 0.5616 core Coverage over next-slot greedy, while the prefix-increment target adds 0.1867 over matchability under the same solver. The greedy solver applied to the learned predictions even falls 0.4681 below the Score baseline, and the matchability target under the exact solver falls 0.0932 below it. The roles are therefore complementary but asymmetric: the target values an extension relative to the existing prefix, and the solver is what converts those values into a joint selection that respects prefix ordering constraints. The design keeps both, and reports the greedy-versus-DP comparison as a primary result rather than a secondary ablation, because it demonstrates that the allocation decision itself — not only the quality of the learned values — is a substantive part of the contribution. The unresolved effect of explicit prefix features is a separate issue; their removal retains prefix-based targets, candidate rank, and image context.

**Utility preferences and domain transfer.** Class weighting and threshold averaging change what the allocator values. The aggregation study shows that improving QUALITY can sacrifice micro-coverage even with the same predictor, and that the IoU-0.50 aggregation M00 is the stronger choice if the downstream interface is scored by single-threshold coverage counting. COCO adaptation further separates fitting the estimator from selecting the objective: refitting improves on frozen transfer, but retains the LC-v1 class weights and a core-budget coverage–quality trade-off against Score (Table II). This suggests treating utility selection and target-domain fitting as separate decisions when adapting the method. The current weights encode class-frequency correction; choosing weights for downstream task priorities would require an application-specific basis.

**Candidate sources and budget placement.** The YOLOv8n and Faster R-CNN studies extend prefix allocation to dense and RoI-based candidate streams with separately fitted features and predictors (Tables III and IV). M11 improves on both fixed quotas and score-based allocation in each branch. Budget placement remains important: in the YOLO post-selection NMS reference, equal pre-NMS budgets yield different survivor counts. An equal-final-output comparison would therefore need to enforce its budget after suppression. Constructing prefixes from post-suppression candidates and checking their feasible capacities would test how this change in candidate population affects allocation.

**Cost and system integration.** The timing results favor investigating cheaper features before replacing the solver. Omitting prefix statistics roughly halves the matched processing cost, whereas the cheaper greedy solver incurs a larger quality loss (Section IV-E). The uncertain quality difference for the prefix block warrants testing a less expensive representation rather than immediately removing it. Moving beyond offline groups would also require defining input arrivals, a waiting-time limit, and the downstream capacity. A system-level evaluation should measure allocator overhead, downstream workload, and task quality together to establish whether the retained-detection gains justify the extra computation.

# VI. Conclusion

This paper formulated output-budget allocation as the joint selection of score-ordered detection prefixes under an exact group budget. With the detector and candidate streams fixed, the allocation module determines how many records each image retains. The proposed method learns the additional matching coverage obtained by extending a prefix at multiple localization thresholds, accounting for objects already covered by earlier records. Calibrated predictions are combined with class weights to construct image-level utility curves, and exact dynamic programming selects prefix lengths jointly across images. This connects supervision to the allocation decision and accounts for the preceding records required to reach a valuable candidate.

Experiments on the BDD100K-derived confirmation set show core-budget gains of 0.0792 Coverage/image and 0.1100 QUALITY/image over score-based allocation. Separately fitted allocators for YOLOv8n and Faster R-CNN also improve coverage, AP, and AR100 over fixed quotas and score-based allocation on their respective fixed pre-NMS candidate streams. Controlled comparisons identify prefix-increment supervision and exact optimization as important contributors, with the solver carrying the larger measured effect, while COCO experiments expose the limits of frozen transfer. Target-domain fitting improves on the transferred estimator, but retains a core-budget coverage deficit and a small positive QUALITY difference against the score baseline. Taken together, these results support learning the value of prefix extensions as a means of improving retained detections under shared capacity constraints. They also distinguish the benefits of the allocation method from the transferability of a fitted estimator. Further work will investigate lower-cost feature construction and evaluate whether the gains in retained detection quality justify the allocation overhead within downstream perception pipelines.

# Appendix A. Procedural Disclosure on TEST Annotation Access

One procedural deviation affected the confirmation protocol and is disclosed in full here rather than in a footnote.

**What triggered the exposure.** During the identity-recovery stage that binds the LC-v1 split roles, a retrieval operation was issued over project storage to recover image and split identities. The command returned more file content than the identity fields it was intended to read, and a truncated excerpt of TEST annotations was displayed in the process output.

**Scope of the excerpt.** The deviation is recorded as an early retrieval-scope deviation, and the affected non-identity fields were not used for scientific analysis, split construction, model development, or delivered exclusion identities. The exposure was limited to a truncated annotation excerpt. It did not include per-image allocations, candidate records, model outputs, or metric values, and it was not produced by, or fed into, any allocation or selection step.

**Controls that prevented use.** The allocation path consumes no annotations: its inputs are candidate records, detector-native state, and image-context features computed from the frozen Top100 stream (Section III). The retrieval operation is not part of the selection entry point, and it changed neither the frozen method nor the recorded configuration nor the split definitions. The confirmation run re-evaluates the frozen configuration and recomputes all reported metrics from the frozen selection outputs; no configuration or hyperparameter was changed after the exposure, and no seed, budget, or aggregation was selected on the basis of the excerpt. This disposition is recorded as PROTOCOL_QUALIFIED_IDENTITY_RECOVERY, not a scientific-use event, and no TEST-driven decision is reported from it.

**Inspectable artifacts.** The disposition of the event and the frozen split construction are recorded in the data-protocol document of the LC-v1 release package (`docs/DATA_PROTOCOL_PUBLIC.md`), where the frozen upstream decision is bound by the P0 decision SHA carried in that release's manifest. The frozen allocator configuration evaluated here is the reference asset set of the project handoff package `LC_ALLOC_M11_HANDOFF_v1` (version 1.0.0, created 28 September 2026, reference replay PASS), whose `provenance/SHA256SUMS.csv` lists the SHA256 of every file: the three trained seed models, the temperature parameters, PCA, scaler, class weights, feature schema, and candidate freeze, together with the group manifests that define the allocation groups for each dataset role. Recomputing these hashes verifies that the assets evaluated here are the frozen assets that the package distributes, so the configuration behind the reported TEST metrics can be identified independently of the execution log.

# Acknowledgment

Funding: This research was partially supported by the National Natural Science Foundation of China (NSFC, Grant No. 42671235). XJTLU provided experimental facilities and contributed to data collection, analysis and interpretation, alongside manuscript preparation. The NSFC additionally facilitated research networking opportunities to support further investigations into advanced simulation.

# References

[1] Y. Zhao, W. Lv, S. Xu, J. Wei, G. Wang, Q. Dang, Y. Liu, and J. Chen, "DETRs beat YOLOs on real-time object detection," in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024, pp. 16965–16974.

[2] W. Lv, Y. Zhao, Q. Chang, K. Huang, G. Wang, and Y. Liu, "RT-DETRv2: Improved baseline with bag-of-freebies for real-time detection transformer," arXiv preprint arXiv:2407.17140, 2024.

[3] N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, and S. Zagoruyko, "End-to-end object detection with transformers," in Proc. European Conf. Comput. Vis. (ECCV), 2020, pp. 213–229.

[4] X. Zhu, W. Su, L. Lu, B. Li, X. Wang, and J. Dai, "Deformable DETR: Deformable transformers for end-to-end object detection," in Proc. Int. Conf. Learn. Representations (ICLR), 2021.

[5] H. Zhang, F. Li, S. Liu, L. Zhang, H. Su, J. Zhu, L. Ni, and H.-Y. Shum, "DINO: DETR with improved denoising anchor boxes for end-to-end object detection," in Proc. Int. Conf. Learn. Representations (ICLR), 2023.

[6] B. Roh, J. Shin, W. Shin, and S. Kim, "Sparse DETR: Efficient end-to-end object detection with learnable sparsity," in Proc. Int. Conf. Learn. Representations (ICLR), 2022.

[7] Z. Yao, J. Ai, B. Li, and C. Zhang, "Efficient DETR: Improving end-to-end object detector with dense prior," arXiv preprint arXiv:2104.01318, 2021.

[8] W. Liu, D. Anguelov, D. Erhan, C. Szegedy, S. Reed, C.-Y. Fu, and A. C. Berg, "SSD: Single shot MultiBox detector," in Proc. European Conf. Comput. Vis. (ECCV), 2016, pp. 21–37.

[9] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, "You only look once: Unified, real-time object detection," in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2016, pp. 779–788.

[10] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, "Focal loss for dense object detection," in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), 2017, pp. 2999–3007.

[11] Z. Cai and N. Vasconcelos, "Cascade R-CNN: Delving into high quality object detection," in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2018, pp. 6154–6162.

[12] Z. Lin, Y. Wang, J. Zhang, and X. Chu, "DynamicDet: A unified dynamic architecture for object detection," in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2023, pp. 6282–6291.

[13] Y. Han, G. Huang, S. Song, L. Yang, H. Wang, and Y. Wang, "Dynamic neural networks: A survey," IEEE Trans. Pattern Anal. Mach. Intell., vol. 44, no. 11, pp. 7436–7456, 2022.

[14] L. Yang, Y. Han, X. Chen, S. Song, J. Dai, and G. Huang, "Resolution adaptive networks for efficient inference," in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2020, pp. 2366–2375.

[15] Z. He and Y. Li, "Determining the proper number of proposals for individual images," IET Computer Vision, 2024.

[16] S. Kuzucu, K. Oksuz, J. Sadeghi, and P. K. Dokania, "On calibration of object detectors: Pitfalls, evaluation and baselines," in Proc. European Conf. Comput. Vis. (ECCV), 2024.

[17] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in Proc. Int. Conf. Mach. Learn. (ICML), 2017, pp. 1321–1330.

[18] F. Yu, H. Chen, X. Wang, W. Xian, Y. Chen, F. Liu, V. Madhavan, and T. Darrell, "BDD100K: A diverse driving dataset for heterogeneous multitask learning," in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2020, pp. 2636–2645.

[19] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollár, and C. L. Zitnick, "Microsoft COCO: Common objects in context," in Proc. European Conf. Comput. Vis. (ECCV), 2014, pp. 740–755.

[20] X. Hou, M. Liu, S. Zhang, P. Wei, and B. Chen, "Salience DETR: Enhancing detection transformer with hierarchical salience filtering refinement," in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024, pp. 17574–17583.

[21] Z. Kang, J. Zhuang, K. Mo, Q. Chen, R. Liu, and Y. Zhang, "PaQ-DETR: Learning pattern and quality-aware dynamic queries for object detection," in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2026, pp. 25504–25513.

[22] S. Ren, K. He, R. Girshick, and J. Sun, "Faster R-CNN: Towards real-time object detection with region proposal networks," in Advances in Neural Information Processing Systems, vol. 28, 2015.

[23] W. Guan, X. Zhou, G. Li, and Y. Zou, "Selecting optimal proposal number for image-based object detection," in Proc. IEEE Int. Conf. Acoust., Speech Signal Process. (ICASSP), 2019, pp. 3797–3801.

[24] N. Bodla, B. Singh, R. Chellappa, and L. S. Davis, "Soft-NMS—improving object detection with one line of code," in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), 2017, pp. 5561–5569.

[25] J. Hosang, R. Benenson, and B. Schiele, "Learning non-maximum suppression," in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2017, pp. 6469–6477.

[26] K. Oksuz, T. Joy, and P. K. Dokania, "Towards building self-aware object detectors via reliable uncertainty quantification and calibration," in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2023, pp. 9263–9274.

[27] Y. Hu, S. Fang, Z. Lei, Y. Zhong, and S. Chen, "Where2comm: Communication-efficient collaborative perception via spatial confidence maps," in Advances in Neural Information Processing Systems, vol. 35, 2022.

[28] G. Chen, C. Zhang, and X. Zhao, "WhisperNet: A scalable solution for bandwidth-efficient collaboration," in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2026, pp. 32154–32163.

[29] R. Xu, H. Xiang, Z. Tu, X. Xia, M.-H. Yang, and J. Ma, "V2X-ViT: Vehicle-to-everything cooperative perception with vision transformer," in Proc. European Conf. Comput. Vis. (ECCV), 2022, pp. 107–124.

[30] Y.-C. Liu, J. Tian, N. Glaser, and Z. Kira, "When2com: Multi-agent perception via communication graph grouping," in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2020, pp. 4105–4114.

[31] Y. Li, S. Ren, P. Wu, S. Chen, C. Feng, and W. Zhang, "Learning distilled collaboration graph for multi-agent perception," in Advances in Neural Information Processing Systems, vol. 34, 2021.

[32] G. Jocher, A. Chaurasia, and J. Qiu, "Ultralytics YOLOv8," Software, 2023. [Online]. Available: https://github.com/ultralytics/ultralytics

[33] T.-Y. Lin, P. Dollár, R. Girshick, K. He, B. Hariharan, and S. Belongie, "Feature pyramid networks for object detection," in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2017, pp. 2117–2125.
