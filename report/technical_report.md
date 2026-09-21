# Technical Report
## Explainable and Verifiable ML for Requirements Classification

**Research Grant:** IT137-26-330  
**Project:** Digital Person – Safely Integrating Machine Learning into Clinical Decision Support Systems  
**Institution:** University of Coimbra  
**Author:** Amir Hozhabri  
**Date:** September 2026

---

## 1. Introduction

### 1.1 Motivation

This implementation study examines whether a small and interpretable requirements classifier can be augmented with confidence estimation, feature-space counterfactual explanations, and formal consistency checking in one reproducible pipeline.

The study is inspired by the VerifAI-LORE architecture described by Barbosa et al. (2026). The reference architecture combines classification, confidence indication, local explanations, counterfactual rules, and deductive verification of explanation consistency. The present work does not reproduce the complete distributed architecture or its medical case study. Instead, it adapts the methodological core to a simpler software-requirements classification problem.

### 1.2 Research Question

**Can a small and interpretable requirements classifier be augmented with confidence estimation, feature-space counterfactual explanations, and formal consistency checking in one reproducible pipeline?**

The study is intentionally non-clinical. It does not establish clinical validity, medical safety, or deployment readiness.

---

## 2. Methodological Basis

The pipeline follows four methodological stages that correspond conceptually to the work-plan components of IT137-26-330:

| IT137-26-330 component | Mini-project implementation |
|---|---|
| Knowledge Base | Interpretable feature extraction from requirements text |
| ML-based Recommendations | Decision Tree classification and probability-based confidence |
| Formal Assurances | Exact Decision Tree SMT encoding and Z3 consistency checks |
| Evaluation | Classification, calibration, counterfactual, and verification metrics |

The reference architecture motivates the separation between prediction, explanation, and formal checking. In particular, a classification result can be accompanied by a confidence indication and a local counterfactual explanation, after which a theorem prover can check whether the stated explanation is consistent with the encoded classifier.

The implementation here uses a compact feature representation so that both counterfactual search and logical encoding remain inspectable.

---

## 3. Dataset and Feature Representation

### 3.1 Dataset

The experiment uses the expanded PROMISE requirements-classification dataset (PROMISE_exp), containing **969 requirements**. The public PROMISE_exp distribution contains the original requirement text, project identifier, and class label fields used by this implementation. The expansion of the PROMISE corpus is described by Lima et al. (2019), which reports the PROMISE corpus expansion for software-requirements classification.

The dataset is hosted in the public repository:

https://github.com/AleksandarMitrevski/se-requirements-classification

The PROMISE_exp ARFF distribution includes a Creative Commons Attribution-Share Alike 3.0 notice for the PROMISE dataset and requests acknowledgment of the PROMISE repository when publishing work based on the data. The exact redistribution terms should be preserved when the dataset is shared.

The original labels are:

| Label | Meaning | Count |
|---|---|---:|
| F | Functional Requirement | 444 |
| A | Availability | 31 |
| FT | Fault Tolerance | 18 |
| L | Legal | 15 |
| LF | Look-and-Feel | 49 |
| MN | Maintainability | 24 |
| O | Operability | 77 |
| PE | Performance | 67 |
| PO | Portability | 12 |
| SC | Scalability | 22 |
| SE | Security | 125 |
| US | Usability | 85 |

For the binary experiment, `F` is mapped to **FR (label 0)** and every other class is mapped to **NFR (label 1)**. The resulting distribution is therefore 444 FR and 525 NFR instances.

The implementation expects the local dataset file:

```text
data/requirements_dataset.csv
```

with columns `ProjectID`, `RequirementText`, and `_class_`.

### 3.2 Feature Space

Twelve explicit features are extracted from each requirement:

1. `length_words`
2. `sentence_count`
3. `fr_cue_count`
4. `nfr_cue_count`
5. `negation_count`
6. `has_digit`
7. `has_unit`
8. `has_shall`
9. `has_must`
10. `passive_indicator`
11. `has_list_marker`
12. `uppercase_ratio`

The features were selected to remain small, domain-neutral, and interpretable. They include lexical cues, quantitative/measurability indicators, simple structural properties, and a quality-attribute cue count.

An important boundary is that a counterfactual produced in this study is a counterfactual in **feature space**. It is not automatically a fluent or semantically valid rewritten requirement sentence.

---

## 4. Methodology

### 4.1 Classification

A `DecisionTreeClassifier` was selected because its decision function can be inspected through root-to-leaf conditions. The model configuration was fixed before evaluation:

- maximum depth = 5
- random seed = 42
- class weighting = balanced
- stratified train/test split = 80/20

The resulting split contains 775 training instances and 194 test instances.

### 4.2 Probability-Based Confidence

For each test prediction, confidence is defined as the maximum class probability returned by the Decision Tree:

\[
confidence(x)=\max_c P(c\mid x)
\]

This is a classifier output and is not a formal guarantee.

### 4.3 Local Stability

Local stability is estimated by perturbing valid feature values within the predefined feature domain and measuring how often the model changes its predicted class. The score is defined as:

\[
local\_stability = 1 - \frac{\text{neighbour label flips}}{\text{number of neighbours}}
\]

This is a simplified approximation of local robustness and should not be equated with the search-based confidence verification mechanism of the reference architecture.

### 4.4 Counterfactual Search

For a test instance \(x\) and target class \(t\), the counterfactual module searches the predefined feature space for a valid vector \(x_{cf}\) such that:

\[
f(x_{cf})=t
\]

The search prioritizes minimum feature sparsity first and minimum normalized feature change second. Each generated counterfactual is explicitly checked again against the original Decision Tree.

For the coverage experiment, 30 deterministic test cases were evaluated.

### 4.5 SMT Encoding and Verification

The trained Decision Tree is represented as root-to-leaf logical paths. Each internal node contributes an inequality such as:

```text
feature <= threshold
```

or:

```text
feature > threshold
```

A leaf is represented by the conjunction of its path conditions. A class condition is the disjunction of all leaf paths predicting that class.

The resulting logical representation is checked using Z3.

For an exact counterfactual point, the consistency property is evaluated by asking whether:

\[
R(x) \land \neg Class_t(x)
\]

is satisfiable. `UNSAT` means that no counterexample exists for that exact point within the encoded domain assumptions.

A more general changed-feature rule is also checked. Because only the changed features are fixed and all other features remain unconstrained, this generalized rule can legitimately admit counterexamples.

### 4.6 Encoder Validation

The logical encoding is tested before verification results are interpreted. Three categories of checks are performed:

1. Random valid feature vectors
2. Boundary cases at learned thresholds using valid feature-domain values
3. Leaf coverage and leaf exclusivity

The random encoder test compares the original scikit-learn prediction with the Z3 encoding prediction. A mismatch indicates an encoding or domain-handling error.

---

## 5. Results

All numerical results reported below were obtained from the executed implementation and stored in `results/metrics.json`.

### 5.1 Classification Results

| Metric | Result |
|---|---:|
| Dataset size | 969 |
| Training size | 775 |
| Test size | 194 |
| Accuracy | **0.6753** |
| Macro-F1 | **0.6733** |
| Macro-Precision | 0.6961 |
| Macro-Recall | 0.6863 |
| Decision Tree depth | 5 |
| Decision Tree leaves | 24 |

The confusion matrix is:

```text
[[73, 16],
 [47, 58]]
```

The class order is `FR = 0` and `NFR = 1`.

These results describe the behaviour of the fixed baseline model on the held-out test set. They are not intended as evidence of clinical performance.

### 5.2 Confidence and Calibration

| Metric | Result |
|---|---:|
| Mean confidence | 0.7770 |
| Mean local stability | 0.7964 |
| Expected Calibration Error (ECE) | 0.1017 |
| Brier score | 0.2185 |

The test cases with confidence in the 0.60–0.79 interval had an empirical accuracy of 0.6058, while the 0.80-and-above group had an empirical accuracy of 0.8421. No test predictions fell below 0.60 confidence.

The calibration figure is saved as:

```text
results/figures/calibration_plot.png
```

### 5.3 Counterfactual Results

The counterfactual experiment evaluated 30 selected test cases.

The implementation searches until it finds a target-reaching counterfactual
and then explicitly re-checks the saved counterfactual against the original
Decision Tree. Therefore, the reported validity of 1.0000 should be read as
post-generation verification of all saved counterfactuals, not as an
independent estimate of the search algorithm's failure rate.

| Metric | Result |
|---|---:|
| Selected cases | 30 |
| Counterfactuals generated | 30 |
| Valid counterfactuals | 30 |
| Counterfactual validity | 1.0000 |
| Counterfactual coverage | 1.0000 |
| Average sparsity | 1.0667 |
| Average normalized magnitude | 0.3129 |

All 30 stored counterfactuals were re-evaluated with the original Decision Tree and reached their target class.

Representative examples are stored in:

```text
results/counterfactual_examples.json
```

The examples should be interpreted only in the predefined feature space; changing a feature value does not imply that a corresponding natural-language requirement rewrite would be semantically valid.

#### Representative counterfactual examples

The following examples are taken from the executed counterfactual results. Class labels are `FR = 0` and `NFR = 1`.

**Example 1 - uppercase ratio change**

Original requirement:

> The system shall allow user to confirm the purchase.

- Original prediction: `FR (0)`
- Target prediction: `NFR (1)`
- Changed feature: `uppercase_ratio: 0.0 -> 0.1`
- Sparsity: `1`
- Exact-point Z3 result: `UNSAT`

**Example 2 - digit indicator change**

Original requirement:

> When streaming a movie  the buffering time should take no longer than 10 seconds (plus any latency on the user\92s connection.)

- Original prediction: `NFR (1)`
- Target prediction: `FR (0)`
- Changed feature: `has_digit: 1 -> 0`
- Sparsity: `1`
- Exact-point Z3 result: `UNSAT`

**Example 3 - two-feature change**

Original requirement:

> The product must be designed using Design Patterns and coding best practices.  90% of maintenance software developers are able to integrate new functionality into the product with 2 working days.

- Original prediction: `NFR (1)`
- Target prediction: `FR (0)`
- Changed features: `sentence_count: 2 -> 1` and `has_digit: 1 -> 0`
- Sparsity: `2`
- Exact-point Z3 result: `UNSAT`

For comparison, the generalized changed-feature rule for Example 1 is satisfiable (`SAT`) because it fixes only `uppercase_ratio == 0.1` and leaves the remaining features unconstrained. Z3 returned a concrete witness for that generalized rule. This illustrates the distinction between an exact counterfactual point and a broader rule over feature space.

### 5.4 Formal Verification Results

#### Encoder validation

```text
Random cases checked: 500
Mismatch count: 0
```

The zero mismatch count indicates agreement between the scikit-learn model and the logical encoding on the tested random valid feature vectors.

#### Boundary validation

```text
Boundary cases checked: 42
Boundary mismatches: 0
```

#### Leaf checks

```text
Leaf coverage verified: True
Leaf exclusivity violations: 0
```

#### Exact counterfactual points

```text
Rules checked: 30
UNSAT: 30
SAT: 0
```

For the exact-point property, the solver found no counterexample to the target-class assertion for any of the 30 verified points.

#### Generalized changed-feature rules

```text
UNSAT: 0
SAT: 30
```

These SAT results are not inconsistent with the exact-point results. They arise because the generalized rule fixes only the features changed by the counterfactual and leaves other features unconstrained. The result therefore demonstrates that a point-level counterfactual should not automatically be interpreted as a globally valid rule for every point sharing only the changed-feature conditions.

The median recorded verification time for the exact-point checks was approximately **0.0010 seconds** per rule.

---

## 6. Limitations and Threats to Validity

This study is deliberately small and methodological.

First, software requirements classification is substantially simpler than a clinical decision-support setting. The results therefore cannot be transferred directly to oncology or other clinical applications.

Second, the counterfactual representation is feature-based. It does not guarantee that a corresponding requirement sentence would remain grammatical, realistic, or semantically consistent.

Third, the probability-based confidence score and local-stability score are empirical indicators. They are not formal safety guarantees.

Fourth, SMT verification is only as strong as the logical encoding and the feature-domain assumptions used by the implementation. The encoder validation performed here provides evidence of agreement with the source Decision Tree on the tested domain but does not prove correctness of the dataset, labels, feature extraction, or intended real-world use.

Fifth, the evaluation uses one classifier family and a modest dataset size. Broader generalization requires further experiments with additional classifiers, datasets, feature representations, and domain experts.

Finally, no clinical validation, patient-level assessment, or human-subject evaluation was performed.

---

## 7. Conclusion

The implementation demonstrates a compact pipeline that combines an interpretable requirements classifier with probability-based confidence estimation, local-stability analysis, feature-space counterfactual generation, and SMT-based consistency checking.

On the executed experiment, the Decision Tree achieved 0.6753 test accuracy and 0.6733 Macro-F1. The average confidence was 0.7770 and the ECE was 0.1017. Thirty selected test cases all received valid generated counterfactuals, and all thirty exact counterfactual points passed the corresponding Z3 consistency query with `UNSAT`. The Decision Tree encoding produced zero mismatches in 500 random validation cases and zero mismatches in 42 boundary checks.

The results should be interpreted as evidence about the implemented prototype under its explicit assumptions. They do not establish clinical validity or general formal-verification expertise. The principal methodological contribution of the study is the integrated, reproducible separation of prediction, explanation, and formal consistency checking in a small experimental setting.

---

## References

1. Barbosa, R., Rinzivillo, S., Robin, J., Beretta, A., & Madeira, H. (2026). *A software architecture for verifiable and explainable classification*. Machine Learning, 115, 92. DOI: 10.1007/s10994-026-07006-0.

2. Lima, M., Valle, V., Costa, E., Lira, F., & Gadelha, B. (2019). *Software Engineering Repositories: Expanding the PROMISE Database*. Proceedings of the XXXIII Brazilian Symposium on Software Engineering (SBES 2019), 427–436. DOI: 10.1145/3350768.3350776.

3. Hozhabri, A., Eslaminejad, M., & Mahrouyan, M. (2019). *Chain-based Gateway Nodes Routing for Energy Efficiency in WSN*. International Journal of Engineering and Technology, 11(6S), 101–108. DOI: 10.21817/ijet/2019/v11i6/191106098.

4. Vahabi, S., & Hozhabri, A. (2024). *Automatic Use Case Classification Based on Topic Grouping for Requirements Engineering*. Innovations in Systems and Software Engineering, 20, 85–96. DOI: 10.1007/s11334-023-00535-0.

5. Vahabi, S., Mojab, S. P., Hozhabri, A., & Daneshvar, A. (2023). *Reinforcement Learning Movement Path for Multiple Mobile Sinks in Wireless Sensor Networks*. International Journal of Communication Systems, 36, e5402. DOI: 10.1002/dac.5402.

## Reproducibility Files

The repository should contain:

```text
data/requirements_dataset.csv
src/
results/metrics.json
results/counterfactual_examples.json
results/verification_cases.json
results/figures/calibration_plot.png
tests/
requirements.txt
README.md
```

The full pipeline can be executed from the project root with:

```bash
python src/pipeline.py
```
