# Dataset Note

## Dataset

This project uses the expanded PROMISE requirements-classification dataset (PROMISE_exp) containing 969 requirements.

The original dataset fields used by the implementation are:

- `ProjectID`
- `RequirementText`
- `_class_`

The binary transformation used in this study is:

- `F` -> Functional Requirement (FR, label 0)
- every other original class -> Non-Functional Requirement (NFR, label 1)

This produces 444 FR and 525 NFR instances.

## Provenance

The PROMISE_exp data used in this study are hosted in the public repository:

https://github.com/AleksandarMitrevski/se-requirements-classification

The expansion of the PROMISE corpus is described by:

Lima, M., Valle, V., Costa, E., Lira, F., & Gadelha, B. (2019). *Software Engineering Repositories: Expanding the PROMISE Database*. Proceedings of the XXXIII Brazilian Symposium on Software Engineering (SBES 2019), 427-436. DOI: 10.1145/3350768.3350776.

The PROMISE_exp ARFF distribution contains a Creative Commons Attribution-Share Alike 3.0 notice for the PROMISE dataset and requests acknowledgment of the PROMISE repository when publishing work based on the data.

Before public redistribution, preserve the applicable attribution and license notice and verify that the exact material being redistributed is covered by the stated terms.

## Reproduction

The repository expects the file:

```text
data/requirements_dataset.csv
```

with the columns listed above.
