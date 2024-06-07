# Fairness Checker

This Python module fairness_checker provides a set of methods to evaluate the fairness of a predictive model's outcomes across different demographic groups represented in a CSV file or given a model.

## Dependencies

* Python >= 3.6

## Installation

```bash
pip3 install fairness-checker
```

## Usage

For example:

```python3
from fairness_checker import fairness_csv_checker
c = fairness_csv_checker("compas-scores-two-years.csv")
c.demographic_parity(0.2, lambda row: row['sex'] == 'Male', lambda row: row['score_text'] in {'Medium', 'High'})
```

Output:

```txt
demographic parity
fair: 0.04 < 0.2
```

Note the function signature of `demographic_parity`:
```
demographic_parity(ratio: float,
                   privileged_predicate: Callable[[csv_row], bool],
                   positive_predicate: Callable[[T], bool]) -> bool:
```
Here the `privileged_predicate` is
```
lambda row: row['sex'] == 'Male'
```
meaning the privileged group is the male group, and the `positive_predicate` is
```
lambda row: row['score_text'] in {'Medium', 'High'}
```
meaning the row is positive if the score is categorized as medium or high.
