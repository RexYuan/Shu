# Fairness Checker

This Python module fairness_checker provides a set of methods to evaluate the fairness of a predictive model's outcomes across different demographic groups represented in a CSV file or given a model.

## Dependencies

* Python >= 3.6

## Installation

```bash
pip3 install fairness-checker
```

## Usage

First set up the checker using a benchmark dataset:

```python3
from fairness_checker import fairness_csv_checker
c = fairness_csv_checker("compas-scores-two-years.csv")
```

Then you can call fairness measure functions. For example:

```python
c.demographic_parity(0.2, lambda row: row['sex'] == 'Male', lambda row: row['score_text'] in {'Medium', 'High'})
```

Output:

```txt
demographic parity
fair: 0.04 < 0.2
```

Note the function signature of `demographic_parity`:
```python
demographic_parity(ratio: float,
                   privileged_predicate: Callable[[csv_row], bool],
                   positive_predicate: Callable[[T], bool]) -> bool:
```

Here the `privileged_predicate` is

```python
lambda row: row['sex'] == 'Male'
```

meaning the privileged group is the male group, and the `positive_predicate` is

```python
lambda row: row['score_text'] in {'Medium', 'High'}
```

meaning the row is positive if the score is categorized as medium or high.

For a more complicated example involving parameters:

```python
c.conditional_statistical_parity(0.2, lambda row: row['sex'] == sex, lambda row: row['score_text'] in {'Medium', 'High'}, lambda x: (lambda row: int(row['priors_count']) > x), (0,))
```

Output:
```txt
conditional statistical parity
fair: 0.04 < 0.2
```

Note the function signature of `conditional_statistical_parity`:
```python
def conditional_statistical_parity(ratio: float,
                                   privileged_predicate: Callable[[csv_row], bool],
                                   positive_predicate: Callable[[csv_row], bool],
                                   legitimate_predicate_h: Callable[..., Callable[[csv_row], bool]],
                                   legitimate_arg: Tuple[Any, ...]) -> bool:
```

Here the higher order function `legitimate_predicate_h` is

```python
lambda x: (lambda row: int(row['priors_count']) > x)
```

and the argument to it, `legitimate_arg`, is `(0,)`.
