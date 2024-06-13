# Fairness Checker

This Python module fairness_checker provides a set of methods to evaluate the fairness of a predictive model's outcomes across different demographic groups represented in a CSV file or given a model.

## Dependencies

* Python >= 3.6

## Installation

```bash
pip3 install fairness-checker
```

## Usage

## As a library

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

## As a command line CLI

Prepare your dataset file. Create a predicate definition file containing arguments to the measure functions. For example, to calculate negative balance, create a file `test_predicates1.py` containing the following:

```python
def privileged_predicate(row):
    return row['sex'] == 'Male'

def score_predicate(row):
    return int(row['decile_score'])

def truth_predicate(row):
    return row['is_recid'] == '1'
```

Make sure the order of the definitions are the same as the order of the function signature.

Then execute the client in command line:

```bash
python3 -m fairness_checker
```

You'll be asked a few questions about which fairness measure you want to calculate and what ratio you want to set, like so:

```txt
Input dataset file name: compas-scores-two-years.csv
Input ratio: 0.2
Input the fairness measure: negative balance
Input the predicate definitions file name: test_predicates1.py
```

Output:

```
negative balance
fair: 0.07 < 0.2
```

For another example, let's calculate equal calibration with a predicate file `test_predicates2.py` containing the following:

```python
def privileged_predicate(row):
    return row['sex'] == 'Male'

def truth_predicate(row):
    return row['is_recid'] == '1'

def calib_predicate_h(u, l):
    def tmp(row):
        return l <= int(row['decile_score']) and int(row['decile_score']) <= u
    return tmp

calib_arg = (7, 5)
```

Again, the order of the definition matters. They must match that of the function signature.

Execute in command line:

```bash
python3 -m fairness_checker
```

You'll be asked a few questions about which fairness measure you want to calculate and what ratio you want to set, like so:

```txt
Input dataset file name: compas-scores-two-years.csv
Input ratio: 0.2
Input the fairness measure: equal calibration
Input the predicate definitions file name: test_predicates2.py
```

Output:

```
equal calibration
fair: 0.10 < 0.2
```
