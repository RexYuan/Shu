import os
import csv
import math

from typing import Callable, Iterable, Dict, TypeVar, Any, Tuple, Protocol, Union, Sequence
from contextlib import contextmanager

csv_row = Dict[str, str]
T = TypeVar('T')
K = TypeVar('K')
class Predictable(Protocol):
    def predict(self, filename : str) -> Sequence[Any]:
        ...

def print_stats(title, measure, ratio):
    print(title)
    fair_string = "fair" if measure < ratio else "unfair"
    if measure >= 0.01:
        print(f"{fair_string}: {measure:.2f} < {ratio}")
    else:
        precision = math.ceil(-math.log10(abs(measure) - abs(math.floor(measure))))
        print(f"{fair_string}: {measure:.{precision}f} < {ratio}")

class fairness_model_checker:
    def __init__(self, raw_file: str, verbose: bool = True):
        """
        Initializes the fairness checker.

        Parameters:
        raw_file (str): Path to the raw data file in CSV format.
        """
        self.verbose: bool = verbose
        with open(raw_file, 'r') as file:
            reader = csv.DictReader(file)
            self.fieldnames = list(reader.fieldnames or [])
            self.reader = list(reader)

    def write_to_csv(self, filename: str, rows: Iterable[csv_row]):
        with open(filename, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: v for k, v in row.items() if k in self.fieldnames})

    def remove_csv(self, filename: str):
        os.remove(filename)

    @contextmanager
    def csv_file_written(self, filename: str, rows: Iterable[csv_row]):
        self.write_to_csv(filename, rows)
        yield
        self.remove_csv(filename)

    def disparate_impact(self,
                         ratio: float,
                         model: Predictable,
                         privileged_predicate: Callable[[csv_row], bool],
                         positive_predicate: Callable[[T], bool],
                         value: bool = False) -> Union[bool, float]:
        """
        Evaluates the disparate impact of the model's predictions between privileged and unprivileged groups.

        Parameters:
        ratio (float): The fairness threshold ratio.
        model (object): The predictive model object.
            model.predict: Callable[[str], Iterable[T]]
        privileged_predicate (Callable[[csv_row], bool]): A function that determines whether a data point is privileged or not.
        positive_predicate (Callable[[T], bool]): A function that determines whether the model's prediction is positive.

        Returns:
        bool: If the input model satisfies the required fairness threshold ratio.
        """
        privileged = list(filter(lambda row: privileged_predicate(row), self.reader))
        unprivileged = list(filter(lambda row: not privileged_predicate(row), self.reader))

        with self.csv_file_written('privileged.csv', privileged):
            with self.csv_file_written('unprivileged.csv', unprivileged):
                privileged_Y_pred = model.predict('privileged.csv')
                unprivileged_Y_pred = model.predict('unprivileged.csv')

        privileged_percentage = len(list(filter(positive_predicate, privileged_Y_pred))) / len(privileged_Y_pred)
        unprivileged_percentage = len(list(filter(positive_predicate, unprivileged_Y_pred))) / len(unprivileged_Y_pred)

        measure = unprivileged_percentage / privileged_percentage

        if self.verbose:
            print_stats('disparate impact', measure, ratio)

        if value:
            return measure
        return measure > ratio

    def demographic_parity(self,
                           ratio: float,
                           model: Predictable,
                           privileged_predicate: Callable[[csv_row], bool],
                           positive_predicate: Callable[[T], bool],
                           value: bool = False) -> Union[bool, float]:
        """
        Evaluates the demographic parity of the model's predictions between privileged and unprivileged groups.

        Parameters:
        ratio (float): The fairness threshold ratio.
        model (object): The predictive model object.
            model.predict: Callable[[str], Iterable[T]]
        privileged_predicate (Callable[[csv_row], bool]): A function that determines whether a data point is privileged or not.
        positive_predicate (Callable[[T], bool]): A function that determines whether the model's prediction is positive.

        Returns:
        bool: If the input model satisfies the required fairness threshold ratio.
        """
        privileged = filter(lambda row: privileged_predicate(row), self.reader)
        unprivileged = filter(lambda row: not privileged_predicate(row), self.reader)

        with self.csv_file_written('privileged.csv', privileged):
            with self.csv_file_written('unprivileged.csv', unprivileged):
                privileged_Y_pred = model.predict('privileged.csv')
                unprivileged_Y_pred = model.predict('unprivileged.csv')

        privileged_percentage = len(list(filter(positive_predicate, privileged_Y_pred))) / len(privileged_Y_pred)
        unprivileged_percentage = len(list(filter(positive_predicate, unprivileged_Y_pred))) / len(unprivileged_Y_pred)

        measure = abs(unprivileged_percentage - privileged_percentage)

        if self.verbose:
            print_stats('demographic parity', measure, ratio)

        if value:
            return measure
        return measure < ratio

    def equalized_odds(self,
                       ratio: float,
                       model: Predictable,
                       privileged_predicate: Callable[[csv_row], bool],
                       positive_predicate: Callable[[T], bool],
                       truth_predicate: Callable[[csv_row], bool],
                       value: bool = False) -> Union[bool, Tuple[float, float]]:
        """
        Evaluates the equalized odds of the model's predictions between privileged and unprivileged groups.

        Parameters:
        ratio (float): The fairness threshold ratio.
        model (object): The predictive model object.
            model.predict: Callable[[str], Iterable[T]]
        privileged_predicate (Callable[[csv_row], bool]): A function that determines whether a data point is privileged or not.
        positive_predicate (Callable[[T], bool]): A function that determines whether the model's prediction is positive.
        truth_predicate: Callable[[csv_row], bool]: A function that determines a data point's ground truth.

        Returns:
        bool: If the input model satisfies the required fairness threshold ratio.
        """
        # false positive
        privileged_untruth = filter(lambda row: privileged_predicate(row) and not truth_predicate(row), self.reader)
        unprivileged_untruth = filter(lambda row: not privileged_predicate(row) and not truth_predicate(row), self.reader)

        with self.csv_file_written('privileged_untruth.csv', privileged_untruth):
            with self.csv_file_written('unprivileged_untruth.csv', unprivileged_untruth):
                privileged_untruth_Y_pred = model.predict('privileged_untruth.csv')
                unprivileged_untruth_Y_pred = model.predict('unprivileged_untruth.csv')

        privileged_untruth_percentage = len(list(filter(positive_predicate, privileged_untruth_Y_pred))) / len(privileged_untruth_Y_pred)
        unprivileged_untruth_percentage = len(list(filter(positive_predicate, unprivileged_untruth_Y_pred))) / len(unprivileged_untruth_Y_pred)
        measure1 = abs(privileged_untruth_percentage - unprivileged_untruth_percentage)

        if self.verbose:
            print_stats('equalized odds: false positive', measure1, ratio)

        # true positive
        privileged_truth = filter(lambda row: privileged_predicate(row) and truth_predicate(row), self.reader)
        unprivileged_truth = filter(lambda row: not privileged_predicate(row) and truth_predicate(row), self.reader)

        with self.csv_file_written('privileged_truth.csv', privileged_truth):
            with self.csv_file_written('unprivileged_truth.csv', unprivileged_truth):
                privileged_truth_Y_pred = model.predict('privileged_truth.csv')
                unprivileged_truth_Y_pred = model.predict('unprivileged_truth.csv')

        privileged_truth_percentage = len(list(filter(positive_predicate, privileged_truth_Y_pred))) / len(privileged_truth_Y_pred)
        unprivileged_truth_percentage = len(list(filter(positive_predicate, unprivileged_truth_Y_pred))) / len(unprivileged_truth_Y_pred)
        measure2 = abs(privileged_truth_percentage - unprivileged_truth_percentage)

        if self.verbose:
            print_stats('equalized odds: true positive', measure2, ratio)

        if value:
            return measure1, measure2
        return measure1 < ratio and measure2 < ratio

    def equal_opportunity(self,
                          ratio: float,
                          model: Predictable,
                          privileged_predicate: Callable[[csv_row], bool],
                          positive_predicate: Callable[[T], bool],
                          truth_predicate: Callable[[csv_row], bool],
                          value: bool = False) -> Union[bool, float]:
        """
        Evaluates the equal opportunity of the model's predictions between privileged and unprivileged groups.

        Parameters:
        ratio (float): The fairness threshold ratio.
        model (object): The predictive model object.
            model.predict: Callable[[str], Iterable[T]]
        privileged_predicate (Callable[[csv_row], bool]): A function that determines whether a data point is privileged or not.
        positive_predicate (Callable[[T], bool]): A function that determines whether the model's prediction is positive.
        truth_predicate: Callable[[csv_row], bool]: A function that determines a data point's ground truth.

        Returns:
        bool: If the input model satisfies the required fairness threshold ratio.
        """
        # true positive
        privileged_truth = filter(lambda row: privileged_predicate(row) and truth_predicate(row), self.reader)
        unprivileged_truth = filter(lambda row: not privileged_predicate(row) and truth_predicate(row), self.reader)

        with self.csv_file_written('privileged_truth.csv', privileged_truth):
            with self.csv_file_written('unprivileged_truth.csv', unprivileged_truth):
                privileged_truth_Y_pred = model.predict('privileged_truth.csv')
                unprivileged_truth_Y_pred = model.predict('unprivileged_truth.csv')

        privileged_truth_percentage = len(list(filter(positive_predicate, privileged_truth_Y_pred))) / len(privileged_truth_Y_pred)
        unprivileged_truth_percentage = len(list(filter(positive_predicate, unprivileged_truth_Y_pred))) / len(unprivileged_truth_Y_pred)
        measure = abs(privileged_truth_percentage - unprivileged_truth_percentage)

        if self.verbose:
            print_stats('equal_opportunity: true positive', measure, ratio)

        if value:
            return measure
        return measure < ratio

    def accuracy_eqaulity(self,
                          ratio: float,
                          model: Predictable,
                          privileged_predicate: Callable[[csv_row], bool],
                          positive_predicate: Callable[[T], bool],
                          truth_predicate: Callable[[csv_row], bool],
                          value: bool = False) -> Union[bool, float]:
        """
        Evaluates the accuracy eqaulity of the model's predictions between privileged and unprivileged groups.

        Parameters:
        ratio (float): The fairness threshold ratio.
        model (object): The predictive model object.
            model.predict: Callable[[str], Iterable[T]]
        privileged_predicate (Callable[[csv_row], bool]): A function that determines whether a data point is privileged or not.
        positive_predicate (Callable[[csv_row], bool]): A function that determines whether the model's prediction is positive.
        truth_predicate: Callable[[csv_row], bool]: A function that determines a data point's ground truth.

        Returns:
        bool: If the input model satisfies the required fairness threshold ratio.
        """
        privileged = list(filter(lambda row: privileged_predicate(row), self.reader))
        unprivileged = list(filter(lambda row: not privileged_predicate(row), self.reader))

        with self.csv_file_written('privileged.csv', privileged):
            with self.csv_file_written('unprivileged.csv', unprivileged):
                privileged_Y_pred = model.predict('privileged.csv')
                unprivileged_Y_pred = model.predict('unprivileged.csv')

        privileged_accurate = list(filter(lambda row_Y: (positive_predicate(row_Y[1]) and truth_predicate(row_Y[0])) or
                                                      (not positive_predicate(row_Y[1]) and not truth_predicate(row_Y[0])), zip(privileged, privileged_Y_pred)))
        unprivileged_accurate = list(filter(lambda row_Y: (positive_predicate(row_Y[1]) and truth_predicate(row_Y[0])) or
                                                      (not positive_predicate(row_Y[1]) and not truth_predicate(row_Y[0])), zip(unprivileged, unprivileged_Y_pred)))

        privileged_accurate_percentage = len(privileged_accurate) / len(privileged)
        unprivileged_accurate_percentage = len(unprivileged_accurate) / len(unprivileged)

        measure = abs(privileged_accurate_percentage - unprivileged_accurate_percentage)

        if self.verbose:
            print_stats('accuracy equality', measure, ratio)

        if value:
            return measure
        return measure < ratio

    def predictive_parity(self,
                          ratio: float,
                          model: Predictable,
                          privileged_predicate: Callable[[csv_row], bool],
                          positive_predicate: Callable[[T], bool],
                          truth_predicate: Callable[[csv_row], bool],
                          value: bool = False) -> Union[bool, float]:
        """
        Evaluates the predictive parity of the model's predictions between privileged and unprivileged groups.

        Parameters:
        ratio (float): The fairness threshold ratio.
        model (object): The predictive model object.
            model.predict: Callable[[str], Iterable[T]]
        privileged_predicate (Callable[[csv_row], bool]): A function that determines whether a data point is privileged or not.
        positive_predicate (Callable[[csv_row], bool]): A function that determines whether the model's prediction is positive.
        truth_predicate: Callable[[csv_row], bool]: A function that determines a data point's ground truth.

        Returns:
        bool: If the input model satisfies the required fairness threshold ratio.
        """
        privileged = list(filter(lambda row: privileged_predicate(row), self.reader))
        unprivileged = list(filter(lambda row: not privileged_predicate(row), self.reader))

        with self.csv_file_written('privileged.csv', privileged):
            with self.csv_file_written('unprivileged.csv', unprivileged):
                privileged_Y_pred = model.predict('privileged.csv')
                unprivileged_Y_pred = model.predict('unprivileged.csv')

        privileged_positive = list(filter(lambda row_Y: privileged_predicate(row_Y[0]) and positive_predicate(row_Y[1]), zip(self.reader, privileged_Y_pred)))
        unprivileged_positive = list(filter(lambda row_Y: not privileged_predicate(row_Y[0]) and positive_predicate(row_Y[1]), zip(self.reader, unprivileged_Y_pred)))

        privileged_positive_csv_row = list(map(lambda row_Y: row_Y[0], privileged_positive))
        unprivileged_positive_csv_row = list(map(lambda row_Y: row_Y[0], unprivileged_positive))

        privileged_positive_truth = list(filter(lambda row: truth_predicate(row), privileged_positive_csv_row))
        unprivileged_positive_truth = list(filter(lambda row: truth_predicate(row), unprivileged_positive_csv_row))

        privileged_positive_truth_percentage = len(privileged_positive_truth) / len(privileged_positive)
        unprivileged_positive_truth_percentage = len(unprivileged_positive_truth) / len(unprivileged_positive)
        measure = abs(privileged_positive_truth_percentage - unprivileged_positive_truth_percentage)

        if self.verbose:
            print_stats('predictive parity', measure, ratio)

        if value:
            return measure
        return measure < ratio

    def equal_calibration(self,
                          ratio: float,
                          model: Predictable,
                          privileged_predicate: Callable[[csv_row], bool],
                          truth_predicate: Callable[[csv_row], bool],
                          calib_predicate_h: Callable[..., Callable[[T], bool]],
                          calib_arg: Tuple[Any, ...],
                          value: bool = False) -> Union[bool, float]:
        """
        Evaluates the equal calibration of the model's predictions between privileged and unprivileged groups.

        Parameters:
        ratio (float): The fairness threshold ratio.
        model (object): The predictive model object.
            model.predict: Callable[[str], Iterable[T]]
        privileged_predicate (Callable[[csv_row], bool]): A function that determines whether a data point is privileged or not.
        positive_predicate (Callable[[csv_row], bool]): A function that determines whether the model's prediction is positive.
        truth_predicate: Callable[[csv_row], bool]: A function that determines a data point's ground truth.

        Returns:
        bool: If the input model satisfies the required fairness threshold ratio.
        """
        privileged = list(filter(lambda row: privileged_predicate(row), self.reader))
        unprivileged = list(filter(lambda row: not privileged_predicate(row), self.reader))

        with self.csv_file_written('privileged.csv', privileged):
            with self.csv_file_written('unprivileged.csv', unprivileged):
                privileged_Y_pred = model.predict('privileged.csv')
                unprivileged_Y_pred = model.predict('unprivileged.csv')

        calib_predicate = calib_predicate_h(*calib_arg)

        privileged_score = list(filter(lambda row_Y: calib_predicate(row_Y[1]), zip(privileged, privileged_Y_pred)))
        unprivileged_score = list(filter(lambda row_Y: calib_predicate(row_Y[1]), zip(unprivileged, unprivileged_Y_pred)))

        privileged_score_csv_row = list(map(lambda row_Y: row_Y[0], privileged_score))
        unprivileged_score_csv_row = list(map(lambda row_Y: row_Y[0], unprivileged_score))

        privileged_positive_truth = list(filter(lambda row: truth_predicate(row), privileged_score_csv_row))
        unprivileged_positive_truth = list(filter(lambda row: truth_predicate(row), unprivileged_score_csv_row))

        privileged_positive_truth_percentage = len(privileged_positive_truth) / len(privileged_score)
        unprivileged_positive_truth_percentage = len(unprivileged_positive_truth) / len(unprivileged_score)
        measure = abs(privileged_positive_truth_percentage - unprivileged_positive_truth_percentage)

        if self.verbose:
            print_stats('equal calibration', measure, ratio)

        if value:
            return measure
        return measure < ratio

    def conditional_statistical_parity(self,
                                       ratio: float,
                                       model: Predictable,
                                       privileged_predicate: Callable[[csv_row], bool],
                                       positive_predicate: Callable[[T], bool],
                                       legitimate_predicate_h: Callable[..., Callable[[csv_row], bool]],
                                       legitimate_arg: Tuple[Any, ...],
                                       value: bool = False) -> Union[bool, float]:
        """
        Evaluates the conditional statistical parity of the model's predictions between privileged and unprivileged groups.

        Parameters:
        ratio (float): The fairness threshold ratio.
        model (object): The predictive model object.
            model.predict: Callable[[str], Iterable[T]]
        privileged_predicate (Callable[[csv_row], bool]): A function that determines whether a data point is privileged or not.
        positive_predicate (Callable[[csv_row], bool]): A function that determines whether the model's prediction is positive.
        legitimate_predicate: Callable[[csv_row], bool]: A function that determines a data point's is legitimate.

        Returns:
        bool: If the input model satisfies the required fairness threshold ratio.
        """
        legitimate_predicate = legitimate_predicate_h(*legitimate_arg)
        privileged_legitimate = list(filter(lambda row: privileged_predicate(row) and legitimate_predicate(row), self.reader))
        unprivileged_legitimate = list(filter(lambda row: not privileged_predicate(row) and legitimate_predicate(row), self.reader))

        privileged = list(filter(lambda row: privileged_predicate(row), self.reader))
        unprivileged = list(filter(lambda row: not privileged_predicate(row), self.reader))

        with self.csv_file_written('privileged.csv', privileged):
            with self.csv_file_written('unprivileged.csv', unprivileged):
                privileged_Y_pred = model.predict('privileged.csv')
                unprivileged_Y_pred = model.predict('unprivileged.csv')

        privileged_legitimate_positive = list(filter(lambda Y: positive_predicate(Y), privileged_Y_pred))
        unprivileged_legitimate_positive = list(filter(lambda Y: positive_predicate(Y), unprivileged_Y_pred))

        privileged_legitimate_positive_percentage = len(privileged_legitimate_positive) / len(privileged_legitimate)
        unprivileged_legitimate_positive_percentage = len(unprivileged_legitimate_positive) / len(unprivileged_legitimate)
        measure = abs(privileged_legitimate_positive_percentage - unprivileged_legitimate_positive_percentage)

        if self.verbose:
            print_stats('conditional statistical parity', measure, ratio)

        if value:
            return measure
        return measure < ratio

    def predictive_equality(self,
                            ratio: float,
                            model: Predictable,
                            privileged_predicate: Callable[[csv_row], bool],
                            positive_predicate: Callable[[T], bool],
                            truth_predicate: Callable[[csv_row], bool],
                            value: bool = False) -> Union[bool, float]:
        """
        Evaluates the predictive equality of the model's predictions between privileged and unprivileged groups.

        Parameters:
        ratio (float): The fairness threshold ratio.
        model (object): The predictive model object.
            model.predict: Callable[[str], Iterable[T]]
        privileged_predicate (Callable[[csv_row], bool]): A function that determines whether a data point is privileged or not.
        positive_predicate (Callable[[T], bool]): A function that determines whether the model's prediction is positive.
        truth_predicate: Callable[[csv_row], bool]: A function that determines a data point's ground truth.

        Returns:
        bool: If the input model satisfies the required fairness threshold ratio.
        """
        # false positive
        privileged_untruth = filter(lambda row: privileged_predicate(row) and not truth_predicate(row), self.reader)
        unprivileged_untruth = filter(lambda row: not privileged_predicate(row) and not truth_predicate(row), self.reader)

        with self.csv_file_written('privileged_untruth.csv', privileged_untruth):
            with self.csv_file_written('unprivileged_untruth.csv', unprivileged_untruth):
                privileged_untruth_Y_pred = model.predict('privileged_untruth.csv')
                unprivileged_untruth_Y_pred = model.predict('unprivileged_untruth.csv')

        privileged_untruth_percentage = len(list(filter(positive_predicate, privileged_untruth_Y_pred))) / len(privileged_untruth_Y_pred)
        unprivileged_untruth_percentage = len(list(filter(positive_predicate, unprivileged_untruth_Y_pred))) / len(unprivileged_untruth_Y_pred)
        measure = abs(privileged_untruth_percentage - unprivileged_untruth_percentage)

        if self.verbose:
            print_stats('predictive equality: false positive', measure, ratio)

        if value:
            return measure
        return measure < ratio

    def conditional_use_accuracy_equality(self,
                                          ratio: float,
                                          model: Predictable,
                                          privileged_predicate: Callable[[csv_row], bool],
                                          positive_predicate: Callable[[T], bool],
                                          truth_predicate: Callable[[csv_row], bool],
                                          value: bool = False) -> Union[bool, Tuple[float, float]]:
        """
        Evaluates the conditional use accuracy equality of the model's predictions between privileged and unprivileged groups.

        Parameters:
        ratio (float): The fairness threshold ratio.
        model (object): The predictive model object.
            model.predict: Callable[[str], Iterable[T]]
        privileged_predicate (Callable[[csv_row], bool]): A function that determines whether a data point is privileged or not.
        positive_predicate (Callable[[csv_row], bool]): A function that determines whether the model's prediction is positive.
        truth_predicate: Callable[[csv_row], bool]: A function that determines a data point's ground truth.

        Returns:
        bool: If the input model satisfies the required fairness threshold ratio.
        """
        privileged = list(filter(lambda row: privileged_predicate(row), self.reader))
        unprivileged = list(filter(lambda row: not privileged_predicate(row), self.reader))

        with self.csv_file_written('privileged.csv', privileged):
            with self.csv_file_written('unprivileged.csv', unprivileged):
                privileged_Y_pred = model.predict('privileged.csv')
                unprivileged_Y_pred = model.predict('unprivileged.csv')

        privileged_positive = list(filter(lambda row_Y: privileged_predicate(row_Y[0]) and positive_predicate(row_Y[1]), zip(self.reader, privileged_Y_pred)))
        unprivileged_positive = list(filter(lambda row_Y: not privileged_predicate(row_Y[0]) and positive_predicate(row_Y[1]), zip(self.reader, unprivileged_Y_pred)))

        privileged_positive_csv_row = list(map(lambda row_Y: row_Y[0], privileged_positive))
        unprivileged_positive_csv_row = list(map(lambda row_Y: row_Y[0], unprivileged_positive))

        privileged_positive_truth = list(filter(lambda row: truth_predicate(row), privileged_positive_csv_row))
        unprivileged_positive_truth = list(filter(lambda row: truth_predicate(row), unprivileged_positive_csv_row))

        privileged_positive_truth_percentage = len(privileged_positive_truth) / len(privileged_positive)
        unprivileged_positive_truth_percentage = len(unprivileged_positive_truth) / len(unprivileged_positive)
        measure1 = abs(privileged_positive_truth_percentage - unprivileged_positive_truth_percentage)

        if self.verbose:
            print_stats('conditional use accuracy equality: true positive', measure1, ratio)

        privileged_negative = list(filter(lambda row_Y: privileged_predicate(row_Y[0]) and not positive_predicate(row_Y[1]), zip(self.reader, privileged_Y_pred)))
        unprivileged_negative = list(filter(lambda row_Y: not privileged_predicate(row_Y[0]) and not positive_predicate(row_Y[1]), zip(self.reader, unprivileged_Y_pred)))

        privileged_negative_csv_row = list(map(lambda row_Y: row_Y[0], privileged_negative))
        unprivileged_negative_csv_row = list(map(lambda row_Y: row_Y[0], unprivileged_negative))

        privileged_negative_untruth = list(filter(lambda row: not truth_predicate(row), privileged_negative_csv_row))
        unprivileged_negative_untruth = list(filter(lambda row: not truth_predicate(row), unprivileged_negative_csv_row))

        privileged_negative_untruth_percentage = len(privileged_negative_untruth) / len(privileged_negative)
        unprivileged_negative_untruth_percentage = len(unprivileged_negative_untruth) / len(unprivileged_negative)
        measure2 = abs(privileged_negative_untruth_percentage - unprivileged_negative_untruth_percentage)

        if self.verbose:
            print_stats('conditional use accuracy equality: true negative', measure2, ratio)

        if value:
            return measure1, measure2
        return measure1 < ratio and measure2 < ratio and measure1 < measure2

    def positive_balance(self,
                         ratio: float,
                         model: Predictable,
                         privileged_predicate: Callable[[csv_row], bool],
                         score_predicate: Callable[[T], K],
                         truth_predicate: Callable[[csv_row], bool],
                         value: bool = False) -> Union[bool, float]:
        """
        Evaluates the positive balance of the model's predictions between privileged and unprivileged groups.

        Parameters:
        ratio (float): The fairness threshold ratio.
        model (object): The predictive model object.
            model.predict: Callable[[str], Iterable[T]]
        privileged_predicate (Callable[[csv_row], bool]): A function that determines whether a data point is privileged or not.
        positive_predicate (Callable[[csv_row], bool]): A function that determines whether the model's prediction is positive.
        truth_predicate: Callable[[csv_row], bool]: A function that determines a data point's ground truth.

        Returns:
        bool: If the input model satisfies the required fairness threshold ratio.
        """
        privileged_truth = list(filter(lambda row: privileged_predicate(row) and truth_predicate(row), self.reader))
        unprivileged_truth = list(filter(lambda row: not privileged_predicate(row) and truth_predicate(row), self.reader))

        privileged = list(filter(lambda row: privileged_predicate(row), self.reader))
        unprivileged = list(filter(lambda row: not privileged_predicate(row), self.reader))

        with self.csv_file_written('privileged.csv', privileged):
            with self.csv_file_written('unprivileged.csv', unprivileged):
                privileged_Y_pred = model.predict('privileged.csv')
                unprivileged_Y_pred = model.predict('unprivileged.csv')

        privileged_truth_score_mean = sum(privileged_Y_pred) / len(privileged_truth)
        unprivileged_truth_score_mean = sum(unprivileged_Y_pred) / len(unprivileged_truth)

        measure = abs(privileged_truth_score_mean - unprivileged_truth_score_mean)

        if self.verbose:
            print_stats('positive balance', measure, ratio)

        if value:
            return measure
        return measure < ratio

    def negative_balance(self,
                         ratio: float,
                         model: Predictable,
                         privileged_predicate: Callable[[csv_row], bool],
                         score_predicate: Callable[[T], K],
                         truth_predicate: Callable[[csv_row], bool],
                         value: bool = False) -> Union[bool, float]:
        """
        Evaluates the negative balance of the model's predictions between privileged and unprivileged groups.

        Parameters:
        ratio (float): The fairness threshold ratio.
        model (object): The predictive model object.
            model.predict: Callable[[str], Iterable[T]]
        privileged_predicate (Callable[[csv_row], bool]): A function that determines whether a data point is privileged or not.
        positive_predicate (Callable[[csv_row], bool]): A function that determines whether the model's prediction is positive.
        truth_predicate: Callable[[csv_row], bool]: A function that determines a data point's ground truth.

        Returns:
        bool: If the input model satisfies the required fairness threshold ratio.
        """
        privileged_untruth = list(filter(lambda row: privileged_predicate(row) and not truth_predicate(row), self.reader))
        unprivileged_untruth = list(filter(lambda row: not privileged_predicate(row) and not truth_predicate(row), self.reader))

        privileged = list(filter(lambda row: privileged_predicate(row), self.reader))
        unprivileged = list(filter(lambda row: not privileged_predicate(row), self.reader))

        with self.csv_file_written('privileged.csv', privileged):
            with self.csv_file_written('unprivileged.csv', unprivileged):
                privileged_Y_pred = model.predict('privileged.csv')
                unprivileged_Y_pred = model.predict('unprivileged.csv')

        privileged_untruth_score_mean = sum(privileged_Y_pred) / len(privileged_untruth)
        unprivileged_untruth_score_mean = sum(unprivileged_Y_pred) / len(unprivileged_untruth)

        measure = abs(privileged_untruth_score_mean - unprivileged_untruth_score_mean)

        if self.verbose:
            print_stats('negative balance', measure, ratio)

        if value:
            return measure
        return measure < ratio

    def mean_difference(self,
                        ratio: float,
                        model: Predictable,
                        privileged_predicate: Callable[[csv_row], bool],
                        positive_predicate: Callable[[T], bool],
                        value: bool = False) -> Union[bool, float]:
        """
        Evaluates the mean difference of the model's predictions between privileged and unprivileged groups.

        Parameters:
        ratio (float): The fairness threshold ratio.
        model (object): The predictive model object.
            model.predict: Callable[[str], Iterable[T]]
        privileged_predicate (Callable[[csv_row], bool]): A function that determines whether a data point is privileged or not.
        positive_predicate (Callable[[csv_row], bool]): A function that determines whether the model's prediction is positive.

        Returns:
        bool: If the input model satisfies the required fairness threshold ratio.
        """
        privileged = list(filter(lambda row: privileged_predicate(row), self.reader))
        unprivileged = list(filter(lambda row: not privileged_predicate(row), self.reader))

        with self.csv_file_written('privileged.csv', privileged):
            with self.csv_file_written('unprivileged.csv', unprivileged):
                privileged_Y_pred = model.predict('privileged.csv')
                unprivileged_Y_pred = model.predict('unprivileged.csv')

        privileged_positive = list(filter(lambda row_Y: privileged_predicate(row_Y[0]) and positive_predicate(row_Y[1]), zip(self.reader, privileged_Y_pred)))
        unprivileged_positive = list(filter(lambda row_Y: not privileged_predicate(row_Y[0]) and positive_predicate(row_Y[1]), zip(self.reader, unprivileged_Y_pred)))

        privileged_positive_percentage = len(privileged_positive) / len(privileged)
        unprivileged_positive_percentage = len(unprivileged_positive) / len(unprivileged)

        measure = abs(privileged_positive_percentage - unprivileged_positive_percentage)

        if self.verbose:
            print_stats('mean difference', measure, ratio)

        if value:
            return measure
        return measure < ratio

class fairness_csv_checker:
    def __init__(self, raw_file: str, verbose: bool = True):
        """
        Initializes the fairness checker.

        Parameters:
        raw_file (str): Path to the raw data file in CSV format.
        """
        self.verbose = verbose
        with open(raw_file, 'r') as file:
            reader = csv.DictReader(file)
            self.fieldnames = reader.fieldnames
            self.reader = list(reader)

    def disparate_impact(self,
                         ratio: float,
                         privileged_predicate: Callable[[csv_row], bool],
                         positive_predicate: Callable[[csv_row], bool],
                         value: bool = False) -> Union[bool, float]:
        """
        Evaluates the disparate impact of the model's predictions between privileged and unprivileged groups.

        Parameters:
        ratio (float): The fairness threshold ratio.
        privileged_predicate (Callable[[csv_row], bool]): A function that determines whether a data point is privileged or not.
        positive_predicate (Callable[[csv_row], bool]): A function that determines whether the model's prediction is positive.

        Returns:
        bool: If the input model satisfies the required fairness threshold ratio.
        """
        privileged = list(filter(lambda row: privileged_predicate(row), self.reader))
        unprivileged = list(filter(lambda row: not privileged_predicate(row), self.reader))

        privileged_Y_result = list(filter(lambda row: positive_predicate(row), privileged))
        unprivileged_Y_result = list(filter(lambda row: positive_predicate(row), unprivileged))

        privileged_percentage = len(privileged_Y_result) / len(privileged)
        unprivileged_percentage = len(unprivileged_Y_result) / len(unprivileged)

        measure = unprivileged_percentage / privileged_percentage

        if self.verbose:
            print_stats('disparate impact', measure, ratio)

        if value:
            return measure
        return measure > ratio

    def demographic_parity(self,
                           ratio: float,
                           privileged_predicate: Callable[[csv_row], bool],
                           positive_predicate: Callable[[csv_row], bool],
                           value: bool = False) -> Union[bool, float]:
        """
        Evaluates the demographic parity of the model's predictions between privileged and unprivileged groups.

        Parameters:
        ratio (float): The fairness threshold ratio.
        privileged_predicate (Callable[[csv_row], bool]): A function that determines whether a data point is privileged or not.
        positive_predicate (Callable[[csv_row], bool]): A function that determines whether the model's prediction is positive.

        Returns:
        bool: If the input model satisfies the required fairness threshold ratio.
        """
        privileged = list(filter(lambda row: privileged_predicate(row), self.reader))
        unprivileged = list(filter(lambda row: not privileged_predicate(row), self.reader))

        privileged_Y_result = list(filter(lambda row: positive_predicate(row), privileged))
        unprivileged_Y_result = list(filter(lambda row: positive_predicate(row), unprivileged))

        privileged_percentage = len(privileged_Y_result) / len(privileged)
        unprivileged_percentage = len(unprivileged_Y_result) / len(unprivileged)

        measure = abs(unprivileged_percentage - privileged_percentage)

        if self.verbose:
            print_stats('demographic parity', measure, ratio)

        if value:
            return measure
        return measure < ratio

    def equalized_odds(self,
                       ratio: float,
                       privileged_predicate: Callable[[csv_row], bool],
                       positive_predicate: Callable[[csv_row], bool],
                       truth_predicate: Callable[[csv_row], bool],
                       value: bool = False) -> Union[bool, Tuple[float, float]]:
        """
        Evaluates the equalized odds of the model's predictions between privileged and unprivileged groups.

        Parameters:
        ratio (float): The fairness threshold ratio.
        privileged_predicate (Callable[[csv_row], bool]): A function that determines whether a data point is privileged or not.
        positive_predicate (Callable[[csv_row], bool]): A function that determines whether the model's prediction is positive.
        truth_predicate: Callable[[csv_row], bool]: A function that determines a data point's ground truth.

        Returns:
        bool: If the input model satisfies the required fairness threshold ratio.
        """
        # false positive
        privileged_untruth = list(filter(lambda row: privileged_predicate(row) and not truth_predicate(row), self.reader))
        unprivileged_untruth = list(filter(lambda row: not privileged_predicate(row) and not truth_predicate(row), self.reader))

        privileged_untruth_Y_result = list(filter(lambda row: positive_predicate(row), privileged_untruth))
        unprivileged_untruth_Y_result = list(filter(lambda row: positive_predicate(row), unprivileged_untruth))

        privileged_untruth_percentage = len(privileged_untruth_Y_result) / len(privileged_untruth)
        unprivileged_untruth_percentage = len(unprivileged_untruth_Y_result) / len(unprivileged_untruth)
        measure1 = abs(privileged_untruth_percentage - unprivileged_untruth_percentage)

        if self.verbose:
            print_stats('equalized odds: false positive', measure1, ratio)

        # true positive
        privileged_truth = list(filter(lambda row: privileged_predicate(row) and truth_predicate(row), self.reader))
        unprivileged_truth = list(filter(lambda row: not privileged_predicate(row) and truth_predicate(row), self.reader))

        privileged_truth_Y_result = list(filter(lambda row: positive_predicate(row), privileged_truth))
        unprivileged_truth_Y_result = list(filter(lambda row: positive_predicate(row), unprivileged_truth))

        privileged_truth_percentage = len(privileged_truth_Y_result) / len(privileged_truth)
        unprivileged_truth_percentage = len(unprivileged_truth_Y_result) / len(unprivileged_truth)
        measure2 = abs(privileged_truth_percentage - unprivileged_truth_percentage)

        if self.verbose:
            print_stats('equalized odds: true positive', measure2, ratio)

        if value:
            return measure1, measure2
        return measure1 < ratio and measure2 < ratio

    def equal_opportunity(self,
                          ratio: float,
                          privileged_predicate: Callable[[csv_row], bool],
                          positive_predicate: Callable[[csv_row], bool],
                          truth_predicate: Callable[[csv_row], bool],
                          value: bool = False) -> Union[bool, float]:
        """
        Evaluates the equal opportunity of the model's predictions between privileged and unprivileged groups.

        Parameters:
        ratio (float): The fairness threshold ratio.
        privileged_predicate (Callable[[csv_row], bool]): A function that determines whether a data point is privileged or not.
        positive_predicate (Callable[[csv_row], bool]): A function that determines whether the model's prediction is positive.
        truth_predicate: Callable[[csv_row], bool]: A function that determines a data point's ground truth.

        Returns:
        bool: If the input model satisfies the required fairness threshold ratio.
        """
        # true positive
        privileged_truth = list(filter(lambda row: privileged_predicate(row) and truth_predicate(row), self.reader))
        unprivileged_truth = list(filter(lambda row: not privileged_predicate(row) and truth_predicate(row), self.reader))

        privileged_truth_Y_result = list(filter(lambda row: positive_predicate(row), privileged_truth))
        unprivileged_truth_Y_result = list(filter(lambda row: positive_predicate(row), unprivileged_truth))

        privileged_truth_percentage = len(privileged_truth_Y_result) / len(privileged_truth)
        unprivileged_truth_percentage = len(unprivileged_truth_Y_result) / len(unprivileged_truth)
        measure = abs(privileged_truth_percentage - unprivileged_truth_percentage)

        if self.verbose:
            print_stats('equal_opportunity: true positive', measure, ratio)

        if value:
            return measure
        return measure < ratio

    def accuracy_eqaulity(self,
                          ratio: float,
                          privileged_predicate: Callable[[csv_row], bool],
                          positive_predicate: Callable[[csv_row], bool],
                          truth_predicate: Callable[[csv_row], bool],
                          value: bool = False) -> Union[bool, float]:
        """
        Evaluates the accuracy eqaulity of the model's predictions between privileged and unprivileged groups.

        Parameters:
        ratio (float): The fairness threshold ratio.
        privileged_predicate (Callable[[csv_row], bool]): A function that determines whether a data point is privileged or not.
        positive_predicate (Callable[[csv_row], bool]): A function that determines whether the model's prediction is positive.
        truth_predicate: Callable[[csv_row], bool]: A function that determines a data point's ground truth.

        Returns:
        bool: If the input model satisfies the required fairness threshold ratio.
        """
        privileged = list(filter(lambda row: privileged_predicate(row), self.reader))
        unprivileged = list(filter(lambda row: not privileged_predicate(row), self.reader))

        privileged_accurate = list(filter(lambda row: (positive_predicate(row) and truth_predicate(row)) or
                                                      (not positive_predicate(row) and not truth_predicate(row)), privileged))
        unprivileged_accurate = list(filter(lambda row: (positive_predicate(row) and truth_predicate(row)) or
                                                      (not positive_predicate(row) and not truth_predicate(row)), unprivileged))

        privileged_accurate_percentage = len(privileged_accurate) / len(privileged)
        unprivileged_accurate_percentage = len(unprivileged_accurate) / len(unprivileged)

        measure = abs(privileged_accurate_percentage - unprivileged_accurate_percentage)

        if self.verbose:
            print_stats('accuracy equality', measure, ratio)

        if value:
            return measure
        return measure < ratio

    def predictive_parity(self,
                          ratio: float,
                          privileged_predicate: Callable[[csv_row], bool],
                          positive_predicate: Callable[[csv_row], bool],
                          truth_predicate: Callable[[csv_row], bool],
                          value: bool = False) -> Union[bool, float]:
        """
        Evaluates the predictive parity of the model's predictions between privileged and unprivileged groups.

        Parameters:
        ratio (float): The fairness threshold ratio.
        privileged_predicate (Callable[[csv_row], bool]): A function that determines whether a data point is privileged or not.
        positive_predicate (Callable[[csv_row], bool]): A function that determines whether the model's prediction is positive.
        truth_predicate: Callable[[csv_row], bool]: A function that determines a data point's ground truth.

        Returns:
        bool: If the input model satisfies the required fairness threshold ratio.
        """
        privileged_positive = list(filter(lambda row: privileged_predicate(row) and positive_predicate(row), self.reader))
        unprivileged_positive = list(filter(lambda row: not privileged_predicate(row) and positive_predicate(row), self.reader))

        privileged_positive_truth = list(filter(lambda row: truth_predicate(row), privileged_positive))
        unprivileged_positive_truth = list(filter(lambda row: truth_predicate(row), unprivileged_positive))

        privileged_positive_truth_percentage = len(privileged_positive_truth) / len(privileged_positive)
        unprivileged_positive_truth_percentage = len(unprivileged_positive_truth) / len(unprivileged_positive)
        measure = abs(privileged_positive_truth_percentage - unprivileged_positive_truth_percentage)

        if self.verbose:
            print_stats('predictive parity', measure, ratio)

        if value:
            return measure
        return measure < ratio

    def equal_calibration(self,
                          ratio: float,
                          privileged_predicate: Callable[[csv_row], bool],
                          truth_predicate: Callable[[csv_row], bool],
                          calib_predicate_h: Callable[..., Callable[[csv_row], bool]],
                          calib_arg: Tuple[Any, ...],
                          value: bool = False) -> Union[bool, float]:
        """
        Evaluates the equal calibration of the model's predictions between privileged and unprivileged groups.

        Parameters:
        ratio (float): The fairness threshold ratio.
        privileged_predicate (Callable[[csv_row], bool]): A function that determines whether a data point is privileged or not.
        positive_predicate (Callable[[csv_row], bool]): A function that determines whether the model's prediction is positive.
        truth_predicate: Callable[[csv_row], bool]: A function that determines a data point's ground truth.

        Returns:
        bool: If the input model satisfies the required fairness threshold ratio.
        """
        calib_predicate = calib_predicate_h(*calib_arg)

        privileged_score = list(filter(lambda row: privileged_predicate(row) and calib_predicate(row), self.reader))
        unprivileged_score = list(filter(lambda row: not privileged_predicate(row) and calib_predicate(row), self.reader))

        privileged_positive_truth = list(filter(lambda row: truth_predicate(row), privileged_score))
        unprivileged_positive_truth = list(filter(lambda row: truth_predicate(row), unprivileged_score))

        privileged_positive_truth_percentage = len(privileged_positive_truth) / len(privileged_score)
        unprivileged_positive_truth_percentage = len(unprivileged_positive_truth) / len(unprivileged_score)
        measure = abs(privileged_positive_truth_percentage - unprivileged_positive_truth_percentage)

        if self.verbose:
            print_stats('equal calibration', measure, ratio)

        if value:
            return measure
        return measure < ratio

    def conditional_statistical_parity(self,
                                       ratio: float,
                                       privileged_predicate: Callable[[csv_row], bool],
                                       positive_predicate: Callable[[csv_row], bool],
                                       legitimate_predicate_h: Callable[..., Callable[[csv_row], bool]],
                                       legitimate_arg: Tuple[Any, ...],
                                       value: bool = False) -> Union[bool, float]:
        """
        Evaluates the conditional statistical parity of the model's predictions between privileged and unprivileged groups.

        Parameters:
        ratio (float): The fairness threshold ratio.
        privileged_predicate (Callable[[csv_row], bool]): A function that determines whether a data point is privileged or not.
        positive_predicate (Callable[[csv_row], bool]): A function that determines whether the model's prediction is positive.
        legitimate_predicate: Callable[[csv_row], bool]: A function that determines a data point's is legitimate.

        Returns:
        bool: If the input model satisfies the required fairness threshold ratio.
        """
        legitimate_predicate = legitimate_predicate_h(*legitimate_arg)
        privileged_legitimate = list(filter(lambda row: privileged_predicate(row) and legitimate_predicate(row), self.reader))
        unprivileged_legitimate = list(filter(lambda row: not privileged_predicate(row) and legitimate_predicate(row), self.reader))

        privileged_legitimate_positive = list(filter(lambda row: positive_predicate(row), privileged_legitimate))
        unprivileged_legitimate_positive = list(filter(lambda row: positive_predicate(row), unprivileged_legitimate))

        privileged_legitimate_positive_percentage = len(privileged_legitimate_positive) / len(privileged_legitimate)
        unprivileged_legitimate_positive_percentage = len(unprivileged_legitimate_positive) / len(unprivileged_legitimate)
        measure = abs(privileged_legitimate_positive_percentage - unprivileged_legitimate_positive_percentage)

        if self.verbose:
            print_stats('conditional statistical parity', measure, ratio)

        if value:
            return measure
        return measure < ratio

    def predictive_equality(self,
                            ratio: float,
                            privileged_predicate: Callable[[csv_row], bool],
                            positive_predicate: Callable[[csv_row], bool],
                            truth_predicate: Callable[[csv_row], bool],
                            value: bool = False) -> Union[bool, float]:
        """
        Evaluates the predictive equality of the model's predictions between privileged and unprivileged groups.

        Parameters:
        ratio (float): The fairness threshold ratio.
        privileged_predicate (Callable[[csv_row], bool]): A function that determines whether a data point is privileged or not.
        positive_predicate (Callable[[csv_row], bool]): A function that determines whether the model's prediction is positive.
        truth_predicate: Callable[[csv_row], bool]: A function that determines a data point's ground truth.

        Returns:
        bool: If the input model satisfies the required fairness threshold ratio.
        """
        # false positive
        privileged_untruth = list(filter(lambda row: privileged_predicate(row) and not truth_predicate(row), self.reader))
        unprivileged_untruth = list(filter(lambda row: not privileged_predicate(row) and not truth_predicate(row), self.reader))

        privileged_untruth_Y_result = list(filter(lambda row: positive_predicate(row), privileged_untruth))
        unprivileged_untruth_Y_result = list(filter(lambda row: positive_predicate(row), unprivileged_untruth))

        privileged_untruth_percentage = len(privileged_untruth_Y_result) / len(privileged_untruth)
        unprivileged_untruth_percentage = len(unprivileged_untruth_Y_result) / len(unprivileged_untruth)
        measure = abs(privileged_untruth_percentage - unprivileged_untruth_percentage)

        if self.verbose:
            print_stats('predictive equality: false positive', measure, ratio)

        if value:
            return measure
        return measure < ratio

    def conditional_use_accuracy_equality(self,
                                          ratio: float,
                                          privileged_predicate: Callable[[csv_row], bool],
                                          positive_predicate: Callable[[csv_row], bool],
                                          truth_predicate: Callable[[csv_row], bool],
                                          value: bool = False) -> Union[bool, Tuple[float, float]]:
        """
        Evaluates the conditional use accuracy equality of the model's predictions between privileged and unprivileged groups.

        Parameters:
        ratio (float): The fairness threshold ratio.
        privileged_predicate (Callable[[csv_row], bool]): A function that determines whether a data point is privileged or not.
        positive_predicate (Callable[[csv_row], bool]): A function that determines whether the model's prediction is positive.
        truth_predicate: Callable[[csv_row], bool]: A function that determines a data point's ground truth.

        Returns:
        bool: If the input model satisfies the required fairness threshold ratio.
        """
        privileged_positive = list(filter(lambda row: privileged_predicate(row) and positive_predicate(row), self.reader))
        unprivileged_positive = list(filter(lambda row: not privileged_predicate(row) and positive_predicate(row), self.reader))

        privileged_positive_truth = list(filter(lambda row: truth_predicate(row), privileged_positive))
        unprivileged_positive_truth = list(filter(lambda row: truth_predicate(row), unprivileged_positive))

        privileged_positive_truth_percentage = len(privileged_positive_truth) / len(privileged_positive)
        unprivileged_positive_truth_percentage = len(unprivileged_positive_truth) / len(unprivileged_positive)
        measure1 = abs(privileged_positive_truth_percentage - unprivileged_positive_truth_percentage)

        if self.verbose:
            print_stats('conditional use accuracy equality: true positive', measure1, ratio)

        privileged_negative = list(filter(lambda row: privileged_predicate(row) and not positive_predicate(row), self.reader))
        unprivileged_negative = list(filter(lambda row: not privileged_predicate(row) and not positive_predicate(row), self.reader))

        privileged_negative_untruth = list(filter(lambda row: not truth_predicate(row), privileged_negative))
        unprivileged_negative_untruth = list(filter(lambda row: not truth_predicate(row), unprivileged_negative))

        privileged_negative_untruth_percentage = len(privileged_negative_untruth) / len(privileged_negative)
        unprivileged_negative_untruth_percentage = len(unprivileged_negative_untruth) / len(unprivileged_negative)
        measure2 = abs(privileged_negative_untruth_percentage - unprivileged_negative_untruth_percentage)

        if self.verbose:
            print_stats('conditional use accuracy equality: true negative', measure2, ratio)

        if value:
            return measure1, measure2
        return measure1 < ratio and measure2 < ratio and measure1 < measure2

    def positive_balance(self,
                         ratio: float,
                         privileged_predicate: Callable[[csv_row], bool],
                         score_predicate: Callable[[csv_row], bool],
                         truth_predicate: Callable[[csv_row], bool],
                         value: bool = False) -> Union[bool, float]:
        """
        Evaluates the positive balance of the model's predictions between privileged and unprivileged groups.

        Parameters:
        ratio (float): The fairness threshold ratio.
        privileged_predicate (Callable[[csv_row], bool]): A function that determines whether a data point is privileged or not.
        positive_predicate (Callable[[csv_row], bool]): A function that determines whether the model's prediction is positive.
        truth_predicate: Callable[[csv_row], bool]: A function that determines a data point's ground truth.

        Returns:
        bool: If the input model satisfies the required fairness threshold ratio.
        """
        privileged_truth = list(filter(lambda row: privileged_predicate(row) and truth_predicate(row), self.reader))
        unprivileged_truth = list(filter(lambda row: not privileged_predicate(row) and truth_predicate(row), self.reader))

        privileged_truth_score_mean = sum(map(lambda row: score_predicate(row), privileged_truth)) / len(privileged_truth)
        unprivileged_truth_score_mean = sum(map(lambda row: score_predicate(row), unprivileged_truth)) / len(unprivileged_truth)

        measure = abs(privileged_truth_score_mean - unprivileged_truth_score_mean)

        if self.verbose:
            print_stats('positive balance', measure, ratio)

        if value:
            return measure
        return measure < ratio

    def negative_balance(self,
                         ratio: float,
                         privileged_predicate: Callable[[csv_row], bool],
                         score_predicate: Callable[[csv_row], bool],
                         truth_predicate: Callable[[csv_row], bool],
                         value: bool = False) -> Union[bool, float]:
        """
        Evaluates the negative balance of the model's predictions between privileged and unprivileged groups.

        Parameters:
        ratio (float): The fairness threshold ratio.
        privileged_predicate (Callable[[csv_row], bool]): A function that determines whether a data point is privileged or not.
        positive_predicate (Callable[[csv_row], bool]): A function that determines whether the model's prediction is positive.
        truth_predicate: Callable[[csv_row], bool]: A function that determines a data point's ground truth.

        Returns:
        bool: If the input model satisfies the required fairness threshold ratio.
        """
        privileged_untruth = list(filter(lambda row: privileged_predicate(row) and not truth_predicate(row), self.reader))
        unprivileged_untruth = list(filter(lambda row: not privileged_predicate(row) and not truth_predicate(row), self.reader))

        privileged_untruth_score_mean = sum(map(lambda row: score_predicate(row), privileged_untruth)) / len(privileged_untruth)
        unprivileged_untruth_score_mean = sum(map(lambda row: score_predicate(row), unprivileged_untruth)) / len(unprivileged_untruth)

        measure = abs(privileged_untruth_score_mean - unprivileged_untruth_score_mean)

        if self.verbose:
            print_stats('negative balance', measure, ratio)

        if value:
            return measure
        return measure < ratio

    def mean_difference(self,
                        ratio: float,
                        privileged_predicate: Callable[[csv_row], bool],
                        positive_predicate: Callable[[csv_row], bool],
                        value: bool = False) -> Union[bool, float]:
        """
        Evaluates the mean difference of the model's predictions between privileged and unprivileged groups.

        Parameters:
        ratio (float): The fairness threshold ratio.
        privileged_predicate (Callable[[csv_row], bool]): A function that determines whether a data point is privileged or not.
        positive_predicate (Callable[[csv_row], bool]): A function that determines whether the model's prediction is positive.

        Returns:
        bool: If the input model satisfies the required fairness threshold ratio.
        """
        privileged = list(filter(lambda row: privileged_predicate(row), self.reader))
        unprivileged = list(filter(lambda row: not privileged_predicate(row), self.reader))

        privileged_positive = list(filter(lambda row: positive_predicate(row), privileged))
        unprivileged_positive = list(filter(lambda row: positive_predicate(row), unprivileged))

        privileged_positive_percentage = len(privileged_positive) / len(privileged)
        unprivileged_positive_percentage = len(unprivileged_positive) / len(unprivileged)

        measure = abs(privileged_positive_percentage - unprivileged_positive_percentage)

        if self.verbose:
            print_stats('mean difference', measure, ratio)

        if value:
            return measure
        return measure < ratio
