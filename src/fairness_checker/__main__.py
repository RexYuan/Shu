from .main import *
import importlib
import inspect

def main():
    # get the test data
    target_dataset = input("Input dataset file name: ")
    # target_dataset = "compas-scores-two-years.csv"

    try:
        checker = fairness_csv_checker(target_dataset)
    except FileNotFoundError as e:
        print("Failed to locate dataset file:", e)
        return

    # get the ratio
    target_ratio = input("Input ratio: ")
    # target_ratio = "0.2"
    ratio = float(target_ratio)

    # get the fairness measure
    target = input("Input the fairness measure: ")
    # target = "negative balance"
    # target = "equal calibration"

    availables = inspect.getmembers(fairness_csv_checker)
    availables = {name: obj for name, obj in availables if inspect.isfunction(obj)}

    try:
        target = target.replace(" ", "_")
        target_func = availables[target]
    except KeyError as e:
        print("Invalid fairness measure name:", e)
        return

    # get the predicate definitions
    predicate_target = input("Input the predicate definitions file name: ")
    # predicate_target = "test_predicates2.py"

    # keep the order of the definition by line number for later sorting
    with open(predicate_target, 'r') as file:
        lines = file.readlines()
    string_line_numbers = {}
    for i, line in enumerate(lines, start=1):
        line = line.strip()
        if line:
            string_line_numbers[line] = i
    def get_line_number(s):
        for line, number in string_line_numbers.items():
            if s in line:
                return number

    predicate_target = predicate_target.replace(".py", "")
    try:
        module = importlib.import_module(predicate_target)
    except ImportError as e:
        print("Failed to import predicate file:", e)
        return

    members = inspect.getmembers(module)
    members = [(name, obj) for name, obj in members if not name.startswith('__')]

    # sort the functions by line numbers which is definition order
    functions = [(name, obj) for name, obj in members if inspect.isfunction(obj)]
    functions = sorted(functions, key=lambda x: get_line_number(x[0]))
    functions = list(map(lambda x: x[1], functions))

    variables = [obj for _, obj in members if not inspect.isfunction(obj)]

    # calling the function
    getattr(checker, target)(ratio, *functions, *variables)
    return

if __name__ == "__main__":
    main()
