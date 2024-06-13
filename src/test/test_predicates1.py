def privileged_predicate(row):
    return row['sex'] == 'Male'

def score_predicate(row):
    return int(row['decile_score'])

def truth_predicate(row):
    return row['is_recid'] == '1'
