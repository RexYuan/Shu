def privileged_predicate(row):
    return row['sex'] == 'Male'

def truth_predicate(row):
    return row['is_recid'] == '1'

def calib_predicate_h(u, l):
    def tmp(row):
        return l <= int(row['decile_score']) and int(row['decile_score']) <= u
    return tmp

calib_arg = (7, 5)
