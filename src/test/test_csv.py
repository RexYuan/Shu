from fairness_checker import *

race = 'Native American'
sex = 'Male'
degree = 'M'
age = 'Greater than 45'

c = fairness_csv_checker("compas-scores-two-years.csv")

c.disparate_impact(0.8, lambda row:                  row['sex'] == sex, lambda row: row['score_text'] in {'Medium', 'High'})
c.demographic_parity(0.2, lambda row:                row['sex'] == sex, lambda row: row['score_text'] in {'Medium', 'High'})
c.equalized_odds(0.2, lambda row:                    row['sex'] == sex, lambda row: row['score_text'] in {'Medium', 'High'}, lambda row: row['is_recid'] == '1')
c.equal_opportunity(0.2, lambda row:                 row['sex'] == sex, lambda row: row['score_text'] in {'Medium', 'High'}, lambda row: row['is_recid'] == '1')
c.accuracy_eqaulity(0.2, lambda row:                 row['sex'] == sex, lambda row: row['score_text'] in {'Medium', 'High'}, lambda row: row['is_recid'] == '1')
c.predictive_parity(0.2, lambda row:                 row['sex'] == sex, lambda row: row['score_text'] in {'Medium', 'High'}, lambda row: row['is_recid'] == '1')
c.equal_calibration(0.2, lambda row:                 row['sex'] == sex, lambda row: row['is_recid'] == '1', lambda u,l: (lambda row: l <= int(row['decile_score']) and int(row['decile_score']) <= u), (7, 5))
c.conditional_statistical_parity(0.2, lambda row:    row['sex'] == sex, lambda row: row['score_text'] in {'Medium', 'High'}, lambda x: (lambda row: int(row['priors_count']) > x), (0,))
c.predictive_equality(0.2, lambda row:               row['sex'] == sex, lambda row: row['score_text'] in {'Medium', 'High'}, lambda row: row['is_recid'] == '1')
c.conditional_use_accuracy_equality(0.2, lambda row: row['sex'] == sex, lambda row: row['score_text'] in {'Medium', 'High'}, lambda row: row['is_recid'] == '1')
c.positive_balance(5, lambda row:                    row['sex'] == sex, lambda row: int(row['decile_score']), lambda row: row['is_recid'] == '1')
c.negative_balance(5, lambda row:                    row['sex'] == sex, lambda row: int(row['decile_score']), lambda row: row['is_recid'] == '1')
c.mean_difference(0.2, lambda row:                   row['sex'] == sex, lambda row: row['score_text'] in {'Medium', 'High'})
