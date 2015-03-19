#!/Users/colinc/anaconda/bin/python
import pandas
import os
import scipy
import csv

DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(DIR, "data")
PRED_DIR = os.path.join(DATA_DIR, "preds")


def get_preds():
    return os.listdir(PRED_DIR)


def log_loss(act, pred):
    epsilon = 1e-15
    pred = scipy.maximum(epsilon, pred)
    pred = scipy.minimum(1 - epsilon, pred)
    ll = sum(act * scipy.log(pred) + scipy.subtract(1, act) * scipy.log(scipy.subtract(1, pred)))
    return ll * -1.0 / len(act)


def score_all():
    result_df = pandas.read_csv(os.path.join(DATA_DIR, 'results.csv'))

    loss_lookup = []
    for filename in get_preds():
        df = pandas.read_csv(os.path.join(PRED_DIR, filename))

        p, r = [], []

        for index, row in result_df.iterrows():
            for index2, row2 in df.loc[df['id'] == row['id']].iterrows():
                if row['id'] != row2['id']:
                    continue
                else:
                    p.append(row2['pred'])
                    r.append(row['result'])
                    break

        loss_lookup.append((filename, log_loss(r, p)))

    return sorted(loss_lookup, key=lambda j: j[1])


def write_scores():
    with open(os.path.join(DIR, 'readme.md'), 'wb') as buff:
        writer = csv.writer(buff, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['index|filename|score'])
        writer.writerow(['-----|-----|-----'])
        for j, row in enumerate(score_all(), start=1):
            writer.writerow([str(j) + '|' + str(row[0]) + '|' + str(row[1])])


if __name__ == '__main__':
    write_scores()
