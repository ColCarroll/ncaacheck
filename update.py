import os
import numpy
import scipy
import csv

DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(DIR, "data")
PRED_DIR = os.path.join(DATA_DIR, "preds")


def get_preds():
    return os.listdir(PRED_DIR)


def log_loss(actual, predicted):
    epsilon = 1e-15
    actual = numpy.array(actual)
    predicted = scipy.minimum(1 - epsilon, scipy.maximum(epsilon, numpy.array(predicted)))
    return -(actual * scipy.log(predicted) + (1 - actual) * scipy.log(1 - predicted)).mean()


def get_results(filename):
    with open(filename, 'rU') as buff:
        results = {row["id"]: float(row["pred"]) for row in csv.DictReader(buff)}
    return results


def score_all():
    results = get_results(os.path.join(DATA_DIR, "results.csv"))
    losses = []
    game_ids, actual = zip(*results.items())
    for filename in get_preds():
        file_preds = get_results(os.path.join(PRED_DIR, filename))
        predictions = [file_preds[game_id] for game_id in game_ids]
        losses.append((filename, log_loss(actual, predictions)))
    return sorted(losses, key=lambda j: j[1])


def team_lookup():
    with open(os.path.join(DATA_DIR, "teams.csv")) as buff:
        return {row["team_id"]: row["team_name"] for row in csv.DictReader(buff)}


def print_results():
    teams = team_lookup()
    with open(os.path.join(DATA_DIR, "results.csv")) as buff:
        next(buff)
        for j, row in enumerate(buff, start=1):
            game_id, result = row.strip().split(",")
            _, team_one, team_two = game_id.split("_")
            result = int(result)
            outcome = "beat" if result == 1 else "lost to"
            print("{:d}. {:s} {:s} {:s}".format(j, teams[team_one], outcome, teams[team_two]))


def write_scores():
    with open(os.path.join(DIR, 'readme.md'), 'wb') as buff:
        buff.write('index|filename|score\n')
        buff.write('-----|-----|-----\n')
        buff.write("\n".join(
            ["{:s}".format("|".join(map(str, [j, row[0], row[1]]))) for j, row in enumerate(score_all(), start=1)]))


if __name__ == '__main__':
    print_results()
    write_scores()
