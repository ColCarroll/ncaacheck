import os
import numpy
import scipy
import csv
import itertools

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


def score_possible(ordered_team_list=map(str, [1112, 1458, 1246, 1323, 1181, 1211, 1257, 1277]), username="carroll"):
    """
    this is a little brittle:
    ordered_team_list has to be strings of team ids, in order so that everything collapses the right way --
    so if the winner of kentucky vs nd plays the winner of wisco vs az, then you can't put duke in the middle of those
    four.
    """
    outcomes = []
    teams = team_lookup()
    games_left = len(ordered_team_list) - 1
    with open(os.path.join(DATA_DIR, "results.csv"), 'rU') as buff:
        results = [{"id": row["id"], "pred": float(row["pred"])} for row in csv.DictReader(buff)]
    game_ids = [j["id"] for j in results] + ["" for _ in range(games_left)]
    actual = [j["pred"] for j in results] + [0 for _ in range(games_left)]

    predictions = {}
    for filename in get_preds():
        predictions[filename] = get_results(os.path.join(PRED_DIR, filename))

    for possibility in itertools.product("01", repeat=games_left):
        poss = list(map(int, possibility))
        new_ids = []
        wins = []
        teams_left = ordered_team_list[:]
        for idx, j in enumerate(poss):
            team_one, team_two = teams_left[2 * idx: 2 * idx + 2]
            if team_two < team_one:
                j = (j + 1) % 2
                team_one, team_two = team_two, team_one
            if j == 0:
                wteam, lteam = team_two, team_one
            else:
                wteam, lteam = team_one, team_two
            teams_left.append(wteam)
            new_ids.append("2015_{:s}_{:s}".format(team_one, team_two))
            wins.append("{:s} def. {:s}".format(teams[wteam], teams[lteam], new_ids[-1], j))

        game_ids[-games_left:] = new_ids
        actual[-games_left:] = poss

        losses = []
        for filename, preds in predictions.iteritems():
            losses.append((filename, log_loss(actual, [preds[game_id] for game_id in game_ids])))

        team_places = {}
        for j, row in enumerate(sorted(losses, key=lambda j: j[1]), start=1):
            team_name = row[0].split("_")[0].replace("-", " ")
            if team_name not in team_places:
                team_places[team_name] = len(team_places) + 1
            if username in team_name:
                outcomes.append([team_places[team_name]] + wins)
                break
    with open('{:s}_outcomes.txt'.format(username), 'w') as buff:
        for k in sorted(outcomes, key=lambda j: j[0]):
            buff.write("{:d}.\n\t{:s}".format(k[0], "\n\t".join(k[1:])))
            buff.write("\n")


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
    team_places = {}

    with open(os.path.join(DIR, 'readme.md'), 'wb') as buff:
        buff.write('bracket_place|team_place|filename|score\n')
        buff.write('-----|-----|-----|-----\n')
        for j, row in enumerate(score_all(), start=1):
            team_name = row[0].split("_")[0].replace("-", " ")
            if team_name not in team_places:
                team_places[team_name] = len(team_places) + 1
            buff.write("{:s}\n".format("|".join(map(str, [j, team_places[team_name], row[0], row[1]]))))


if __name__ == '__main__':
    # print_results()
    # write_scores()
    score_possible()
