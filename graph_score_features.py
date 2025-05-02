import torch
import matplotlib.pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

ACT_COUNT_PATH = "dqns/summarystatsonly/s[NUM]/act_counts.pt"
TEST_RUN_PATH = "delta_logs/autoencoder[NUM].out"

runs = [str(i) for i in range(10)]

def count_live_features(path):
    data = torch.load(path, map_location = device)
    return sum([i > 0 for i in data])

def get_average_score(path):
    scores = []
    with open(path, "r") as test_log:
        for line in test_log:
            string = line.strip()
            if "score : " in string:
                scores.append(float(string.split()[-1]))
    return sum(scores)/len(scores)

def main():
    scores = []
    feature_counts = []
    for run in runs:
        scores.append(get_average_score(TEST_RUN_PATH.replace('[NUM]', run)))
        feature_counts.append(count_live_features(ACT_COUNT_PATH.replace('[NUM]', run)))
        print("Run " + run + " has an average score of " + str(int(scores[-1])) + " and a total of " + str(int(feature_counts[-1].item())) + " live features")
    
    estimated = np.polyfit(feature_counts, scores, 1)
    est_value = np.poly1d(estimated)
    estimated_para = np.polyfit(feature_counts, scores, 2)
    est_para_value = lambda lst: [estimated_para[0]*x**2 + estimated_para[1]*x + estimated_para[2] for x in lst]
    
    plt.scatter(feature_counts, scores)
    plt.plot(sorted(feature_counts), est_value(sorted(feature_counts)), "r--")
    plt.plot(range(min(feature_counts), max(feature_counts)+1), est_para_value(range(min(feature_counts), max(feature_counts)+1)), "g--")
    # plt.xlim(0, 2048)
    plt.ylim(0, 2500)
    plt.show()

if __name__ == "__main__":
    main()
