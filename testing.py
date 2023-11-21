from HopfieldNetwork import *
np.random.seed(4200)
patterns = [load_csv(p) for p in PATTERN_NAMES]

oja = HopfieldNetwork((7, 7), OJA)
# print(np.array([patterns[1][i][:40000] for i in range(len(patterns[1]))]))
# oja.load_patterns_and_learn(np.array([patterns[1][i][:160000] for i in range(len(patterns[1]))]), 0.8, 1)
oja.load_patterns_and_learn(patterns[8], 0.8, 100)
score, _ = oja.calc_score()

avg_score, stabilities, _ = oja.report_score()
print(score, avg_score, stabilities)
