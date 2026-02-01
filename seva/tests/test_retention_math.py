import math
import unittest


def decay_factor(age_days: float, decay_rate: float) -> float:
    if decay_rate <= 0.0:
        return 1.0
    return float(math.exp(-decay_rate * max(0.0, age_days)))


def adjusted_score(similarity: float, age_days: float, reinforcement: float, decay_rate: float, reinforcement_boost: float) -> float:
    base = float(similarity) * decay_factor(age_days, decay_rate)
    return base + (float(reinforcement_boost) * float(reinforcement))


def prune_sort_key(md: dict, policy: str):
    ts = md.get("timestamp")
    tsf = float(ts) if isinstance(ts, (int, float)) else 0.0
    reinf = md.get("reinforcement")
    rf = float(reinf) if isinstance(reinf, (int, float)) else 0.0
    if policy == "least_reinforced":
        return (rf, tsf)
    return (tsf, rf)


class TestRetentionMath(unittest.TestCase):
    def test_decay_factor_zero_rate(self):
        self.assertEqual(decay_factor(100.0, 0.0), 1.0)

    def test_decay_factor_increases_with_recency(self):
        self.assertGreater(decay_factor(1.0, 0.1), decay_factor(10.0, 0.1))

    def test_adjusted_score_reinforcement_boost(self):
        s0 = adjusted_score(0.5, 0.0, 0.0, 0.0, 0.1)
        s1 = adjusted_score(0.5, 0.0, 2.0, 0.0, 0.1)
        self.assertAlmostEqual(s1 - s0, 0.2, places=6)

    def test_prune_sort_oldest(self):
        a = {"timestamp": 10, "reinforcement": 0}
        b = {"timestamp": 20, "reinforcement": 0}
        self.assertLess(prune_sort_key(a, "oldest"), prune_sort_key(b, "oldest"))

    def test_prune_sort_least_reinforced(self):
        a = {"timestamp": 20, "reinforcement": 0}
        b = {"timestamp": 10, "reinforcement": 1}
        self.assertLess(prune_sort_key(a, "least_reinforced"), prune_sort_key(b, "least_reinforced"))


if __name__ == "__main__":
    unittest.main()
