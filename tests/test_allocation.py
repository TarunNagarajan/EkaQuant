import unittest
import torch
import torch.nn as nn
from ekaquant.selection import select_layers


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Use bfloat16 to match the library's target precision
        self.layer1 = nn.Linear(1024, 1024, dtype=torch.bfloat16)
        self.layer2 = nn.Linear(1024, 1024, dtype=torch.bfloat16)
        self.layer3 = nn.Linear(1024, 1024, dtype=torch.bfloat16)


class TestKnapsackAllocation(unittest.TestCase):
    def setUp(self):
        self.model = MockModel()
        self.sensitivity_map = {"layer1": 100.0, "layer2": 50.0, "layer3": 10.0}

    def test_knapsack_strict_budget(self):
        # 1M params in bf16 = 2MB. Cost to keep vs 4-bit = 1.5MB.
        budget = 2.0

        selected = select_layers(
            model=self.model,
            sensitivity_map=self.sensitivity_map,
            method="knapsack",
            budget_mb=budget,
        )

        self.assertIn("layer1", selected)
        self.assertNotIn("layer2", selected)
        self.assertNotIn("layer3", selected)
        self.assertEqual(len(selected), 1)

    def test_knapsack_generous_budget(self):
        selected = select_layers(
            model=self.model,
            sensitivity_map=self.sensitivity_map,
            method="knapsack",
            budget_mb=100.0,
        )
        self.assertEqual(len(selected), 3)
        self.assertIn("layer1", selected)
        self.assertIn("layer2", selected)
        self.assertIn("layer3", selected)


if __name__ == "__main__":
    unittest.main()
