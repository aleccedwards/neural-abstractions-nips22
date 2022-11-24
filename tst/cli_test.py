import unittest
import cli

class testCLI(unittest.TestCase):
    def test_initial(self):
        args = cli.parse_command(["-c", "config.yaml", "--initial", "[[0.45, -0.6], [0.45, -0.55], [0.5, -0.6], [0.5, -0.55]]"])
        self.assertEqual(args.initial, [[0.45, -0.6], [0.45, -0.55], [0.5, -0.6], [0.5, -0.55]])

if __name__ == "__main__":
    unittest.main()