import unittest

from helmet_logic import associate_helmets, helmet_label, helmet_status_mode


class HelmetLogicTests(unittest.TestCase):
    def test_dataset_classes_enable_paired_status_estimates(self) -> None:
        self.assertEqual(
            helmet_status_mode(["helmet", "motorcycle"]),
            "paired",
        )

    def test_explicit_with_and_without_classes_take_precedence(self) -> None:
        self.assertEqual(
            helmet_status_mode(["with_helmet", "without_helmet"]),
            "explicit",
        )
        self.assertEqual(
            helmet_status_mode(["With Helmet", "Without Helmet"]),
            "explicit",
        )
        self.assertEqual(helmet_label("With Helmet"), "With helmet")
        self.assertEqual(helmet_label("without_helmet"), "No helmet")

    def test_helmet_boxes_are_assigned_to_the_closest_motorcycle(self) -> None:
        motorcycles = [
            (0.0, 100.0, 100.0, 300.0),
            (120.0, 100.0, 220.0, 300.0),
        ]
        helmets = [(42.0, 72.0, 48.0, 88.0)]

        self.assertEqual(
            associate_helmets(motorcycles, helmets),
            [True, False],
        )

    def test_motorcycle_without_a_nearby_helmet_is_flagged(self) -> None:
        motorcycles = [(0.0, 100.0, 100.0, 300.0)]

        self.assertEqual(associate_helmets(motorcycles, []), [False])
        self.assertEqual(
            associate_helmets(motorcycles, [(42.0, 20.0, 48.0, 35.0)]),
            [False],
        )


if __name__ == "__main__":
    unittest.main()
