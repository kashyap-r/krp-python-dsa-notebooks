"""
A trucking company has been having issues with late deliveries
and has asked you to investigate. They have provided you 3
logs:
- (truck, package) pairs
- (package, expected delivery time, destination) triples
- (truck, location, arrival time) triples

Using this data, write a function to return which was the
latest truck.

Example:
truck_packages:
(T1, P1)
(T1, P2)
(T2, P3)
(T2, P4)
(T3, P5)
(T3, P6)
(T4, P7)
(T1, P8)

package_info:
(P1, 10, Seattle)
(P2, 12, Portland)
(P3, 9, LA)
(P4, 11, Seattle)
(P5, 13, Portland)
(P6, 15, LA)
(P7, 14, Seattle)
(P8, 9, Portland)

arrival_logs:
(T1, Seattle,  10)
(T1, Portland, 14)
(T2, LA,       10)
(T2, Seattle,  11)
(T3, Portland, 13)
(T3, LA,       18)
(T4, Seattle,  20)

Returns: T1

Complexity Analysis variables:
t = Number of trucks
p = Number of packages
d = Number of destinations
----
"""
import unittest


def find_latest_truck(truck_packages, package_info, arrival_logs):
    # TODO: Implement this function
    # Reference implementation added below the original prompt.
    if not truck_packages or not package_info or not arrival_logs:
        return None

    details = {package: (due, destination)
               for package, due, destination in package_info}
    earliest = {}
    for truck, location, time in arrival_logs:
        stop = (truck, location)
        if stop not in earliest or time < earliest[stop]:
            earliest[stop] = time

    scores = {}
    for truck, package in truck_packages:
        due, destination = details[package]
        actual = earliest.get((truck, destination))
        if actual is not None and actual > due:
            count, hours = scores.get(truck, (0, 0))
            scores[truck] = (count + 1, hours + actual - due)

    return max(scores, key=scores.get, default=None)


def find_latest_truck_naive(truck_packages, package_info, arrival_logs):
    """Scan the two other logs separately for each assigned package."""
    if not truck_packages or not package_info or not arrival_logs:
        return None

    scores = {}
    for truck, package in truck_packages:
        for package_id, due, destination in package_info:
            if package_id == package:
                break
        else:
            raise KeyError(package)

        actual = None
        for truck_id, location, time in arrival_logs:
            if truck_id == truck and location == destination:
                if actual is None or time < actual:
                    actual = time

        if actual is not None and actual > due:
            count, hours = scores.get(truck, (0, 0))
            scores[truck] = (count + 1, hours + actual - due)

    return max(scores, key=scores.get, default=None)


class TestLatestTruck(unittest.TestCase):
    def test_example_case(self):
        truck_packages = [
            ("T1", "P1"),
            ("T1", "P2"),
            ("T2", "P3"),
            ("T2", "P4"),
            ("T3", "P5"),
            ("T3", "P6"),
            ("T4", "P7"),
            ("T1", "P8"),
        ]
        package_info = [
            ("P1", 10, "Seattle"),
            ("P2", 12, "Portland"),
            ("P3", 9, "LA"),
            ("P4", 11, "Seattle"),
            ("P5", 13, "Portland"),
            ("P6", 15, "LA"),
            ("P7", 14, "Seattle"),
            ("P8", 9, "Portland"),
        ]
        arrival_logs = [
            ("T1", "Seattle", 10),
            ("T1", "Portland", 14),
            ("T2", "LA", 10),
            ("T2", "Seattle", 11),
            ("T3", "Portland", 13),
            ("T3", "LA", 18),
            ("T4", "Seattle", 20)
        ]
        result = find_latest_truck(truck_packages, package_info, arrival_logs)
        self.assertEqual(result, "T1")

    # Add new tests here

    def test_multiple_packages_same_destination_same_truck(self):
        truck_packages = [
            ("T1", "P1"),
            ("T1", "P2"),
            ("T2", "P3")
        ]
        package_info = [
            ("P1", 10, "Seattle"),
            ("P2", 12, "Seattle"),
            ("P3", 9, "Portland")
        ]
        arrival_logs = [
            ("T1", "Seattle", 13),
            ("T2", "Portland", 20)
        ]
        result = find_latest_truck(truck_packages, package_info, arrival_logs)
        self.assertEqual(result, "T1")

    def test_undelivered_package(self):
        truck_packages = [("T1", "P1"), ("T2", "P2")]
        package_info = [("P1", 10, "Seattle"), ("P2", 9, "Portland")]
        arrival_logs = [("T1", "Seattle", 12)]
        result = find_latest_truck(truck_packages, package_info, arrival_logs)
        self.assertEqual(result, "T1")

    def test_empty_logs(self):
        result = find_latest_truck([], [], [])
        self.assertIsNone(result)

    def test_tie_broken_by_total_hours(self):
        truck_packages = [("T2", "P1"), ("T1", "P2")]
        package_info = [("P1", 10, "Seattle"), ("P2", 10, "Portland")]
        arrival_logs = [
            ("T2", "Seattle", 13),
            ("T1", "Portland", 12)
        ]
        result = find_latest_truck(truck_packages, package_info, arrival_logs)
        self.assertEqual(result, "T2")

    def test_full_tie(self):
        truck_packages = [("T1", "P1"), ("T2", "P2")]
        package_info = [("P1", 10, "Seattle"), ("P2", 10, "Portland")]
        arrival_logs = [
            ("T1", "Seattle", 12),
            ("T2", "Portland", 12)
        ]
        result = find_latest_truck(truck_packages, package_info, arrival_logs)
        self.assertIn(result, {"T1", "T2"})

    def test_multiple_visits_same_location(self):
        truck_packages = [("T1", "P1"), ("T1", "P2"), ("T2", "P3")]
        package_info = [
            ("P1", 10, "Seattle"),
            ("P2", 12, "Seattle"),
            ("P3", 13, "Seattle")
        ]
        arrival_logs = [("T1", "Seattle", 14), ("T1", "Seattle", 10), ("T2", "Seattle", 14)]
        result = find_latest_truck(truck_packages, package_info, arrival_logs)
        self.assertEqual(result, "T2")

    def test_all_on_time(self):
        truck_packages = [("T1", "P1"), ("T2", "P2")]
        package_info = [("P1", 10, "Seattle"), ("P2", 9, "LA")]
        arrival_logs = [("T1", "Seattle", 10), ("T2", "LA", 9)]
        result = find_latest_truck(truck_packages, package_info, arrival_logs)
        self.assertIsNone(result)

    def test_null_inputs(self):
        truck_packages = None
        package_info = None
        arrival_logs = None
        result = find_latest_truck(truck_packages, package_info, arrival_logs)
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
