# import pytest 
# import sys
# import os 

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# from range_sum_query import NumArray

# # 1. Define a shared sample input array for testing
# @pytest.fixture
# def sample_arr():
#     return [1, 2, 3, 4, 5]


# # 2. Test the sumRange method with various queries
# def test_sum_range(sample_arr):
#     num_array = NumArray(sample_arr)

#     # Test case 1: Query the sum from index 0 to 2 (1 + 2 + 3)
#     assert num_array.sumRange(0, 2) == 6

#     # Test case 2: Query the sum from index 1 to 3 (2 + 3 + 4)
#     assert num_array.sumRange(1, 3) == 9

#     # Test case 3: Query the sum from index 0 to 4 (1 + 2 + 3 + 4 + 5)
#     assert num_array.sumRange(0, 4) == 15

#     # Test case 4: Query the sum from index 2 to 4 (3 + 4 + 5)
#     assert num_array.sumRange(2, 4) == 12

#     # Test case 5: Query the sum from index 3 to 3 (single element)
#     assert num_array.sumRange(3, 3) == 4


import sys
import os
import pytest

# Append the absolute path of the 'src' directory to sys.path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from src.range_sum_query import RangeSumQuery  # type: ignore

# =====================================================================
# PYTEST FIXTURES (Shared Context Setup)
# =====================================================================

@pytest.fixture
def mixed_array():
    """Fixture for standard testing (Case 6)."""
    return [-2, 0, 3, -5, 2, -1]

@pytest.fixture
def query_engine_mixed(mixed_array):
    """Instantiated RangeSumQuery instance for mixed array tests."""
    return RangeSumQuery(mixed_array)


# =====================================================================
# COMPREHENSIVE TESTS COVERING CASES 1 - 11
# =====================================================================

def test_case_1_empty_array():
    """1. If the input array is empty, return 0 for any query."""
    query_engine = RangeSumQuery([])
    assert query_engine.sumRange(0, 0) == 0
    assert query_engine.sumRange(0, 5) == 0


def test_case_2_same_left_right():
    """2. If left and right are the same, return the value at that index."""
    nums = [10, -5, 20, 30]
    query_engine = RangeSumQuery(nums)
    assert query_engine.sumRange(1, 1) == -5
    assert query_engine.sumRange(3, 3) == 30


def test_case_3_single_element_array():
    """3. If the input array contains only one element, return that element for any query."""
    query_engine = RangeSumQuery([42])
    assert query_engine.sumRange(0, 0) == 42


def test_case_4_all_negative_numbers():
    """4. If the input array contains all negative numbers, the sum can be negative."""
    nums = [-2, -3, -1, -5]
    query_engine = RangeSumQuery(nums)
    assert query_engine.sumRange(0, 2) == -6
    assert query_engine.sumRange(1, 3) == -9


def test_case_5_all_positive_numbers():
    """5. If the input array contains all positive numbers, the sum will always be non-negative."""
    nums = [2, 3, 4, 5]
    query_engine = RangeSumQuery(nums)
    assert query_engine.sumRange(1, 3) >= 0
    assert query_engine.sumRange(1, 3) == 12


def test_case_6_mix_positive_and_negative(query_engine_mixed):
    """6. If the input array contains a mix, the sum can be positive, negative, or zero."""
    assert query_engine_mixed.sumRange(0, 2) == 1   # Positive sum (-2 + 0 + 3)
    assert query_engine_mixed.sumRange(2, 5) == -1  # Negative sum (3 + -5 + 2 + -1)


def test_case_7_zeros_in_array():
    """7. If the input array contains zeros, the sum can be zero for certain ranges."""
    nums = [0, 0, 5, -5, 0]
    query_engine = RangeSumQuery(nums)
    assert query_engine.sumRange(0, 1) == 0
    assert query_engine.sumRange(2, 3) == 0  # 5 + (-5) = 0


def test_case_8_large_numbers_no_overflow():
    """8. If the input array contains large numbers, ensure that the sum does not overflow."""
    large_val = sys.maxsize
    nums = [large_val, large_val, -large_val]
    query_engine = RangeSumQuery(nums)
    # Python automatically scales integers to arbitrary precision
    assert query_engine.sumRange(0, 1) == large_val * 2
    assert query_engine.sumRange(0, 2) == large_val


def test_case_9_duplicate_numbers():
    """9. If the input array contains duplicate numbers, the sum includes all occurrences."""
    nums = [5, 5, 5, 10]
    query_engine = RangeSumQuery(nums)
    assert query_engine.sumRange(0, 2) == 15


def test_case_10_sorted_array():
    """10. If the input array is sorted, the sum is calculated correctly."""
    nums = [1, 3, 5, 7]
    query_engine = RangeSumQuery(nums)
    assert query_engine.sumRange(1, 3) == 15  # 3 + 5 + 7


def test_case_11_boundary_indices():
    """11. If the left and right indices are at the boundaries (0 and len(nums)-1), return full sum."""
    nums = [10, 20, 30, 40]
    query_engine = RangeSumQuery(nums)
    assert query_engine.sumRange(0, len(nums) - 1) == 100
