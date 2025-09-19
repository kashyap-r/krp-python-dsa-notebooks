# Prefix-sum practice: functions + unit tests + randomized fuzz checks
# Run: python prefix_sum_test_harness.py

import random
import unittest
from collections import defaultdict

# ---------------------- Problem Implementations ----------------------

def running_sum(nums):
    rs, s = [], 0
    for x in nums:
        s += x
        rs.append(s)
    return rs

class NumArray:
    def __init__(self, nums):
        self.nums = list(nums)
        self.ps = [0]
        for x in nums:
            self.ps.append(self.ps[-1] + x)
    def sumRange(self, l, r):
        return self.ps[r+1] - self.ps[l]

def pivot_index(nums):
    total = sum(nums)
    left = 0
    for i, x in enumerate(nums):
        if left == total - left - x:
            return i
        left += x
    return -1

def subarray_sum(nums, k):
    count = 0
    pref = 0
    freq = defaultdict(int)
    freq[0] = 1
    for x in nums:
        pref += x
        count += freq[pref - k]
        freq[pref] += 1
    return count

def subarrays_div_by_k(nums, K):
    count = 0
    pref = 0
    freq = defaultdict(int)
    freq[0] = 1
    for x in nums:
        pref = (pref + x) % K
        count += freq[pref]
        freq[pref] += 1
    return count

def min_subarray_len(target, nums):
    n = len(nums)
    ans = n + 1
    s = left = 0
    for right, x in enumerate(nums):
        s += x
        while s >= target:
            ans = min(ans, right - left + 1)
            s -= nums[left]
            left += 1
    return 0 if ans == n + 1 else ans

def range_add(n, updates):
    diff = [0]*(n+1)
    for l, r, inc in updates:
        if l < 0 or r >= n or l > r:
            raise ValueError("Invalid update interval")
        diff[l] += inc
        diff[r+1] -= inc
    res = [0]*n
    run = 0
    for i in range(n):
        run += diff[i]
        res[i] = run
    return res

class NumMatrix:
    def __init__(self, mat):
        if not mat or not mat[0]:
            self.ps = [[0]]
            return
        m, n = len(mat), len(mat[0])
        ps = [[0]*(n+1) for _ in range(m+1)]
        for i in range(1, m+1):
            row_sum = 0
            for j in range(1, n+1):
                row_sum += mat[i-1][j-1]
                ps[i][j] = ps[i-1][j] + row_sum
        self.ps = ps
    def sumRegion(self, r1, c1, r2, c2):
        ps = self.ps
        return ps[r2+1][c2+1] - ps[r1][c2+1] - ps[r2+1][c1] + ps[r1][c1]

def num_submatrix_sum_target(mat, target):
    if not mat or not mat[0]:
        return 0
    m, n = len(mat), len(mat[0])
    row_ps = [[0]*(n+1) for _ in range(m)]
    for i in range(m):
        for j in range(n):
            row_ps[i][j+1] = row_ps[i][j] + mat[i][j]
    ans = 0
    for c1 in range(n):
        for c2 in range(c1, n):
            freq = defaultdict(int)
            freq[0] = 1
            cur = 0
            for r in range(m):
                cur += row_ps[r][c2+1] - row_ps[r][c1]
                ans += freq[cur - target]
                freq[cur] += 1
    return ans

def can_three_parts_equal_sum(nums):
    total = sum(nums)
    if total % 3 != 0:
        return False
    target = total // 3
    parts, s = 0, 0
    for x in nums:
        s += x
        if s == target:
            parts += 1
            s = 0
    return parts >= 3

# ---------------------- Brute-force Helpers for Fuzz ----------------------

def brute_pivot_index(nums):
    for i in range(len(nums)):
        if sum(nums[:i]) == sum(nums[i+1:]):
            return i
    return -1

def brute_subarray_sum(nums, k):
    n = len(nums)
    cnt = 0
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += nums[j]
            if s == k:
                cnt += 1
    return cnt

def brute_div_by_k(nums, K):
    n = len(nums)
    cnt = 0
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += nums[j]
            if s % K == 0:
                cnt += 1
    return cnt

def brute_min_subarray_len(target, nums):
    n = len(nums)
    best = n + 1
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += nums[j]
            if s >= target:
                best = min(best, j - i + 1)
                break
    return 0 if best == n + 1 else best

def brute_range_add(n, updates):
    arr = [0]*n
    for l, r, inc in updates:
        for i in range(l, r+1):
            arr[i] += inc
    return arr

def brute_sumRegion(mat, r1, c1, r2, c2):
    return sum(mat[i][c1:c2+1] for i in range(r1, r2+1))

def brute_num_submatrix_sum_target(mat, target):
    if not mat or not mat[0]:
        return 0
    m, n = len(mat), len(mat[0])
    cnt = 0
    for r1 in range(m):
        for r2 in range(r1, m):
            for c1 in range(n):
                for c2 in range(c1, n):
                    if brute_sumRegion(mat, r1, c1, r2, c2) == target:
                        cnt += 1
    return cnt

def brute_three_parts(nums):
    n = len(nums)
    for i in range(n-2):
        for j in range(i+1, n-1):
            s1 = sum(nums[:i+1])
            s2 = sum(nums[i+1:j+1])
            s3 = sum(nums[j+1:])
            if s1 == s2 == s3:
                return True
    return False

# ---------------------- Unit Tests ----------------------

class TestPrefixSuite(unittest.TestCase):
    def test_running_sum(self):
        self.assertEqual(running_sum([]), [])
        self.assertEqual(running_sum([1]), [1])
        self.assertEqual(running_sum([1,2,3,4]), [1,3,6,10])
        self.assertEqual(running_sum([3,-1,4,-2,5]), [3,2,6,4,9])

    def test_numarray(self):
        arr = [3, 1, 4, 2, 5]
        obj = NumArray(arr)
        self.assertEqual(obj.sumRange(0, 4), sum(arr))
        self.assertEqual(obj.sumRange(1, 3), 7)
        self.assertEqual(obj.sumRange(2, 2), 4)

    def test_pivot_index(self):
        self.assertEqual(pivot_index([1,7,3,6,5,6]), 3)
        self.assertEqual(pivot_index([1,2,3]), -1)
        self.assertEqual(pivot_index([2,1,-1]), 0)

    def test_subarray_sum_equals_k(self):
        self.assertEqual(subarray_sum([1,1,1], 2), 2)
        self.assertEqual(subarray_sum([1,2,3], 3), 2)
        self.assertEqual(subarray_sum([1,-1,0], 0), 3)

    def test_subarrays_div_by_k(self):
        self.assertEqual(subarrays_div_by_k([4,5,0,-2,-3,1], 5), 7)
        self.assertEqual(subarrays_div_by_k([5], 9), 0)
        self.assertEqual(subarrays_div_by_k([1,2,3], 3), 4)

    def test_min_subarray_len(self):
        self.assertEqual(min_subarray_len(7, [2,3,1,2,4,3]), 2)
        self.assertEqual(min_subarray_len(4, [1,4,4]), 1)
        self.assertEqual(min_subarray_len(11, [1,1,1,1,1,1,1,1]), 0)

    def test_range_add(self):
        self.assertEqual(range_add(5, [(1,3,2),(2,4,3),(0,2,-2)]), [-2,0,3,5,3])
        self.assertEqual(range_add(3, []), [0,0,0])

    def test_nummatrix(self):
        mat = [
            [3, 0, 1, 4, 2],
            [5, 6, 3, 2, 1],
            [1, 2, 0, 1, 5],
            [4, 1, 0, 1, 7],
            [1, 0, 3, 0, 5]
        ]
        nm = NumMatrix(mat)
        self.assertEqual(nm.sumRegion(2,1,4,3), 8)
        self.assertEqual(nm.sumRegion(1,1,2,2), 11)
        self.assertEqual(nm.sumRegion(1,2,2,4), 12)

    def test_num_submatrix_sum_target(self):
        mat = [[0,1,0],[1,1,1],[0,1,0]]
        self.assertEqual(num_submatrix_sum_target(mat, 0), 4)
        self.assertEqual(num_submatrix_sum_target(mat, 2), 8)

    def test_three_parts(self):
        self.assertTrue(can_three_parts_equal_sum([0,2,1,-6,6,-7,9,1,2,0,1]))
        self.assertFalse(can_three_parts_equal_sum([0,2,1,-6,6,7,9,-1,2,0,1]))
        self.assertTrue(can_three_parts_equal_sum([0,0,0,0]))

# ---------------------- Randomized Fuzz Tests ----------------------

class TestFuzz(unittest.TestCase):
    def test_fuzz_all(self):
        random.seed(1337)

        # 1) running sum
        for _ in range(50):
            n = random.randint(0, 20)
            arr = [random.randint(-5, 5) for _ in range(n)]
            self.assertEqual(running_sum(arr), [sum(arr[:i+1]) for i in range(n)])

        # 2) NumArray vs brute
        for _ in range(30):
            n = random.randint(0, 20)
            arr = [random.randint(-10, 10) for _ in range(n)]
            obj = NumArray(arr)
            for _ in range(10):
                if n == 0:
                    continue
                l = random.randint(0, n-1)
                r = random.randint(l, n-1)
                self.assertEqual(obj.sumRange(l, r), sum(arr[l:r+1]))

        # 3) pivot index vs brute
        for _ in range(50):
            n = random.randint(0, 20)
            arr = [random.randint(-5, 5) for _ in range(n)]
            self.assertEqual(pivot_index(arr), brute_pivot_index(arr))

        # 4) subarray sum equals k vs brute
        for _ in range(50):
            n = random.randint(0, 25)
            arr = [random.randint(-5, 5) for _ in range(n)]
            k = random.randint(-5, 5)
            self.assertEqual(subarray_sum(arr, k), brute_subarray_sum(arr, k))

        # 5) subarrays divisible by K vs brute
        for _ in range(40):
            n = random.randint(0, 20)
            arr = [random.randint(-6, 6) for _ in range(n)]
            K = random.randint(1, 6)  # positive K
            self.assertEqual(subarrays_div_by_k(arr, K), brute_div_by_k(arr, K))

        # 6) min subarray len (positives) vs brute
        for _ in range(40):
            n = random.randint(0, 25)
            arr = [random.randint(1, 10) for _ in range(n)]  # positives only
            target = random.randint(1, 30)
            self.assertEqual(min_subarray_len(target, arr), brute_min_subarray_len(target, arr))

        # 7) range add vs brute
        for _ in range(40):
            n = random.randint(1, 20)
            q = random.randint(0, 20)
            updates = []
            for __ in range(q):
                l = random.randint(0, n-1)
                r = random.randint(l, n-1)
                inc = random.randint(-5, 5)
                updates.append((l, r, inc))
            self.assertEqual(range_add(n, updates), brute_range_add(n, updates))

        # 8) NumMatrix vs brute
        for _ in range(20):
            m = random.randint(1, 5)
            n = random.randint(1, 5)
            mat = [[random.randint(-3, 3) for _ in range(n)] for __ in range(m)]
            nm = NumMatrix(mat)
            for __ in range(10):
                r1 = random.randint(0, m-1)
                r2 = random.randint(r1, m-1)
                c1 = random.randint(0, n-1)
                c2 = random.randint(c1, n-1)
                self.assertEqual(nm.sumRegion(r1, c1, r2, c2), brute_sumRegion(mat, r1, c1, r2, c2))

        # 9) num_submatrix_sum_target vs brute (small sizes)
        for _ in range(10):
            m = random.randint(1, 5)
            n = random.randint(1, 5)
            mat = [[random.randint(-2, 2) for _ in range(n)] for __ in range(m)]
            target = random.randint(-3, 3)
            self.assertEqual(num_submatrix_sum_target(mat, target), brute_num_submatrix_sum_target(mat, target))

        # 10) three equal parts vs brute
        for _ in range(50):
            n = random.randint(0, 20)
            arr = [random.randint(-5, 5) for _ in range(n)]
            self.assertEqual(can_three_parts_equal_sum(arr), brute_three_parts(arr))

if __name__ == '__main__':
    random.seed(1337)
    unittest.main(verbosity=2)
