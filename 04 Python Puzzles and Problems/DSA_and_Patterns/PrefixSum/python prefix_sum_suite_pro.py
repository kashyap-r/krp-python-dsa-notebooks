# Prefix-sum practice suite (Pro): unit tests + fuzz + optional benchmarks
# Usage:
#   python prefix_sum_suite_pro.py [--seed 1337] [--fuzz-cases 50] [--no-fuzz]
#                                  [--only running_sum,subarray_sum,...]
#                                  [--bench] [--bench-hard] [--bench-n 200000]
#                                  [--verbosity 2]
#
# Problems keys for --only:
#   running_sum, numarray, pivot_index, subarray_sum, div_by_k,
#   min_subarray_len, range_add, nummatrix, num_submatrix_target, three_parts
#
# Notes:
# - Unit tests always run for selected problems.
# - Fuzz tests can be disabled with --no-fuzz or scaled with --fuzz-cases.
# - Benchmarks are synthetic and single-threaded; they skip the heaviest 2D-count
#   unless --bench-hard is provided.

import argparse
import random
import time
import unittest
from collections import defaultdict

# ---------------------- Implementations ----------------------

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

# ---------------------- Globals for filtering ----------------------

ALL_KEYS = {
    'running_sum', 'numarray', 'pivot_index', 'subarray_sum', 'div_by_k',
    'min_subarray_len', 'range_add', 'nummatrix', 'num_submatrix_target', 'three_parts'
}
ONLY = set()  # filled by CLI

def should_run(key):
    return (not ONLY) or (key in ONLY)

# ---------------------- Brute helpers ----------------------

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

class TestRunningSum(unittest.TestCase):
    def test_running_sum(self):
        if not should_run('running_sum'): self.skipTest('filtered')
        self.assertEqual(running_sum([]), [])
        self.assertEqual(running_sum([1]), [1])
        self.assertEqual(running_sum([1,2,3,4]), [1,3,6,10])
        self.assertEqual(running_sum([3,-1,4,-2,5]), [3,2,6,4,9])

class TestNumArray(unittest.TestCase):
    def test_numarray(self):
        if not should_run('numarray'): self.skipTest('filtered')
        arr = [3, 1, 4, 2, 5]
        obj = NumArray(arr)
        self.assertEqual(obj.sumRange(0, 4), sum(arr))
        self.assertEqual(obj.sumRange(1, 3), 7)
        self.assertEqual(obj.sumRange(2, 2), 4)

class TestPivotIndex(unittest.TestCase):
    def test_pivot_index(self):
        if not should_run('pivot_index'): self.skipTest('filtered')
        self.assertEqual(pivot_index([1,7,3,6,5,6]), 3)
        self.assertEqual(pivot_index([1,2,3]), -1)
        self.assertEqual(pivot_index([2,1,-1]), 0)

class TestSubarraySum(unittest.TestCase):
    def test_subarray_sum_equals_k(self):
        if not should_run('subarray_sum'): self.skipTest('filtered')
        self.assertEqual(subarray_sum([1,1,1], 2), 2)
        self.assertEqual(subarray_sum([1,2,3], 3), 2)
        self.assertEqual(subarray_sum([1,-1,0], 0), 3)

class TestDivByK(unittest.TestCase):
    def test_subarrays_div_by_k(self):
        if not should_run('div_by_k'): self.skipTest('filtered')
        self.assertEqual(subarrays_div_by_k([4,5,0,-2,-3,1], 5), 7)
        self.assertEqual(subarrays_div_by_k([5], 9), 0)
        self.assertEqual(subarrays_div_by_k([1,2,3], 3), 4)

class TestMinLen(unittest.TestCase):
    def test_min_subarray_len(self):
        if not should_run('min_subarray_len'): self.skipTest('filtered')
        self.assertEqual(min_subarray_len(7, [2,3,1,2,4,3]), 2)
        self.assertEqual(min_subarray_len(4, [1,4,4]), 1)
        self.assertEqual(min_subarray_len(11, [1,1,1,1,1,1,1,1]), 0)

class TestRangeAdd(unittest.TestCase):
    def test_range_add(self):
        if not should_run('range_add'): self.skipTest('filtered')
        self.assertEqual(range_add(5, [(1,3,2),(2,4,3),(0,2,-2)]), [-2,0,3,5,3])
        self.assertEqual(range_add(3, []), [0,0,0])

class TestNumMatrix(unittest.TestCase):
    def test_nummatrix(self):
        if not should_run('nummatrix'): self.skipTest('filtered')
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

class TestNumSubmatrixTarget(unittest.TestCase):
    def test_num_submatrix_sum_target(self):
        if not should_run('num_submatrix_target'): self.skipTest('filtered')
        mat = [[0,1,0],[1,1,1],[0,1,0]]
        self.assertEqual(num_submatrix_sum_target(mat, 0), 4)
        self.assertEqual(num_submatrix_sum_target(mat, 2), 8)

class TestThreeParts(unittest.TestCase):
    def test_three_parts(self):
        if not should_run('three_parts'): self.skipTest('filtered')
        self.assertTrue(can_three_parts_equal_sum([0,2,1,-6,6,-7,9,1,2,0,1]))
        self.assertFalse(can_three_parts_equal_sum([0,2,1,-6,6,7,9,-1,2,0,1]))
        self.assertTrue(can_three_parts_equal_sum([0,0,0,0]))

# ---------------------- Fuzz Tests (separate per-problem) ----------------------

class TestFuzz(unittest.TestCase):
    FUZZ = 50
    SEED = 1337

    def setUp(self):
        random.seed(self.SEED)

    def test_running_sum_fuzz(self):
        if not should_run('running_sum'): self.skipTest('filtered')
        for _ in range(self.FUZZ):
            n = random.randint(0, 25)
            arr = [random.randint(-10, 10) for _ in range(n)]
            self.assertEqual(running_sum(arr), [sum(arr[:i+1]) for i in range(n)])

    def test_numarray_fuzz(self):
        if not should_run('numarray'): self.skipTest('filtered')
        for _ in range(self.FUZZ):
            n = random.randint(0, 25)
            arr = [random.randint(-10, 10) for _ in range(n)]
            obj = NumArray(arr)
            if n == 0: 
                continue
            for __ in range(10):
                l = random.randint(0, n-1)
                r = random.randint(l, n-1)
                self.assertEqual(obj.sumRange(l, r), sum(arr[l:r+1]))

    def test_pivot_index_fuzz(self):
        if not should_run('pivot_index'): self.skipTest('filtered')
        for _ in range(self.FUZZ):
            n = random.randint(0, 25)
            arr = [random.randint(-10, 10) for _ in range(n)]
            self.assertEqual(pivot_index(arr), brute_pivot_index(arr))

    def test_subarray_sum_fuzz(self):
        if not should_run('subarray_sum'): self.skipTest('filtered')
        for _ in range(self.FUZZ):
            n = random.randint(0, 20)
            arr = [random.randint(-5, 5) for _ in range(n)]
            k = random.randint(-5, 5)
            self.assertEqual(subarray_sum(arr, k), brute_subarray_sum(arr, k))

    def test_div_by_k_fuzz(self):
        if not should_run('div_by_k'): self.skipTest('filtered')
        for _ in range(self.FUZZ):
            n = random.randint(0, 20)
            arr = [random.randint(-6, 6) for _ in range(n)]
            K = random.randint(1, 6)  # positive K
            self.assertEqual(subarrays_div_by_k(arr, K), brute_div_by_k(arr, K))

    def test_min_len_fuzz(self):
        if not should_run('min_subarray_len'): self.skipTest('filtered')
        for _ in range(self.FUZZ):
            n = random.randint(0, 25)
            arr = [random.randint(1, 10) for _ in range(n)]  # positives only
            target = random.randint(1, 30)
            self.assertEqual(min_subarray_len(target, arr), brute_min_subarray_len(target, arr))

    def test_range_add_fuzz(self):
        if not should_run('range_add'): self.skipTest('filtered')
        for _ in range(self.FUZZ):
            n = random.randint(1, 20)
            q = random.randint(0, 20)
            updates = []
            for __ in range(q):
                l = random.randint(0, n-1)
                r = random.randint(l, n-1)
                inc = random.randint(-5, 5)
                updates.append((l, r, inc))
            self.assertEqual(range_add(n, updates), brute_range_add(n, updates))

    def test_nummatrix_fuzz(self):
        if not should_run('nummatrix'): self.skipTest('filtered')
        for _ in range(self.FUZZ // 2):
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

    def test_num_submatrix_target_fuzz(self):
        if not should_run('num_submatrix_target'): self.skipTest('filtered')
        for _ in range(max(1, self.FUZZ // 5)):
            m = random.randint(1, 5)
            n = random.randint(1, 5)
            mat = [[random.randint(-2, 2) for _ in range(n)] for __ in range(m)]
            target = random.randint(-3, 3)
            self.assertEqual(num_submatrix_sum_target(mat, target), brute_num_submatrix_sum_target(mat, target))

    def test_three_parts_fuzz(self):
        if not should_run('three_parts'): self.skipTest('filtered')
        for _ in range(self.FUZZ):
            n = random.randint(0, 25)
            arr = [random.randint(-5, 5) for _ in range(n)]
            self.assertEqual(can_three_parts_equal_sum(arr), brute_three_parts(arr))

# ---------------------- Benchmarks ----------------------

def bench(title, fn, *args, repeats=1):
    t0 = time.perf_counter()
    out = None
    for _ in range(repeats):
        out = fn(*args)
    t1 = time.perf_counter()
    dt = t1 - t0
    per = dt / repeats if repeats > 0 else 0.0
    print(f"[BENCH] {title:<28s} time={per:.4f}s repeats={repeats}")
    return out, per

def run_benchmarks(seed=1337, bench_n=200000, bench_hard=False):
    random.seed(seed)
    print(f"== Benchmarks (seed={seed}, n={bench_n}, hard={bench_hard}) ==")

    # 1) running_sum
    arr = [random.randint(-10, 10) for _ in range(bench_n)]
    bench("running_sum", running_sum, arr)

    # 2) subarray_sum (O(n))
    arr = [random.randint(-5, 5) for _ in range(bench_n)]
    k = random.randint(-5, 5)
    bench("subarray_sum (k)", subarray_sum, arr, k)

    # 3) subarrays_div_by_k
    arr = [random.randint(-6, 6) for _ in range(bench_n)]
    bench("div_by_k (K=5)", subarrays_div_by_k, arr, 5)

    # 4) min_subarray_len (positives)
    arr = [random.randint(1, 10) for _ in range(bench_n)]
    target = random.randint(bench_n//10, bench_n//5)
    bench("min_subarray_len", min_subarray_len, target, arr)

    # 5) range_add
    n = bench_n
    q = bench_n // 4
    updates = []
    for _ in range(q):
        l = random.randint(0, n-1)
        r = random.randint(l, n-1)
        inc = random.randint(-3, 3)
        updates.append((l, r, inc))
    bench("range_add (q ~ n/4)", range_add, n, updates)

    # 6) NumMatrix build + queries
    m = max(1, int((bench_n // 500) ** 0.5))  # keep small-ish
    nn = m
    mat = [[random.randint(-3, 3) for _ in range(nn)] for __ in range(m)]
    t0 = time.perf_counter()
    nm = NumMatrix(mat)
    t1 = time.perf_counter()
    print(f"[BENCH] NumMatrix build {m}x{nn}        time={t1 - t0:.4f}s")
    # queries
    q = min(10000, bench_n // 20)
    t0 = time.perf_counter()
    for _ in range(q):
        r1 = random.randint(0, m-1); r2 = random.randint(r1, m-1)
        c1 = random.randint(0, nn-1); c2 = random.randint(c1, nn-1)
        _ = nm.sumRegion(r1,c1,r2,c2)
    t1 = time.perf_counter()
    print(f"[BENCH] NumMatrix queries x{q:<6d}   time={t1 - t0:.4f}s")

    # 7) num_submatrix_sum_target (hard)
    if bench_hard:
        m = 40; nn = 40
        mat = [[random.randint(-2, 2) for _ in range(nn)] for __ in range(m)]
        t0 = time.perf_counter()
        _ = num_submatrix_sum_target(mat, 0)
        t1 = time.perf_counter()
        print(f"[BENCH] num_submatrix_sum_target {m}x{nn} time={t1 - t0:.4f}s")

# ---------------------- CLI / Main ----------------------

def parse_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--fuzz-cases', type=int, default=50)
    p.add_argument('--no-fuzz', action='store_true')
    p.add_argument('--only', type=str, default='')
    p.add_argument('--bench', action='store_true')
    p.add_argument('--bench-hard', action='store_true')
    p.add_argument('--bench-n', type=int, default=200000)
    p.add_argument('--verbosity', type=int, default=2)
    p.add_argument('--help', action='help')
    return p.parse_known_args()

def build_suite(fuzz_cases=50, no_fuzz=False):
    # Configure fuzz class parameters
    TestFuzz.FUZZ = max(1, fuzz_cases)

    suite = unittest.TestSuite()
    # Unit tests
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(TestRunningSum))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(TestNumArray))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(TestPivotIndex))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(TestSubarraySum))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(TestDivByK))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(TestMinLen))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(TestRangeAdd))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(TestNumMatrix))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(TestNumSubmatrixTarget))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(TestThreeParts))
    # Fuzz tests
    if not no_fuzz:
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(TestFuzz))
    return suite

def main():
    args, remaining = parse_args()

    # Setup global filters
    global ONLY
    ONLY = set(k.strip() for k in args.only.split(',')) - {''}
    unknown = ONLY - ALL_KEYS
    if unknown:
        print(f"Warning: unknown --only keys ignored: {sorted(unknown)}")
        ONLY = ONLY & ALL_KEYS
    if ONLY:
        print(f"Running only: {sorted(ONLY)}")

    random.seed(args.seed)

    # Build and run tests
    suite = build_suite(fuzz_cases=args.fuzz_cases, no_fuzz=args.no_fuzz)
    runner = unittest.TextTestRunner(verbosity=args.verbosity)
    result = runner.run(suite)

    # Optional benchmarks
    if args.bench:
        run_benchmarks(seed=args.seed, bench_n=args.bench_n, bench_hard=args.bench_hard)

    # Exit code = 0 if success
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    raise SystemExit(main())
