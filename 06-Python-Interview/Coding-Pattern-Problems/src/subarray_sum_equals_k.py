"""
Date: 2nd Oct, 2026 

Problem: Subrray sum equals K 
Count number of subarrays whose sum equals k

Approach: 
- Use sliding window technique to find the subarrays whose sum equals k
- Use a hashmap to store the cumulative sum and its frequency

"""

class SubArraySumEqualsK:
    def subarray_sum(self, nums, k):
        count = 0
        cumulative_sum = 0
        sum_frequency = {0: 1}  # Initialize with sum 0 having frequency 1

        for num in nums:
            cumulative_sum += num
            if (cumulative_sum - k) in sum_frequency:
                count += sum_frequency[cumulative_sum - k]
            sum_frequency[cumulative_sum] = sum_frequency.get(cumulative_sum, 0) + 1

        return count

    def subarray_sum_brute_force(self, nums, k):
        count = 0
        n = len(nums)

        for start in range(n):
            current_sum = 0
            for end in range(start, n):
                current_sum += nums[end]
                if current_sum == k:
                    count += 1

        return count

    def subarray_sum_prefix_sum(self, nums, k):
        count = 0
        n = len(nums)
        prefix_sum = [0] * (n + 1)

        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]

        for start in range(n):
            for end in range(start, n):
                if prefix_sum[end + 1] - prefix_sum[start] == k:
                    count += 1

        return count

    def subarray_sum_sliding_window(self, nums, k):
        count = 0
        left = 0
        current_sum = 0

        for right in range(len(nums)):
            current_sum += nums[right]

            while current_sum > k and left <= right:
                current_sum -= nums[left]
                left += 1

            if current_sum == k:
                count += 1

        return count

    def subarray_sum_hashmap(self, nums, k):
            count = 0
            cumulative_sum = 0
            sum_frequency = {0: 1}  # Initialize with sum 0 having frequency 1

            for num in nums:
                cumulative_sum += num
                if (cumulative_sum - k) in sum_frequency:
                    count += sum_frequency[cumulative_sum - k]
                sum_frequency[cumulative_sum] = sum_frequency.get(cumulative_sum, 0) + 1

            return count

    def subarray_sum_naive(self, arr, k) -> int:
        count = 0
        n = len(arr)
        cumulative_sum = 0

        for i in range(n):
            cumulative_sum += arr[i]
            for j in range (i+1, n):
                print (f"i: {i}, j: {j}, arr[i]: {arr[i]}, arr[j]: {arr[j]} ")

        return cumulative_sum
        



if __name__ == "__main__":
    nums = [1, 1, 2, 3, 4, 5]
    k = 5
    solution = SubArraySumEqualsK()
    # print("Count of subarrays with sum equals k (Sliding Window):", solution.subarray_sum(nums, k))
    # print("Count of subarrays with sum equals k (Brute Force):", solution.subarray_sum_brute_force(nums, k))
    # print("Count of subarrays with sum equals k (Prefix Sum):", solution.subarray_sum_prefix_sum(nums, k))  

    cumulative_sum = solution.subarray_sum_naive(nums, k)
    print ("Cumulative sum of the array:", cumulative_sum)
    print ("Total Sum of the array:", sum(nums))
