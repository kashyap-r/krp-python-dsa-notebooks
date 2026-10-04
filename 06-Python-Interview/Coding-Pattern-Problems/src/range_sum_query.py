"""
Date: 2nd Oct, 2026

Range Sum Query (Easy) 

Problem Statement: 

Given an integer array nums, handle multiple queries of the following type:
Calculate the sum of the elements of nums between indices left and right inclusive 
where left <= right. 

Pattern Context:
This problem is a classic example of the prefix sum technique. 
The idea is to preprocess the input array to create a prefix sum array, 
which allows for efficient range sum queries.    

In real-world scenarios, this technique is often used in applications where 
multiple range sum queries need to be answered quickly, such as in financial
data analysis, image processing, and more. 

Another application of this technique is in competitive programming, where 
it is common to encounter problems that require efficient range sum calculations.
For ex. to compute log volume per minute or query total logs between two timestamps.

Constraints (to confirm the problem is well-defined):
1. The input array nums can contain both positive and negative integers.
2. The indices left and right are valid (0 <= left <= right < len(nums)).
3. The number of queries can be large, so an efficient solution is required.

Edge-Cases:
1. If the input array is empty, return 0 for any query.
2. If left and right are the same, return the value at that index.
3. If the input array contains only one element, return that element for any query.
4. If the input array contains all negative numbers, the sum can be negative.
5. If the input array contains all positive numbers, the sum will always be non-negative.
6. If the input array contains a mix of positive and negative numbers, the sum can be positive, negative, or zero depending on the range queried.
7. If the input array contains zeros, the sum can be zero for certain ranges.
8. If the input array contains large numbers, ensure that the sum does not overflow (in Python, integers can grow arbitrarily large, but in other languages, this could be a concern).
9. If the input array contains duplicate numbers, the sum will include all occurrences of those numbers in the specified range.
10. If the input array is sorted, the sum can be calculated more efficiently using binary search to find the range, but this is not necessary for the prefix sum technique.
11. If the left and right indices are at the boundaries of the array (0 and len(nums)-1), the sum will include all elements in the array.
"""
class RangeSumQuery:
    def __init__(self, nums):
        self.prefix_sum = [0] * (len(nums) + 1)
        self.prefix_sum[0] = nums[0] if nums else 0

        for i in range(len(nums)):
            self.prefix_sum[i] = self.prefix_sum[i-1] + nums[i]

    def sumRange(self, left, right):
        if left == 0:
            return self.prefix_sum[right]
        return self.prefix_sum[right] - self.prefix_sum[left-1]
