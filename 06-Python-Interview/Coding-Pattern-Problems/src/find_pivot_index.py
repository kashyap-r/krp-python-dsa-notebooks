"""
Find Pivot Index

Find the index  where left sum = right sum 

Approach: 
1. Compute total sum of the array.
2. Maintain running left sum while iterating through the array.
3. For each index, check if left sum equals total sum minus left sum minus current element
    left_sum = total_sum - left_sum - nums[i]
4. If found, return the index; otherwise, return -1 if no pivot index exists    

Edge Cases:
1. If the input array is empty, return -1.
2. If the input array contains only one element, return 0 as the pivot index.
3. If the input array contains all negative numbers, the pivot index can still exist if the left and right sums are equal.
4. If the input array contains all positive numbers, the pivot index can still exist if the left and right sums are equal.
5. If the input array contains a mix of positive and negative numbers, the pivot index can still exist if the left and right sums are equal.
6. If the input array contains zeros, the pivot index can still exist if the left and right sums are equal
7. If the input array contains duplicate numbers, the pivot index can still exist if the left and right sums are equal.
8. If the input array is sorted, the pivot index can still exist if the left and right sums are equal, but sorting is not necessary for this problem.

Boundary Conditions:
1. If the pivot index is at the start of the array (index 0), the left sum is considered to be 0, and the right sum is the sum of the rest of the elements.
2. If the pivot index is at the end of the array (last index), the right sum is considered to be 0, and the left sum is the sum of all the previous elements.
3. If the pivot index is in the middle of the array, both left and right sums are calculated based on the elements to the left and right of the pivot index, respectively.  
4. If there are multiple pivot indices, return the leftmost one.
"""


class FindPivotIndex:
    def pivotIndex(self, nums: list[int]) -> int:
        total_sum = sum(nums)
        left_sum = 0

        for i, num in enumerate(nums):
            if left_sum == (total_sum - left_sum - num):
                return i
            left_sum += num

        return -1

    def pivotIndex2(self, nums: list[int]) -> int:
        total_sum = sum(nums)
        left_sum = 0

        for i in range(len(nums)):
            if left_sum == (total_sum - left_sum - nums[i]):
                return i
            left_sum += nums[i]

        return -1

if __name__ == "__main__":
    nums = [1, 7, 3, 6, 5, 6]
    pivot_finder = FindPivotIndex()
    result = pivot_finder.pivotIndex(nums)
    print(f"Pivot index: {result}")  # Output: Pivot index: 3

    result2 = pivot_finder.pivotIndex2(nums)
    print(f"Pivot index (method 2): {result2}")  # Output: Pivot index (method 2): 3    
    