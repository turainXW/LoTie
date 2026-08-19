from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ListNode:
    val: int = 0
    next: "ListNode | None" = None


class Solution:
    def twoSum(self, nums: list[int], target: int) -> list[int]:
        seen: dict[int, int] = {}
        for index, value in enumerate(nums):
            need = target - value
            if need in seen:
                return [seen[need], index]
            seen[value] = index
        return []

    def addTwoNumbers(self, l1: ListNode | None, l2: ListNode | None) -> ListNode | None:
        dummy = ListNode()
        current = dummy
        carry = 0
        while l1 or l2 or carry:
            total = carry
            if l1:
                total += l1.val
                l1 = l1.next
            if l2:
                total += l2.val
                l2 = l2.next
            carry, digit = divmod(total, 10)
            current.next = ListNode(digit)
            current = current.next
        return dummy.next

    def lengthOfLongestSubstring(self, s: str) -> int:
        left = 0
        best = 0
        last_seen: dict[str, int] = {}
        for right, char in enumerate(s):
            if char in last_seen and last_seen[char] >= left:
                left = last_seen[char] + 1
            last_seen[char] = right
            best = max(best, right - left + 1)
        return best

    def findMedianSortedArrays(self, nums1: list[int], nums2: list[int]) -> float:
        if len(nums1) > len(nums2):
            nums1, nums2 = nums2, nums1
        if not nums1 and not nums2:
            raise ValueError("at least one input array must be non-empty")

        total = len(nums1) + len(nums2)
        half = (total + 1) // 2
        low, high = 0, len(nums1)

        while low <= high:
            partition1 = (low + high) // 2
            partition2 = half - partition1

            left1 = nums1[partition1 - 1] if partition1 else float("-inf")
            right1 = nums1[partition1] if partition1 < len(nums1) else float("inf")
            left2 = nums2[partition2 - 1] if partition2 else float("-inf")
            right2 = nums2[partition2] if partition2 < len(nums2) else float("inf")

            if left1 <= right2 and left2 <= right1:
                if total % 2:
                    return float(max(left1, left2))
                return (max(left1, left2) + min(right1, right2)) / 2.0
            if left1 > right2:
                high = partition1 - 1
            else:
                low = partition1 + 1

        raise ValueError("input arrays must be sorted")

    def longestPalindrome(self, s: str) -> str:
        if len(s) < 2:
            return s

        def expand(left: int, right: int) -> tuple[int, int]:
            while left >= 0 and right < len(s) and s[left] == s[right]:
                left -= 1
                right += 1
            return left + 1, right - 1

        best_left = 0
        best_right = 0
        for center in range(len(s)):
            odd_left, odd_right = expand(center, center)
            even_left, even_right = expand(center, center + 1)
            if odd_right - odd_left > best_right - best_left:
                best_left, best_right = odd_left, odd_right
            if even_right - even_left > best_right - best_left:
                best_left, best_right = even_left, even_right
        return s[best_left : best_right + 1]

    def convert(self, s: str, numRows: int) -> str:
        if numRows == 1 or numRows >= len(s):
            return s
        rows = [""] * numRows
        row = 0
        direction = 1
        for char in s:
            rows[row] += char
            if row == 0:
                direction = 1
            elif row == numRows - 1:
                direction = -1
            row += direction
        return "".join(rows)

    def reverse(self, x: int) -> int:
        sign = -1 if x < 0 else 1
        value = int(str(abs(x))[::-1]) * sign
        if value < -(2**31) or value > 2**31 - 1:
            return 0
        return value

    def myAtoi(self, s: str) -> int:
        index = 0
        n = len(s)
        while index < n and s[index] == " ":
            index += 1

        sign = 1
        if index < n and s[index] in "+-":
            sign = -1 if s[index] == "-" else 1
            index += 1

        value = 0
        while index < n and s[index].isdigit():
            value = value * 10 + int(s[index])
            index += 1
            if sign * value < -(2**31):
                return -(2**31)
            if sign * value > 2**31 - 1:
                return 2**31 - 1
        return sign * value

    def isPalindrome(self, x: int) -> bool:
        if x < 0 or (x % 10 == 0 and x != 0):
            return False
        reversed_half = 0
        while x > reversed_half:
            reversed_half = reversed_half * 10 + x % 10
            x //= 10
        return x == reversed_half or x == reversed_half // 10

    def isMatch(self, s: str, p: str) -> bool:
        rows = len(s) + 1
        cols = len(p) + 1
        dp = [[False] * cols for _ in range(rows)]
        dp[0][0] = True

        for col in range(2, cols):
            if p[col - 1] == "*":
                dp[0][col] = dp[0][col - 2]

        for row in range(1, rows):
            for col in range(1, cols):
                pattern = p[col - 1]
                if pattern == "." or pattern == s[row - 1]:
                    dp[row][col] = dp[row - 1][col - 1]
                elif pattern == "*":
                    dp[row][col] = dp[row][col - 2]
                    prev_pattern = p[col - 2]
                    if prev_pattern == "." or prev_pattern == s[row - 1]:
                        dp[row][col] = dp[row][col] or dp[row - 1][col]
        return dp[-1][-1]


def linked_list_from_values(values: list[int]) -> ListNode | None:
    dummy = ListNode()
    current = dummy
    for value in values:
        current.next = ListNode(value)
        current = current.next
    return dummy.next


def linked_list_to_values(node: ListNode | None) -> list[int]:
    values = []
    while node:
        values.append(node.val)
        node = node.next
    return values
