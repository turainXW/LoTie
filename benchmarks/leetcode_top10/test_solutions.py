import unittest

from solutions import Solution, linked_list_from_values, linked_list_to_values


class LeetCodeTop10Test(unittest.TestCase):
    def setUp(self) -> None:
        self.solution = Solution()

    def test_001_two_sum(self) -> None:
        self.assertEqual(self.solution.twoSum([2, 7, 11, 15], 9), [0, 1])
        self.assertEqual(self.solution.twoSum([3, 2, 4], 6), [1, 2])
        self.assertEqual(self.solution.twoSum([3, 3], 6), [0, 1])

    def test_002_add_two_numbers(self) -> None:
        self.assertEqual(
            linked_list_to_values(
                self.solution.addTwoNumbers(
                    linked_list_from_values([2, 4, 3]),
                    linked_list_from_values([5, 6, 4]),
                )
            ),
            [7, 0, 8],
        )
        self.assertEqual(
            linked_list_to_values(
                self.solution.addTwoNumbers(
                    linked_list_from_values([0]),
                    linked_list_from_values([0]),
                )
            ),
            [0],
        )
        self.assertEqual(
            linked_list_to_values(
                self.solution.addTwoNumbers(
                    linked_list_from_values([9, 9, 9, 9, 9, 9, 9]),
                    linked_list_from_values([9, 9, 9, 9]),
                )
            ),
            [8, 9, 9, 9, 0, 0, 0, 1],
        )

    def test_003_longest_substring_without_repeating_characters(self) -> None:
        self.assertEqual(self.solution.lengthOfLongestSubstring("abcabcbb"), 3)
        self.assertEqual(self.solution.lengthOfLongestSubstring("bbbbb"), 1)
        self.assertEqual(self.solution.lengthOfLongestSubstring("pwwkew"), 3)
        self.assertEqual(self.solution.lengthOfLongestSubstring(""), 0)
        self.assertEqual(self.solution.lengthOfLongestSubstring("abba"), 2)

    def test_004_median_of_two_sorted_arrays(self) -> None:
        self.assertEqual(self.solution.findMedianSortedArrays([1, 3], [2]), 2.0)
        self.assertEqual(self.solution.findMedianSortedArrays([1, 2], [3, 4]), 2.5)
        self.assertEqual(self.solution.findMedianSortedArrays([], [1]), 1.0)
        self.assertEqual(self.solution.findMedianSortedArrays([0, 0], [0, 0]), 0.0)
        with self.assertRaises(ValueError):
            self.solution.findMedianSortedArrays([], [])

    def test_005_longest_palindromic_substring(self) -> None:
        self.assertIn(self.solution.longestPalindrome("babad"), {"bab", "aba"})
        self.assertEqual(self.solution.longestPalindrome("cbbd"), "bb")
        self.assertEqual(self.solution.longestPalindrome("a"), "a")
        self.assertEqual(self.solution.longestPalindrome("aacabdkacaa"), "aca")

    def test_006_zigzag_conversion(self) -> None:
        self.assertEqual(self.solution.convert("PAYPALISHIRING", 3), "PAHNAPLSIIGYIR")
        self.assertEqual(self.solution.convert("PAYPALISHIRING", 4), "PINALSIGYAHRPI")
        self.assertEqual(self.solution.convert("A", 1), "A")
        self.assertEqual(self.solution.convert("AB", 3), "AB")

    def test_007_reverse_integer(self) -> None:
        self.assertEqual(self.solution.reverse(123), 321)
        self.assertEqual(self.solution.reverse(-123), -321)
        self.assertEqual(self.solution.reverse(120), 21)
        self.assertEqual(self.solution.reverse(1534236469), 0)
        self.assertEqual(self.solution.reverse(-2147483648), 0)

    def test_008_string_to_integer_atoi(self) -> None:
        self.assertEqual(self.solution.myAtoi("42"), 42)
        self.assertEqual(self.solution.myAtoi("   -42"), -42)
        self.assertEqual(self.solution.myAtoi("4193 with words"), 4193)
        self.assertEqual(self.solution.myAtoi("words and 987"), 0)
        self.assertEqual(self.solution.myAtoi("-91283472332"), -2147483648)
        self.assertEqual(self.solution.myAtoi("91283472332"), 2147483647)
        self.assertEqual(self.solution.myAtoi("+-12"), 0)

    def test_009_palindrome_number(self) -> None:
        self.assertTrue(self.solution.isPalindrome(121))
        self.assertFalse(self.solution.isPalindrome(-121))
        self.assertFalse(self.solution.isPalindrome(10))
        self.assertTrue(self.solution.isPalindrome(0))
        self.assertTrue(self.solution.isPalindrome(1221))

    def test_010_regular_expression_matching(self) -> None:
        self.assertFalse(self.solution.isMatch("aa", "a"))
        self.assertTrue(self.solution.isMatch("aa", "a*"))
        self.assertTrue(self.solution.isMatch("ab", ".*"))
        self.assertTrue(self.solution.isMatch("aab", "c*a*b"))
        self.assertFalse(self.solution.isMatch("mississippi", "mis*is*p*."))
        self.assertFalse(self.solution.isMatch("ab", ".*c"))
        self.assertTrue(self.solution.isMatch("", ""))
        self.assertTrue(self.solution.isMatch("", "a*"))


if __name__ == "__main__":
    unittest.main()
