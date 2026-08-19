import unittest

from solutions import Solution, linked_list_from_values, linked_list_to_values


class AgentGeneratedLeetCodeTop10EdgeCases(unittest.TestCase):
    def setUp(self) -> None:
        self.solution = Solution()

    def test_001_two_sum_ten_edge_cases(self) -> None:
        cases = [
            ([0, 4, 3, 0], 0, [0, 3]),
            ([-3, 4, 3, 90], 0, [0, 2]),
            ([1, 5, 1, 5], 10, [1, 3]),
            ([2, 5, 5, 11], 10, [1, 2]),
            ([1, 2, 3], 7, []),
            ([10**9, -(10**9), 3], 0, [0, 1]),
            ([-1, -2, -3, -4, -5], -8, [2, 4]),
            ([3, 2, 95, 4, -3], 92, [2, 4]),
            ([5, 75, 25], 100, [1, 2]),
            ([1, 3, 4, 2], 6, [2, 3]),
        ]
        for nums, target, expected in cases:
            with self.subTest(nums=nums, target=target):
                self.assertEqual(self.solution.twoSum(nums, target), expected)

    def test_002_add_two_numbers_ten_edge_cases(self) -> None:
        cases = [
            ([0], [0], [0]),
            ([9], [1], [0, 1]),
            ([9, 9], [1], [0, 0, 1]),
            ([1, 8], [0], [1, 8]),
            ([0], [7, 3], [7, 3]),
            ([5], [5], [0, 1]),
            ([2, 4, 3], [5, 6, 4], [7, 0, 8]),
            ([9, 9, 9, 9], [9, 9], [8, 9, 0, 0, 1]),
            ([], [1, 2, 3], [1, 2, 3]),
            ([], [], []),
        ]
        for left, right, expected in cases:
            with self.subTest(left=left, right=right):
                result = self.solution.addTwoNumbers(linked_list_from_values(left), linked_list_from_values(right))
                self.assertEqual(linked_list_to_values(result), expected)

    def test_003_longest_substring_ten_edge_cases(self) -> None:
        cases = [
            ("", 0),
            (" ", 1),
            ("au", 2),
            ("dvdf", 3),
            ("abba", 2),
            ("tmmzuxt", 5),
            ("abcabcbb", 3),
            ("bbbbb", 1),
            ("pwwkew", 3),
            ("abcdefga", 7),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(self.solution.lengthOfLongestSubstring(text), expected)

    def test_004_median_ten_edge_cases(self) -> None:
        cases = [
            ([], [1], 1.0),
            ([1], [], 1.0),
            ([1], [2], 1.5),
            ([1, 2], [3, 4], 2.5),
            ([0, 0], [0, 0], 0.0),
            ([-5, -3], [-4, -1], -3.5),
            ([1, 2, 3, 4, 5], [100], 3.5),
            ([1, 3, 8], [7, 9, 10, 11], 8.0),
            ([2], [1, 3, 4, 5, 6], 3.5),
            ([], [], ValueError),
        ]
        for nums1, nums2, expected in cases:
            with self.subTest(nums1=nums1, nums2=nums2):
                if expected is ValueError:
                    with self.assertRaises(ValueError):
                        self.solution.findMedianSortedArrays(nums1, nums2)
                else:
                    self.assertEqual(self.solution.findMedianSortedArrays(nums1, nums2), expected)

    def test_005_longest_palindrome_ten_edge_cases(self) -> None:
        cases = [
            ("", {""}),
            ("a", {"a"}),
            ("aa", {"aa"}),
            ("ab", {"a"}),
            ("babad", {"bab", "aba"}),
            ("cbbd", {"bb"}),
            ("racecar", {"racecar"}),
            ("forgeeksskeegfor", {"geeksskeeg"}),
            ("aacabdkacaa", {"aca"}),
            ("aaaa", {"aaaa"}),
        ]
        for text, expected_values in cases:
            with self.subTest(text=text):
                result = self.solution.longestPalindrome(text)
                self.assertIn(result, expected_values)
                self.assertEqual(result, result[::-1])

    def test_006_zigzag_ten_edge_cases(self) -> None:
        cases = [
            ("", 1, ""),
            ("A", 1, "A"),
            ("AB", 3, "AB"),
            ("ABC", 2, "ACB"),
            ("ABCD", 2, "ACBD"),
            ("PAYPALISHIRING", 3, "PAHNAPLSIIGYIR"),
            ("PAYPALISHIRING", 4, "PINALSIGYAHRPI"),
            ("HELLOWORLD", 1, "HELLOWORLD"),
            ("HELLOWORLD", 10, "HELLOWORLD"),
            ("ABCDE", 4, "ABCED"),
        ]
        for text, rows, expected in cases:
            with self.subTest(text=text, rows=rows):
                self.assertEqual(self.solution.convert(text, rows), expected)

    def test_007_reverse_integer_ten_edge_cases(self) -> None:
        cases = [
            (0, 0),
            (5, 5),
            (-5, -5),
            (120, 21),
            (-120, -21),
            (1534236469, 0),
            (1463847412, 2147483641),
            (-1463847412, -2147483641),
            (1563847412, 0),
            (-2147483648, 0),
        ]
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(self.solution.reverse(value), expected)

    def test_008_atoi_ten_edge_cases(self) -> None:
        cases = [
            ("", 0),
            ("42", 42),
            ("   -42", -42),
            ("+1", 1),
            ("+-12", 0),
            ("00000-42a1234", 0),
            ("4193 with words", 4193),
            ("words and 987", 0),
            ("91283472332", 2147483647),
            ("-91283472332", -2147483648),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(self.solution.myAtoi(text), expected)

    def test_009_palindrome_number_ten_edge_cases(self) -> None:
        cases = [
            (0, True),
            (1, True),
            (11, True),
            (121, True),
            (1221, True),
            (12321, True),
            (-121, False),
            (10, False),
            (123, False),
            (1000021, False),
        ]
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(self.solution.isPalindrome(value), expected)

    def test_010_regex_matching_ten_edge_cases(self) -> None:
        cases = [
            ("", "", True),
            ("", "a*", True),
            ("aa", "a", False),
            ("aa", "a*", True),
            ("ab", ".*", True),
            ("aab", "c*a*b", True),
            ("mississippi", "mis*is*p*.", False),
            ("ab", ".*c", False),
            ("aaa", "a*a", True),
            ("abcd", "d*", False),
        ]
        for text, pattern, expected in cases:
            with self.subTest(text=text, pattern=pattern):
                self.assertEqual(self.solution.isMatch(text, pattern), expected)


if __name__ == "__main__":
    unittest.main()
