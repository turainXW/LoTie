# LeetCode 前 10 题 Python 测试集

这个目录用于快速测试 Python coding 能力，也可以后续拆成 agent benchmark。

覆盖题目：

1. Two Sum
2. Add Two Numbers
3. Longest Substring Without Repeating Characters
4. Median of Two Sorted Arrays
5. Longest Palindromic Substring
6. Zigzag Conversion
7. Reverse Integer
8. String to Integer (atoi)
9. Palindrome Number
10. Regular Expression Matching

运行：

```bash
python3 -m unittest discover -s code_agent_quickstart/benchmarks/leetcode_top10 -v
```

文件：

- `solutions.py`：LeetCode 风格方法实现。
- `test_solutions.py`：标准库 `unittest` 回归测试。
