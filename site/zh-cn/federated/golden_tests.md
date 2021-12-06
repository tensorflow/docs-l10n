# 黄金测试

TFF 包含一个名为 `golden` 的小型库，有助于轻松编写和维护黄金测试。

## 什么是黄金测试？我应当何时使用它们？

当您希望开发者知道他们的代码更改了函数的输出时，可以使用黄金测试。黄金测试违反了良好单元测试的许多特征，因为它们对函数的确切输出做出了承诺，而不是测试一组特定的清晰、记录完善的属性。有时候不清楚对黄金输出的更改何时是“预期的”，或者更改是否违反了黄金测试试图强制执行的某些属性。因此，构造良好的单元测试通常比黄金测试更可取。

但是，在验证错误消息、诊断或已生成代码的确切内容时，黄金测试极为有用。在这些情况下，黄金测试可以是一种有帮助的信心检查，目的是确认对所生成输出的任何更改“看起来正确”。

## 我应当如何使用 `golden` 编写测试？

`golden.check_string(filename, value)` 是 `golden` 库的主要入口点。它将根据最后一个路径元素为 `filename` 的文件的内容检查 `value` 字符串。必须通过命令行 `--golden <path_to_file>` 参数提供 `filename` 的完整路径。同样，必须使用 `py_test` 构建规则的 `data` 参数使这些文件可用于测试。使用 `location` 函数生成正确的相对路径：

```
py_string_test(
  ...
  args = [
    "--golden",
    "$(location path/to/first_test_output.expected)",
    ...
    "--golden",
    "$(location path/to/last_test_output.expected)",
  ],
  data = [
    "path/to/first_test_output.expected",
    ...
    "path/to/last_test_output.expected",
  ],
  ...
)
```

按照惯例，黄金文件应当置于与其测试目标同名的同级目录中，并以 `_goldens` 为后缀：

```
path/
  to/
    some_test.py
    some_test_goldens/
      test_case_one.expected
      ...
      test_case_last.expected
```

## 如何更新  `.expected` 文件？

可以通过使用参数 `--test_arg=--update_goldens --test_strategy=local` 运行受影响的测试目标来更新 `.expected` 文件。应检查由此产生的差异是否存在意外的变化。
