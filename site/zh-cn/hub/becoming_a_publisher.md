<!--* freshness: { owner: 'maringeo' } *-->

# 成为发布者

## 服务条款

提交模型进行发布，即表示您同意 [https://tfhub.dev/terms](https://tfhub.dev/terms) 所列的 TensorFlow Hub 服务条款。

## 发布流程概述

完整的发布流程包括：

1. 创建模型（请参阅如何[导出模型](exporting_tf2_saved_model.md)）
2. 编写文档（请参阅如何[编写模型文档](writing_model_documentation.md)）
3. 创建发布请求（请参阅如何[贡献](contribute_a_model.md)）

## 发布者页面特定的 Markdown 格式

发布者文档在 Markdown 文件中声明，该文件与[编写模型文档](writing_model_documentation)指南中所述类型相同 ，但语法上略有不同。

TensorFlow Hub 仓库中存储发布者文件的正确位置为：[hub/tfhub_dev/assets/](https://github.com/tensorflow/hub/tree/master/tfhub_dev/assets)/<publisher_name>/<publisher_name.md>

请参见以下精简的发布者文档示例：

```markdown
# Publisher vtab
Visual Task Adaptation Benchmark

[![Icon URL]](https://storage.googleapis.com/vtab/vtab_logo_120.png)

## VTAB
The Visual Task Adaptation Benchmark (VTAB) is a diverse, realistic and
challenging benchmark to evaluate image representations.
```

以上示例指定了发布者名称、简要描述、要使用的图标的路径以及更长的形式不限的 Markdown 文档。

### 发布者名称准则

您的发布者名称既可以是您的 GitHub 用户名，也可以是您所管理的 GitHub 组织的名称。
