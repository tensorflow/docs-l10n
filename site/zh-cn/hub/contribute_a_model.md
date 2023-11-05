# 提交拉取请求

本页面介绍如何向 [tensorflow/tfhub.dev](https://github.com/tensorflow/tfhub.dev) GitHub 仓库提交包含 Markdown 文档文件提交拉取请求。有关如何首先编写 Markdown 文件的更多信息，请参阅[编写文档](writing_documentation.md)指南。

**注**：如果您希望将您的模型镜像到其他模型中心，请使用 MIT、CC0 或 Apache 许可。如果您不希望您的模型被镜像到其他模型中心，请使用其他适当的许可证。

## GitHub 操作检查

[tensorflow/tfhub.dev](https://github.com/tensorflow/tfhub.dev) 仓库使用 GitHub 操作来验证拉取请求中的文件格式。工作流用于验证 [.github/workflows/contributions-validator.yml](https://github.com/tensorflow/tfhub.dev/blob/master/.github/workflows/contributions-validator.yml) 中定义的拉取请求。您可以在工作流之外您自己的分支上运行验证器脚本，但需要确保已安装了所有正确的 PIP 软件包依赖项。

根据 [GitHub 策略](https://github.blog/changelog/2021-04-22-github-actions-maintainers-must-approve-first-time-contributor-workflow-runs/)，首次贡献者只能在仓库维护人员批准的情况下运行自动检查。我们鼓励发布者提交一个用来修复拼写错误的小拉取请求，或改进模型文档，或提交一份只包含其发布者页面的拉取请求作为第一拉取请求，以便能够对后续拉取请求进行自动检查。

重要提示：您的拉取请求必须通过自动检查，然后才能进行审核！

## 提交拉取请求

可以通过以下方式之一将完整的 Markdown 文件拉取到 [tensorflow/tfhub.dev](https://github.com/tensorflow/tfhub.dev/tree/master) 的 master 分支。

### Git CLI 提交

假设确定的 Markdown 文件路径为 `assets/docs/publisher/model/1.md`，您可以按照标准 Git[Hub] 步骤对新增文件创建新的拉取请求。

首先需要复刻 TensorFlow Hub GitHub 仓库，然后[通过此复刻分支创建拉取请求](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request-from-a-fork)将文件拉入 TensorFlow Hub master 分支。

以下是将新文件添加到复刻仓库的 master 分支所需的典型 CLI git 命令。

```bash
git clone https://github.com/[github_username]/tfhub.dev.git
cd tfhub.dev
mkdir -p assets/docs/publisher/model
cp my_markdown_file.md ./assets/docs/publisher/model/1.md
git add *
git commit -m "Added model file."
git push origin master
```

### GitHub GUI 提交

通过 GitHub 图形界面提交是一种更直观的方式。GitHub 支持直接通过 GUI 为[新建文件](https://help.github.com/en/github/managing-files-in-a-repository/creating-new-files)或[文件编辑](https://help.github.com/en/github/managing-files-in-a-repository/editing-files-in-your-repository)创建拉取请求。

1. 在 [TensorFlow Hub GitHub 页面](https://github.com/tensorflow/hub)中，按 `Create new file` 按钮。
2. 设置正确的文件路径：`assets/docs/publisher/model/1.md`
3. 复制并粘贴现有的 Markdown 文件。
4. 在底部，选择“Create a new branch for this commit and start a pull request”。
