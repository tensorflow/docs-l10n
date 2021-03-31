<!--* freshness: { owner: 'maringeo' reviewed: '2021-02-25' review_interval: '3 months' } *-->

# 为模型做贡献

本页面介绍如何将 Markdown 文档文件添加到 GitHub。有关如何编写 Markdown 文件的更多信息，请参阅[编写模型文档](writing_model_documentation.md)指南。

## 提交模型

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
