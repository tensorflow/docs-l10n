# 为 TensorFlow 代码做贡献

无论您是添加损失函数、提高测试覆盖率还是为重大设计变更编写 RFC，本部分贡献者指南将帮助您快速入门。感谢您为改进 TensorFlow 所做的工作和给予的关注。

## 准备工作

在为 TensorFlow 项目贡献源代码之前，请查看该项目的 GitHub 仓库中的 `CONTRIBUTING.md` 文件。（例如，请参阅[核心 TensorFlow 仓库的 CONTRIBUTING.md 文件](https://github.com/tensorflow/tensorflow/blob/master/CONTRIBUTING.md)。）所有代码贡献者都需要签署[贡献者许可协议](https://cla.developers.google.com/clas) (CLA)。

为避免重复工作，在开始处理非普通功能之前，请查看[当前](https://github.com/tensorflow/community/tree/master/rfcs)或[提议的](https://github.com/tensorflow/community/labels/RFC%3A%20Proposed) RFC 并与 TensorFlow 论坛上的开发者 ([developers@tensorflow.org](https://groups.google.com/u/1/a/tensorflow.org/g/developers)) 联系。在决定添加新功能时，我们是有选择性的，为项目做贡献和提供帮助的最佳方法是处理已知问题。

## 新贡献者的问题

新贡献者在搜索对 TensorFlow 代码库的首次贡献时应查找以下标签。我们强烈建议新贡献者先处理具有“good first issue”和“contributions welcome”标签的项目；这有助于贡献者熟悉贡献工作流程，并让核心开发者认识贡献者。

- [good first issue](https://github.com/tensorflow/tensorflow/labels/good%20first%20issue)
- [contributions welcome](https://github.com/tensorflow/tensorflow/labels/stat%3Acontributions%20welcome)

如果您有兴趣招募一个团队来帮助解决大规模问题或处理新功能，请给 [developers@ 小组](https://groups.google.com/a/tensorflow.org/forum/#!forum/developers)发送电子邮件，并查看我们当前的 RFC 列表。

## 代码审查

新功能、错误修复以及对代码库的任何其他更改都必须经过代码审查。

以拉取请求的形式审查贡献到项目的代码是 TensorFlow 开发的关键组成部分。我们鼓励任何人审查其他开发者提交的代码，尤其是在您可能会使用该功能的情况下。

在代码审查过程中，请牢记以下问题：

- *我们在 TensorFlow 中需要这个功能吗？*它有可能被使用吗？作为 TensorFlow 用户，您是否喜欢此更改并打算使用它？此更改在 TensorFlow 的范围内吗？维护新功能的成本是否物有所值？

- *代码与 TensorFlow API 是否一致？*公共函数、类和参数的命名是否合理，设计是否直观？

- *是否包括文档？*所有公共函数、类、参数、返回类型和存储的特性是否都已根据 TensorFlow 约定命名并明确记录？TensorFlow 文档中是否尽可能地描述了新功能并通过示例加以说明？文档是否正确呈现？

- *代码是人类可读的吗？*冗余度低吗？是否应改进变量名称以使其清晰或一致？是否应添加注释？是否应移除任何无用或多余的注释？

- *代码是否高效？*能否轻松重写此代码以更高效地运行？

- 代码是否*向后兼容*先前版本的 TensorFlow？

- 新代码是否会添加与其他库的*新依赖关系*？

## 测试并提高测试覆盖率

高质量的单元测试是 TensorFlow 开发流程的基石。为此，我们使用 Docker 镜像。测试函数经过适当的命名，负责检查算法的有效性以及代码的不同选项。

所有新功能和错误修复都*必须*包括足够的测试覆盖率。我们也欢迎提供新测试用例或改进现有测试的贡献。如果您发现我们现有的测试不完整（即使当前尚未导致错误），请提交问题，并在可能的情况下提交拉取请求。

有关每个 TensorFlow 项目中测试过程的特定详细信息，请参阅 GitHub 上项目仓库中的 `README.md` 和 `CONTRIBUTING.md` 文件。

*充分测试*中需要特别关注的问题：

- 是否*对每个公共函数和类都进行了*测试？
- 是否测试了*一组合理的参数*、它们的值、值类型和组合？
- 测试是否验证了*代码正确无误*，以及它*所执行的操作是否符合文档中所述的*代码意图？
- 如果更改是错误修复，是否包括*非回归测试* ？
- 测试是否*通过持续集成*构建？
- 测试是否*覆盖代码的每一行*？如果未覆盖，例外是否合理且明确？

如果发现任何问题，请考虑帮助贡献者了解这些问题并加以解决。

## 改善错误消息或日志

我们欢迎改善错误消息和日志记录的贡献。

## 贡献工作流

代码贡献（错误修复、新开发、测试改进）全部遵循以 GitHub 为中心的工作流。要参与 TensorFlow 开发，请建立一个 GitHub 帐号。然后：

1. 将您计划处理的仓库分叉。转到项目仓库页面，然后使用 *Fork* 按钮。这将在您的用户名下创建一个仓库副本。（有关如何将仓库分叉的更多详细信息，请参阅[此指南](https://help.github.com/articles/fork-a-repo/)。）

2. 将仓库克隆到本地系统。

    `$ git clone git@github.com:your-user-name/project-name.git`

3. 创建一个新分支来保存您的工作。

    `$ git checkout -b new-branch-name`

4. 处理新代码。编写并运行测试。

5. 提交您的更改。

    `$ git add -A`

    `$ git commit -m "commit message here"`

6. 将您的更改推送到 GitHub 仓库。

    `$ git push origin branch-name`

7. 打开一个*拉取请求* (PR)。转到 GitHub 上的原始项目仓库。随后将显示一条有关您最近推送的分支的消息，询问您是否要打开拉取请求。按照提示进行操作，*在仓库之间进行比较*，然后提交 PR。这将向提交者发送一封电子邮件。您可能需要考虑将电子邮件发送到邮寄名单来提高关注度。（有关更多详细信息，请参阅[关于 PR 的 GitHub 指南](https://help.github.com/articles/creating-a-pull-request-from-a-fork)。）

8. 维护者和其他贡献者将*审查您的 PR*。请参与对话，并尝试*进行任何要求的更改*。一旦 PR 获得批准，便会合并代码。

*在处理下一个贡献之前*，请确保您的本地仓库是最新的。

1. 设置远程的上游仓库。（每个项目只需执行一次，不必每次都执行。）

    `$ git remote add upstream git@github.com:tensorflow/project-repo-name`

2. 切换到本地 master 分支。

    `$ git checkout master`

3. 从上游拉取更改。

    `$ git pull upstream master`

4. 将更改推送到您的 GitHub 帐号。（可选，但这是一个好的做法。）

    `$ git push origin master`

5. 如果您要开始新工作，请创建一个新分支。

    `$ git checkout -b branch-name`

其他 `git` 和 GitHub 资源：

- [Git 文档](https://git-scm.com/documentation)
- [Git 开发工作流](https://docs.scipy.org/doc/numpy/dev/gitwash/development_workflow.html)
- [解决合并冲突](https://help.github.com/articles/resolving-a-merge-conflict-using-the-command-line/)。

## 贡献者核对清单

- 阅读[贡献准则](https://github.com/tensorflow/tensorflow/blob/master/CONTRIBUTING.md)。
- 阅读《行为准则》。
- 确保您已签署《贡献者许可协议》(CLA)。
- 检查您的更改与准则是否一致。
- 检查您的更改与 TensorFlow 编码风格是否一致。
- [运行单元测试](https://github.com/tensorflow/tensorflow/blob/master/CONTRIBUTING.md#running-unit-tests)。
