# TensorFlow RFC 流程

TensorFlow 的每一项新功能都从征求意见 (RFC) 开始。

RFC 是描述需求与解决需求的建议更改的文档。具体来说，RFC 将：

- 根据 [RFC 模板](https://github.com/tensorflow/community/blob/master/rfcs/yyyymmdd-rfc-template.md)进行格式化。
- 作为拉取请求提交到 [community/rfcs](https://github.com/tensorflow/community/tree/master/rfcs) 目录。
- 在接受之前要经过讨论和审查会议。

TensorFlow 征求意见 (RFC) 的目的是通过从利益相关者和专家那里获得反馈，并广泛地交流设计变更，从而促进 TensorFlow 社区成员积极地参与开发工作。

## 如何提交 RFC

1. 在提交 RFC 之前，请与项目贡献者和维护者讨论您的目标，并尽早获得反馈。请使用有关项目的开发者邮寄名单（developers@tensorflow.org 或相关 SIG 的名单）。

2. 起草您的 RFC。

    - 阅读[设计审核标准](https://github.com/tensorflow/community/blob/master/governance/design-reviews.md)
    - 遵循 [RFC 模板](https://github.com/tensorflow/community/blob/master/rfcs/yyyymmdd-rfc-template.md)。
    - 将您的 RFC 文件命名为 `YYYYMMDD-descriptive-name.md`，其中 `YYYYMMDD` 是提交日期，而 `descriptive-name` 与您的 RFC 标题相关。（例如，如果您的 RFC 标题为 *Parallel Widgets API*，则可以使用文件名 `20180531-parallel-widgets.md`）。
    - 如果要提交图像或其他辅助文件，请创建格式为 `YYYYMMDD-descriptive-name` 的目录来存储这些文件。

    编写 RFC 草稿后，请先征求维护者和贡献者的反馈，然后再提交。

    不需要编写实现代码，但它有助于设计讨论。

3. 招募发起人。

    - 发起人必须是项目的维护者。
    - 请先在 RFC 中注明发起人，然后再发布 PR。

    您*可以*在没有发起人的情况下发布 RFC，但是如果在发布 PR 的一个月内仍然没有发起人，则该 PR 将被关闭。

4. 将您的 RFC 作为拉取请求提交到 [tensorflow/community/rfcs](https://github.com/tensorflow/community/tree/master/rfcs)。

    使用 Markdown，在拉取请求的评论中包含头表和*目标*部分的内容。有关示例，请参见[此示例 RFC](https://github.com/tensorflow/community/pull/5)。包括共同作者、审查者和发起人的 GitHub 句柄。

    在 PR 的顶部，标识评论的期限。期限应为自 PR 发布起*至少两周*。

5. 通过开发者邮寄名单向开发者发送简要说明、PR 链接和审查请求。请遵循[此示例](https://groups.google.com/a/tensorflow.org/forum/#!topic/developers/PIChGLLnpTE)所示电子邮件的格式。

6. 发起人将在 RFC PR 发布后的两周内请求召开审查委员会会议。如果讨论过程积极踊跃地提出了问题，请等到问题解决后再进行审查。审查会议的目的是解决小问题；应提前在重大问题上取得共识。

7. 会议可以批准或拒绝 RFC，也可以待更改后重新审查。批准的 RFC 将合并到 [community/rfcs](https://github.com/tensorflow/community/tree/master/rfcs) 中， 而 RFC 被拒的 PR 则会被关闭。

## RFC 参与者

RFC 流程涉及到许多人员：

- **RFC 作者** - 编写 RFC 并致力于在整个流程中倡导 RFC 的一名或多名社区成员

- **RFC 发起人** - 发起 RFC 并在 RFC 审查过程中提供支持的维护者

- **审查委员会** - 一组负责建议是否采纳 RFC 的维护者

- 任何**社区成员**都可以通过提供有关 RFC 是否满足其需求的反馈来参与该流程。

### RFC 发起人

发起人是负责确保 RFC 流程获得最佳结果的项目维护者。职责包括：

- 倡导拟议的设计。
- 引导 RFC 符合现有的设计和样式惯例。
- 引导审查委员会达成富有成效的共识。
- 当审查委员会请求更改时，确保更改得到妥善实施并寻求委员会成员的后续批准。
- 当 RFC 经批准待实现时：
    - 确保建议的实现方法符合设计。
    - 与相关方进行协调以成功落实实现方案。

### RFC 审查委员会

审查委员会在协商一致的基础上决定批准、拒绝还是请求更改。责任包括：

- 确保考虑到公众反馈的实质性内容。
- 添加其会议记录作为 PR 评论。
- 提供其做出决定的理由。

审查委员会的人员构成可能会因每个项目特定的治理方式和领导而异。对于核心 TensorFlow，委员会将由在相关领域具有专业知识的 TensorFlow 项目贡献者组成。

### 社区成员和 RFC 流程

RFC 的目的是确保 TensorFlow 的新变更能够很好地表示和传达社区的想法。社区成员有责任参与他们对其结果感兴趣的 RFC 的审查工作。

对 RFC 感兴趣的社区成员应：

- 尽早**提供反馈**，以留出足够的考虑时间。
- 先**通读 RFC**，然后再提供反馈。
- 以**文明且具有建设性**的方式提供反馈。

## 实现新功能

RFC 一经批准，即可开始实现 RFC。

如果您正在编写用于实现 RFC 的新代码：

- 确保您了解 RFC 中批准的功能和设计。在开始工作之前，提出问题并讨论方法。
- 新功能必须包括新的单元测试，以验证该功能是否按预期工作。建议在编写代码之前先编写这些测试。
- 遵循 [TensorFlow 代码样式指南](#tensorflow-code-style-guide)
- 添加或更新相关的 API 文档。在新文档中引用 RFC。
- 遵循您正在贡献的项目仓库内的 `CONTRIBUTING.md` 文件所列的任何其他准则。
- 先运行单元测试，然后再提交代码。
- 与 RFC 发起人合作以成功落实新代码。

## 保持高标准

我们鼓励和感谢每一位贡献者的贡献，但也同时有意地保持着较高的 RFC 接受门槛。在以下任何一个阶段，新功能都可能会被拒绝或要求重大修改：

- 相关邮寄名单的初始设计对话。
- 未能招募到发起人。
- 反馈阶段存在重大异议。
- 在设计审查期间未能达成共识。
- 在实现过程中出现问题（例如：无法实现向后兼容性、在维护方面存在疑虑）。

如果流程运行得当，则 RFC 失败情况应发生在早期而非后期。RFC 经批准不能作为承诺实现的保证，并且是否接受建议的 RFC 实现仍受常规代码审查流程的约束。

如果您对此流程有任何疑问，请随时通过开发者邮寄名单提问，或在 [tensorflow/community](https://github.com/tensorflow/community/tree/master/rfcs) 中提交问题。
