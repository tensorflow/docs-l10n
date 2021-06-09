<!--* freshness: { owner: 'maringeo' reviewed: '2021-05-28' review_interval: '6 months' } *-->

# 모델 기여하기

This page is about adding Markdown documentation files to GitHub. For more information on how to write the Markdown files in the first place, please see the [writing model documentation guide](writing_model_documentation.md).

## 모델 제출하기

The complete Markdown files can be pulled into the master branch of [tensorflow/tfhub.dev](https://github.com/tensorflow/tfhub.dev/tree/master) by one of the following methods.

### Git CLI submission

Assuming the identified markdown file path is `assets/docs/publisher/model/1.md`, you can follow the standard Git[Hub] steps to create a new Pull Request with a newly added file.

TensorFlow Hub GitHub 리포지토리를 포크한 다음, [이 포크에서 Pull Request](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request-from-a-fork)를 TensorFlow Hub 마스터 브랜치로 생성합니다.

다음은 포크된 리포지토리의 마스터 브랜치에 새 파일을 추가하는 데 필요한 일반적인 CLI git 명령입니다.

```bash
git clone https://github.com/[github_username]/tfhub.dev.git
cd tfhub.dev
mkdir -p assets/docs/publisher/model
cp my_markdown_file.md ./assets/docs/publisher/model/1.md
git add *
git commit -m "Added model file."
git push origin master
```

### GitHub GUI submission

좀 더 간단한 제출 방법은 GitHub 그래픽 사용자 인터페이스를 사용하는 것입니다. GitHub에서는 GUI를 통해 직접 [새 파일](https://help.github.com/en/github/managing-files-in-a-repository/creating-new-files) 또는 [파일 편집](https://help.github.com/en/github/managing-files-in-a-repository/editing-files-in-your-repository)을 위한 PR을 생성할 수 있습니다.

1. [TensorFlow Hub GitHub 페이지](https://github.com/tensorflow/hub)에서 `Create new file` 버튼을 누릅니다.
2. Set the right file path: `assets/docs/publisher/model/1.md`
3. 기존 마크다운을 복사하여 붙여 넣습니다.
4. At the bottom, select "Create a new branch for this commit and start a pull request."
