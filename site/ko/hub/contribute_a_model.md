<!--* freshness: { owner: 'maringeo' reviewed: '2021-11-25' review_interval: '6 months' } *-->

# 풀 요청 제출하기

이 페이지는 Markdown 문서 파일이 포함된 풀 요청을 [tensorflow/tfhub.dev](https://github.com/tensorflow/tfhub.dev) GitHub 저장소에 제출하는 내용을 다룹니다. 먼저 Markdown 파일을 작성하는 방법에 대한 자세한 내용은 [문서 작성 가이드](writing_documentation.md)를 참조하세요.

## GitHub 작업 확인

[tensorflow/tfhub.dev](https://github.com/tensorflow/tfhub.dev) 저장소는 GitHub 작업을 사용하여 PR의 파일 형식을 확인합니다. PR을 검증하는 데 사용되는 워크플로는 [.github/workflows/contributions-validator.yml](https://github.com/tensorflow/tfhub.dev/blob/master/.github/workflows/contributions-validator.yml)에 정의되어 있습니다. 워크플로 외부의 자체 분기에서 유효성 검사기 스크립트를 실행할 수 있지만 모든 pip 패키지 종속성이 올바르게 설치되어 있어야 합니다.

처음 기여자는 [GitHub 정책](https://github.blog/changelog/2021-04-22-github-actions-maintainers-must-approve-first-time-contributor-workflow-runs/)에 따라 저장소 관리자의 승인이 있어야만 자동화된 검사를 실행할 수 있습니다. 게시자는 작은 PR 수정 오타를 제출하거나 모델 문서를 개선하거나, 후속 PR에 대한 자동 검사를 실행할 수 있도록 게시자 페이지만 포함하는 PR을 첫 번째 PR로 제출하는 것이 좋습니다.

중요: 풀 요청은 자동 검사를 통과해야 검토를 받게 됩니다!

## PR 제출하기

전체 마크다운 파일은 다음 방법 중 하나를 사용하여 [tensorflow/tfhub.dev](https://github.com/tensorflow/tfhub.dev/tree/master)의 마스터 브랜치로 가져올 수 있습니다.

### Git CLI 제출

식별된 마크다운 파일 경로를 `assets/docs/publisher/model/1.md`라고 가정하면, 표준 Git[Hub] 단계에 따라 새로 추가된 파일을 사용하여 새 Pull Request를 생성할 수 있습니다.

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

### GitHub GUI 제출

좀 더 간단한 제출 방법은 GitHub 그래픽 사용자 인터페이스를 사용하는 것입니다. GitHub에서는 GUI를 통해 직접 [새 파일](https://help.github.com/en/github/managing-files-in-a-repository/creating-new-files) 또는 [파일 편집](https://help.github.com/en/github/managing-files-in-a-repository/editing-files-in-your-repository)을 위한 PR을 생성할 수 있습니다.

1. [TensorFlow Hub GitHub 페이지](https://github.com/tensorflow/tfhub.dev)에서 `Create new file` 버튼을 누릅니다.
2. `assets/docs/publisher/model/1.md`의 올바른 파일 경로를 설정합니다.
3. 기존 마크다운을 복사하여 붙여 넣습니다.
4. 하단에서 "Create a new branch for this commit and start a pull request"를 선택합니다.
