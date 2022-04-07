# TensorFlow 코드에 기여하기

손실 함수 추가, 테스트 범위 개선 또는 주요 설계 변경에 대한 RFC 작성 등 기여자 가이드의 이 부분은 순조로운 출발을 도와줍니다. TensorFlow 개선에 대한 여러분의 관심과 노고에 감사 드립니다.

## 시작하기 전에

TensorFlow 프로젝트에 소스 코드를 제공하기 전에 프로젝트의 GitHub 리포지토리에서 `CONTRIBUTING.md` 파일을 살펴보기 바랍니다. 예를 들어 핵심 TensorFlow 리포지토리에 대한 [CONTRIBUTING.md](https://github.com/tensorflow/tensorflow/blob/master/CONTRIBUTING.md) 파일을 참조합니다. 모든 코드 제공자는 [제공자 라이선스 계약](https://cla.developers.google.com/clas)(CLA)에 동의해야 합니다.

중복 작업을 피하려면 중요하지 않은 기능에 대한 작업을 시작하기 전에 [현재](https://github.com/tensorflow/community/tree/master/rfcs) 또는 [제안된](https://github.com/tensorflow/community/tree/master/rfcs) RFC를 검토하고 TensorFlow 포럼의 개발자에게 문의하세요([developers@tensorflow.org](https://groups.google.com/u/1/a/tensorflow.org/g/developers)). 우리는 새로운 기능을 추가하기로 결정할 때 다소 선택적이며, 프로젝트에 기여하고 도움을 줄 수 있는 가장 좋은 방법은 알려진 문제를 해결하는 것입니다.

## 새로운 기여자에 조언

새로운 기여자는 TensorFlow 코드베이스에 대한 첫 번째 기여를 검색할 때 다음 태그를 찾아야 합니다. 새로운 기여자들은 "쉬운 첫 문제" 및 "기여 환영" 프로젝트를 먼저 다루는 것이 좋습니다. 이를 통해 기여자가 기여 워크플로에 익숙해지고 핵심 개발자는 기여자에게 익숙해 질 수 있습니다.

- [쉬운 첫 문제](https://github.com/tensorflow/tensorflow/labels/good%20first%20issue)
- [기여 환영](https://github.com/tensorflow/tensorflow/labels/stat%3Acontributions%20welcome)

보다 큰 규모의 문제나 새로운 기능을 다루기 위해 팀을 구성하고 싶다면 [developers@ group](https://groups.google.com/a/tensorflow.org/forum/#!forum/developers)에 이메일을 보내 현재 RFC 목록을 검토하세요.

## 코드 검토

새로운 기능, 버그 수정 및 기타 코드 베이스의 변경은 코드 검토를 받아야 합니다.

풀 요청으로 프로젝트에 기여된 코드를 검토하는 일은 TensorFlow 개발에서 중요한 부분을 차지합니다. 특히 자신이 사용할 가능성이 높은 기능의 경우 다른 개발자가 제출한 코드를 검토하는 것이 좋습니다.

코드 검토 과정에서 염두에 두어야 할 몇 가지 질문이 있습니다.

- *TensorFlow에서 이 코드가 필요한가?* 사용할 가능성이 있는가? TensorFlow 사용자로서 변경이 마음에 들고 사용할 의향이 있는가? 이 변경이 TensorFlow의 범위 내에 있는가? 새로운 기능을 유지하는 데 비용을 들일만한 가치가 있는가?

- *코드가 TensorFlow API와 일치하는가?* 공개 함수, 클래스 및 매개변수가 직관적으로 설계되었고 이름이 잘 지정되었는가?

- *문서가 포함되어 있는가?* 모든 공개 함수, 클래스, 매개변수, 반환 유형 및 저장된 속성이 TensorFlow 규칙에 따라 명명되었고 명확하게 문서화되었는가? 새로운 기능이 TensorFlow 문서에 설명되어 있고 가능한 경우 예가 제시되어 있는가? 문서가 올바르게 작성되었는가?

- *코드를 사람이 읽을 수 있는가?* 중복 수준이 낮은가? 명확성 또는 일관성을 위해 변수 이름에 개선이 필요한가? 주석을 추가해야 하는가? 도움이 되지 않거나 불필요하여 주석을 제거해야 하는가?

- *코드가 효율적인가?* 보다 효율적으로 실행되도록 쉽게 다시 작성할 수 있는가?

- 코드가 이전 버전의 TensorFlow와 *호환*되는가?

- 새 코드가 다른 라이브러리에 *새로운 종속성*을 추가하는가?

## 테스트를 수행하고 테스트 범위 개선하기

고품질 단위 테스트는 TensorFlow 개발 프로세스의 초석입니다. 이를 위해 Docker 이미지를 사용합니다. 테스트 함수는 적절하게 이름이 지정되며 알고리즘의 유효성과 여러 코드 옵션을 확인하는 역할을 합니다.

모든 새로운 기능과 버그 수정에는 적절한 테스트 범위가 포함*되어야* 합니다. 또한 새로운 테스트 사례의 기여나 기존 테스트의 개선도 환영합니다. 기존 테스트가 완전하지 않은 것으로 확인되면 (현재 버그가 발생하지 않더라도) 문제를 제기하고 가능한 경우 풀 요청을 제출하세요.

각 TensorFlow 프로젝트의 테스트 절차에 대한 자세한 내용은 GitHub의 프로젝트 리포지토리에 있는 `README.md` 및 `CONTRIBUTING.md` 파일을 참조하세요.

특히 *적절한 테스트*에서 다음 사항에 의문을 제기합니다.

- *모든 공개 함수와 클래스*가 테스트되는가?
- *합당한 매개변수 세트*, 해당 값, 값 유형 및 조합이 테스트되는가?
- *코드가 올바르고* 코드가 수행하는 것으로*문서에 설명한 내용이 수행*되는지 테스트에서 검증하는가?
- 변경 사항이 버그 수정인 경우 *비회귀 테스트*가 포함되는가?
- 테스트가 *연속 통합* 빌드를 전달하는가?
- 테스트가 *모든 코드 라인을 포괄하는가?* 그렇지 않다면 예외가 합당하고 명백한가?

문제가 발견되면 기여자가 해당 문제를 이해하고 해결하도록 도와주세요.

## 오류 메시지 또는 로그 개선하기

오류 메시지와 로깅을 개선하는 기여를 환영합니다.

## 기여 워크플로

버그 수정, 새로운 개발, 테스트 개선과 같은 코드 기여는 모두 GitHub 중심 워크플로를 따릅니다. TensorFlow 개발에 참여하려면 GitHub 계정을 설정하고 다음을 수행합니다.

1. 작업하려는 리포지토리를 포크합니다. 프로젝트 리포지토리 페이지로 이동하고 *Fork* 버튼을 사용합니다. 그러면 사용자 이름 아래에 리포지토리 사본이 만들어집니다. 리포지토리를 포크하는 방법에 대한 자세한 내용은 [이 가이드](https://help.github.com/articles/fork-a-repo/)를 참조하세요.

2. 리포지토리를 로컬 시스템에 복제합니다.

    `$ git clone git@github.com:your-user-name/project-name.git`

3. 작업을 유지할 새 분기를 만듭니다.

    `$ git checkout -b new-branch-name`

4. 새 코드로 작업합니다. 테스트를 작성하고 실행합니다.

5. 변경 사항을 커밋합니다.

    `$ git add -A`

    `$ git commit -m "commit message here"`

6. GitHub 리포지토리로 변경 사항을 푸시합니다.

    `$ git push origin branch-name`

7. *Pull Request* (PR)를 엽니다. GitHub의 원래 프로젝트 리포지토리로 이동합니다. 풀 요청을 열 것인지 묻는 최근 푸시된 분기에 대한 메시지가 있을 겁니다. 메시지 내용을 따르고 *전체 리포지토리를 비교*한 다음 PR을 제출합니다. 그러면 커밋한 사람에게 이메일이 발송됩니다. 더 많이 드러나도록 메일 그룹에도 이메일을 보낼 수 있습니다. (자세한 내용은 [PR에 관한 GitHub 가이드](https://help.github.com/articles/creating-a-pull-request-from-a-fork)를 참조하세요.)

8. 관리자와 다른 기여자가 *해당 PR을 검토*합니다. 대화에 참여하고 *요청된 변경을 수행*합니다. PR이 승인되면 코드가 병합됩니다.

*다음 기여 작업으로 넘어가기 전에* 로컬 리포지토리가 최신 상태인지 확인합니다.

1. 업스트림 리모트를 설정합니다. (매번이 아니라 프로젝트당 한 번만 수행하면 됩니다.)

    `$ git remote add upstream git@github.com:tensorflow/project-repo-name`

2. 로컬 마스터 분기로 전환합니다.

    `$ git checkout master`

3. 업스트림에서 변경 사항을 풀다운합니다.

    `$ git pull upstream master`

4. 변경 사항을 GitHub 계정에 푸시합니다(선택 사항이지만 권장함).

    `$ git push origin master`

5. 새 작업을 시작하는 경우 새 분기를 만듭니다.

    `$ git checkout -b branch-name`

추가 `git` 및 GitHub 리소스:

- [Git 문서](https://git-scm.com/documentation)
- [Git 개발 워크플로](https://docs.scipy.org/doc/numpy/dev/development_workflow.html)
- [병합 충돌 해결](https://help.github.com/articles/resolving-a-merge-conflict-using-the-command-line/)

## 기여자 체크리스트

- [기여 지침](https://github.com/tensorflow/tensorflow/blob/master/CONTRIBUTING.md) 읽기
- [행동 강령](https://github.com/tensorflow/tensorflow/blob/master/CODE_OF_CONDUCT.md)을 읽습니다.
- [기여자 라이선스 계약(CLA)](https://cla.developers.google.com/)에 동의합니다.
- 변경 사항이 [지침](https://github.com/tensorflow/tensorflow/blob/master/CONTRIBUTING.md#general-guidelines-and-philosophy-for-contribution)과 일치하는지 확인합니다.
- 변경 사항이 [TensorFlow 코딩 스타일](https://www.tensorflow.org/community/contribute/code_style)과 일치하는지 확인합니다.
- [단위 테스트 실행](https://github.com/tensorflow/tensorflow/blob/master/CONTRIBUTING.md#running-unit-tests)
