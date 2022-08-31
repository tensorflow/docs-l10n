# TensorFlow 설명서에 참여

TensorFlow는 문서 기여를 환영합니다. 문서를 개선하면 TensorFlow 라이브러리 자체도 개선됩니다. tensorflow.org의 문서는 다음 범주로 분류됩니다.

- *API 참조* — [API 참조 문서](https://www.tensorflow.org/api_docs/)는 [TensorFlow source code](https://github.com/tensorflow/tensorflow)에 있는 docstrings로 부터 시작되었습니다.
- *설명 문서* -TensorFlow 코드의 일부가 아닌 [튜토리얼](https://www.tensorflow.org/tutorials) , [가이드](https://www.tensorflow.org/guide) 및 기타 글입니다. 이 문서는 [tensorflow / docs](https://github.com/tensorflow/docs) GitHub 저장소에 있습니다.
- *커뮤니티 번역* — 커뮤니타에서 번역한 가이드와 튜토리얼입니다. 모든 커뮤니티 번역은 [tensorflow/docs](https://github.com/tensorflow/docs/tree/master/site) 리포지토리에 있습니다.

일부 [TensorFlow 프로젝트](https://github.com/tensorflow) 는 일반적으로 `docs/` 디렉토리에있는 코드 근처 별도의 저장소에 문서 소스 파일을 보관합니다. 기여하기 위해서는 프로젝트의 `CONTRIBUTING.md` 파일을 보거나 관리자에게 연락하십시오.

TensorFlow 문서 커뮤니티에 참여하려면 다음과 같이 합니다.

- [tensorflow / docs](https://github.com/tensorflow/docs) GitHub 저장소를 확인하세요.
- [TensorFlow 포럼](https://discuss.tensorflow.org/)에서 [설명서](https://discuss.tensorflow.org/tag/docs) 태그를 따라가세요.

## API 참조

자세한 내용은 [TensorFlow API 설명서 기여자 가이드](docs_ref.md)를 참조하세요. [소스 파일](https://www.tensorflow.org/code/tensorflow/python/)을 찾고 심볼의 <a href="https://www.python.org/dev/peps/pep-0257/" class="external">독스트링</a>을 편집하는 방법을 보여줍니다. ​tensorflow.org의 많은 API 참조 페이지에는 기호가 정의 된 소스 파일에 대한 링크가 포함되어 있습니다. 독스트링은 <a href="https://help.github.com/en/articles/about-writing-and-formatting-on-github" class="external">Markdown</a>을 지원하며 <a href="http://tmpvar.com/markdown.html" class="external">Markdown 미리 보기</a> 프로그램을 사용하여 (대략적으로) 미리 볼 수 있습니다.

### 버전과 분기

사이트의 [API 참조](https://www.tensorflow.org/api_docs/python/tf) 버전은 기본적으로 안정적인 최신 바이너리로 설정됩니다. 이는 `pip install tensorflow` 설치된 패키지와 일치합니다.

기본 TensorFlow package는 <a>tensorflow/tensorflow</a> 저장소의 안전한 분기 <code>rX.x</code>로 부터 만들어집니다. 참조 문서는 소스 코드의 <a href="https://www.tensorflow.org/code/tensorflow/python/" class="external">Python</a>, <a href="https://www.tensorflow.org/code/tensorflow/cc/" class="external">C++</a>, and <a href="https://www.tensorflow.org/code/tensorflow/java/" class="external">Java</a>를 위한 코드 주석과 독 스트링으로 부터 시작됩니다.

TensorFlow 문서의 이전 버전은 TensorFlow Docs 저장소에서 [rX.x 분기](https://github.com/tensorflow/docs/branches) 로 제공됩니다. 이러한 분기는 새 버전이 출시 될 때 추가됩니다.

### API 문서 빌드하기

참고 :이 단계는 API 독 스트링을 편집하거나 미리 보는 데 필요하지 않으며 tensorflow.org에서 사용되는 HTML을 생성하는 데만 필요합니다.

#### Python 참조

`tensorflow_docs` 패키지에는 [Python API 참조 문서](https://www.tensorflow.org/api_docs/python/tf) 용 생성기가 포함되어 있습니다. 설치:

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">pip install git+https://github.com/tensorflow/docs</code>
</pre>

TensorFlow 2 참조 문서를 생성하려면 `tensorflow/tools/docs/generate2.py` 스크립트를 사용하세요.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">git clone https://github.com/tensorflow/tensorflow tensorflow</code>
<code class="devsite-terminal">cd tensorflow/tensorflow/tools/docs</code>
<code class="devsite-terminal">pip install tensorflow</code>
<code class="devsite-terminal">python generate2.py --output_dir=/tmp/out</code>
</pre>

참고 :이 스크립트는 *설치된* TensorFlow 패키지를 사용하여 문서를 생성하고 TensorFlow 2.x에서만 작동합니다.

## 서술 문서

TensorFlow [가이드](https://www.tensorflow.org/guide) 및 [가이드](https://www.tensorflow.org/tutorials) 는 <a href="https://guides.github.com/features/mastering-markdown/" class="external">Markdown</a> 파일 및 대화 형 <a href="https://jupyter.org/" class="external">Jupyter</a> 노트북으로 작성됩니다. <a href="https://colab.research.google.com/notebooks/welcome.ipynb" class="external">Google Colaboratory를</a> 사용하여 브라우저에서 노트북을 실행할 수 있습니다. [tensorflow.org](https://www.tensorflow.org) 의 내러티브 문서는 <a href="https://github.com/tensorflow/docs" class="external">tensorflow / docs</a> `master` 브랜치에서 빌드됩니다. 이전 버전은 `rX.x` 릴리스 브랜치의 GitHub에서 사용할 수 있습니다.

### 간단한 변경

Markdown 파일에 대한 간단한 문서 업데이트를 만드는 가장 쉬운 방법은 GitHub의 <a href="https://help.github.com/en/articles/editing-files-in-your-repository" class="external">웹 기반 파일 편집기를 사용하는 것</a> 입니다. 찾아보기 [tensorflow / 문서](https://github.com/tensorflow/docs/tree/master/site/en) 저장소 약은에 해당하는 마크 다운을 찾을 수 <a href="https://www.tensorflow.org">tensorflow.org의</a> URL 구조. 파일보기의 오른쪽 상단에서 연필 아이콘을 클릭합니다. <svg version="1.1" width="14" height="16" viewbox="0 0 14 16" class="octicon octicon-pencil" aria-hidden="true"></svg><path fill-rule="evenodd" d="M0 12v3h3l8-8-3-3-8 8zm3 2H1v-2h1v1h1v1zm10.3-9.3L12 6 9 3l1.3-1.3a.996.996 0 0 1 1.41 0l1.59 1.59c.39.39.39 1.02 0 1.41z"></path> 파일 편집기를 엽니 다. 파일을 편집 한 다음 새 풀 요청을 제출하십시오.

### 로컬 Git 리포지토리 설정하기

다중 파일 편집 또는 더 복잡한 업데이트의 경우 로컬 Git 워크 플로를 사용하여 풀 요청을 만드는 것이 좋습니다.

참고 : <a href="https://git-scm.com/" class="external">Git</a> 은 소스 코드의 변경 사항을 추적하는 데 사용되는 오픈 소스 버전 제어 시스템 (VCS)입니다. <a href="https://github.com" class="external">GitHub</a> 는 Git에서 작동하는 협업 도구를 제공하는 온라인 서비스입니다. GitHub 계정을 설정하고 시작하려면 <a href="https://help.github.com" class="external">GitHub 도움말</a> 을 참조하십시오.

다음 Git 단계는 로컬 프로젝트를 처음 설정할 때만 필요합니다.

#### tensorflow/docs 리포지토리 포크하기

<a href="https://github.com/tensorflow/docs" class="external">tensorflow / docs</a> GitHub 페이지에서 *Fork* 버튼을 클릭합니다. <svg class="octicon octicon-repo-forked" viewbox="0 0 10 16" version="1.1" width="10" height="16" aria-hidden="true"></svg><path fill-rule="evenodd" d="M8 1a1.993 1.993 0 0 0-1 3.72V6L5 8 3 6V4.72A1.993 1.993 0 0 0 2 1a1.993 1.993 0 0 0-1 3.72V6.5l3 3v1.78A1.993 1.993 0 0 0 5 15a1.993 1.993 0 0 0 1-3.72V9.5l3-3V4.72A1.993 1.993 0 0 0 8 1zM2 4.2C1.34 4.2.8 3.65.8 3c0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2zm3 10c-.66 0-1.2-.55-1.2-1.2 0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2zm3-10c-.66 0-1.2-.55-1.2-1.2 0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2z"></path> GitHub 계정에서 자신의 저장소 사본을 만들 수 있습니다. 분기 된 후에는 업스트림 TensorFlow 저장소를 사용하여 저장소 사본을 최신 상태로 유지할 책임이 있습니다.

#### 리포지토리 복제하기

당신의 원격 리포지토리 <var>username</var>/docs 에서 당신의 로컬 기기로 다운로드 하십시오. 이 곳이 당신이 작얼할 디렉토리 입니다.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">git clone git@github.com:&lt;var&gt;username&lt;/var&gt;/docs</code>
<code class="devsite-terminal">cd ./docs</code>
</pre>

#### 최신 상태로 유지하려면 업스트림 리포지토리 추가 (선택 사항)

로컬 리포지토리를 `tensorflow/docs` 와 동기화 상태로 유지하려면 *업스트림* 원격을 추가하여 최신 변경 사항을 다운로드하세요.

참고: 기여  *전에* 당신의 로컬 리포지토리 업데이트를 확인하십시오. 업스트림에 대한 정기적 싱크는 풀 요청시 <a href="https://help.github.com/articles/resolving-a-merge-conflict-using-the-command-line" class="external">merge conflict</a>를 줄여줍니다.

다음과 같이 리모트를 추가합니다.

<pre class="prettyprint lang-bsh">&lt;code class="devsite-terminal"&gt;git remote add upstream git@github.com:tensorflow/docs.git&lt;/code&gt;

# View remote repos
&lt;code class="devsite-terminal"&gt;git remote -v&lt;/code&gt;
origin    git@github.com:&lt;var&gt;username&lt;/var&gt;/docs.git (fetch)
origin    git@github.com:&lt;var&gt;username&lt;/var&gt;/docs.git (push)
upstream  git@github.com:tensorflow/docs.git (fetch)
upstream  git@github.com:tensorflow/docs.git (push)
</pre>

업데이트하려면 다음과 같이 합니다.

<pre class="prettyprint lang-bsh">&lt;code class="devsite-terminal"&gt;git checkout master&lt;/code&gt;
&lt;code class="devsite-terminal"&gt;git pull upstream master&lt;/code&gt;

&lt;code class="devsite-terminal"&gt;git push&lt;/code&gt;  # Push changes to your GitHub account (defaults to origin)
</pre>

### GitHub 워크플로

#### 1. 새로운 분기를 만듭니다.

`tensorflow/docs` 에서 리포지토리를 업데이트 한 후 로컬 *마스터* 브랜치에서 새 브랜치를 만듭니다.

<pre class="prettyprint lang-bsh">&lt;code class="devsite-terminal"&gt;git checkout -b &lt;var&gt;feature-name&lt;/var&gt;&lt;/code&gt;

&lt;code class="devsite-terminal"&gt;git branch&lt;/code&gt;  # List local branches
  master
* &lt;var&gt;feature-name&lt;/var&gt;
</pre>

#### 2. 변경을 수행합니다.

좋아하는 편집기에서 파일을 편집하고 [TensorFlow 문서 스타일 가이드](./docs_style.md) 를 따르세요.

파일 변경 사항을 커밋합니다.

<pre class="prettyprint lang-bsh"># View changes
&lt;code class="devsite-terminal"&gt;git status&lt;/code&gt;  # See which files have changed
&lt;code class="devsite-terminal"&gt;git diff&lt;/code&gt;    # See changes within files

&lt;code class="devsite-terminal"&gt;git add &lt;var&gt;path/to/file.md&lt;/var&gt;&lt;/code&gt;
&lt;code class="devsite-terminal"&gt;git commit -m "Your meaningful commit message for the change."&lt;/code&gt;
</pre>

필요에 따라 더 많은 커밋을 추가합니다.

#### 3. 풀 요청을 생성합니다.

로컬 브랜치를 원격 GitHub 저장소 (github.com/ <var>username</var> / docs)에 업로드합니다.

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">git push</code>
</pre>

push가 완료된 후, 업스트림 리포지토리에 대한 풀 요청을 자동 제출하는 URL 메세지가 보일것입니다. 그렇지 않다면 <a href="https://github.com/tensorflow/docs" class="external">tensorflow/docs</a> 리포지토리 혹은 당신의 리포지토리, 깃허브는 당신에게 풀 요청을 해야한다는 것을 알려줄 것입니다.

#### 4. 검토하기

유지 관리자 및 기타 기여자가 귀하의 풀 요청을 검토 할 것입니다. 토론에 참여하고 요청 된 사항을 변경하십시오. 풀 요청이 승인되면 업스트림 TensorFlow 문서 리포지토리에 병합됩니다.

성공: TensorFlow 문서에 대한 변경 사항이 승인되었습니다.

GitHub 리포지토리에서 [tensorflow.org](https://www.tensorflow.org) 를 업데이트하는 별도의 게시 단계가 있습니다. 일반적으로 변경 사항은 일괄 처리되고 사이트는 정기적으로 업데이트됩니다.

## 대화형 노트북

GitHub의 <a href="https://help.github.com/en/articles/editing-files-in-your-repository" class="external">웹 기반 파일 편집기를 사용</a> 하여 노트북 JSON 파일을 편집 할 수 있지만 잘못된 JSON으로 인해 파일이 손상 될 수 있으므로 권장하지 않습니다. 풀 요청을 제출하기 전에 노트북을 테스트해야합니다.

<a href="https://colab.research.google.com/notebooks/welcome.ipynb" class="external">Google Colaboratory</a>  작성과 실행이 편한 는 노트북 환경을 만들었습니다. 노트북의 GitHub는 다음의 Colab URL을 통해 노트북에 로딩됩니다.<a href="https://github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/classification.ipynb">https://github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/classification.ipynb</a><br> can be loaded into Google Colab at this URL: <a href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/classification.ipynb">https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/classification.ipynb</a>

<!-- github.com path intentionally formatted to hide from import script. -->

GitHub에서 노트북을 검색 할 때이 URL 대체를 수행하는 <a href="https://chrome.google.com/webstore/detail/open-in-colab/iogfkhleblhcpcekbiedikdehleodpjo" class="external">Open in Colab</a> Chrome 확장 프로그램이 있습니다. 상단 버튼은 항상 TensorFlow Docs `master` 브랜치에 연결되므로 저장소 포크에서 노트북을 열 때 유용합니다.

### 노트북 형식 지정하기

노트북 서식 도구를 사용하면 Jupyter 노트북 소스 차이를 일관되고 쉽게 검토할 수 있습니다. 노트북 작성 환경은 파일 출력, 들여쓰기, 메타데이터 및 기타 지정되지 않은 필드와 관련하여 차이가 있기 때문입니다. `nbfmt`는 TensorFlow 문서 Colab 워크플로에 대한 환경 설정과 함께 독자적인 기본값을 사용합니다. 노트북 형식을 지정하려면 <a href="https://github.com/tensorflow/docs/tree/master/tools/tensorflow_docs/tools/" external="class">TensorFlow 문서 노트북 도구</a>를 설치하고 `nbfmt` 도구를 실행하세요.

```
# Install the tensorflow-docs package:
$ python3 -m pip install -U [--user] git+https://github.com/tensorflow/docs

$ python3 -m tensorflow_docs.tools.nbfmt [options] notebook.ipynb [...]
```

TensorFlow 문서 프로젝트의 경우, 출력 셀이 *없는* 노트북이 실행되고 테스트됩니다. 저장된 출력 셀이 *있는* 노트북은 그대로 게시됩니다. `nbfmt`는 노트북 상태를 유지하고 `--remove_outputs` 옵션을 사용하여 명시적으로 출력 셀을 제거합니다.

새 노트북을 작성하려면 <a href="https://github.com/tensorflow/docs/blob/master/tools/templates/notebook.ipynb" external="class">TensorFlow 문서 노트북 템플릿</a>을 복사하고 편집하세요.

### Colab에서 편집하기

Google Colab 환경에서 셀을 두 번 클릭하여 텍스트 및 코드 블록을 편집하십시오. 텍스트 셀은 마크 다운을 사용하며 [TensorFlow 문서 스타일 가이드를](./docs_style.md) 따라야합니다.

Colab에서 *File &gt; Download .pynb*를 이용해 노트북 파일을 다운로드합니다. 이 파일을 자신의 [local Git repo](##set_up_a_local_git_repo)에 커밋하고 풀 요청을 보냅니다.

새 노트북을 작성하려면 <a href="https://github.com/tensorflow/docs/blob/master/tools/templates/notebook.ipynb" external="class">TensorFlow 노트북 템플리트를</a> 복사하고 편집하십시오.

### Colab-GitHub 워크플로

노트북 파일을 다운로드하고 로컬 Git 워크플로를 사용하는 대신 Google Colab에서 직접 포크된 GitHub 리포지토리를 편집하고 업데이트할 수 있습니다.

1. 당신의 Fork <var>username</var>/docs 리포지토리는 <a href="https://help.github.com/articles/creating-and-deleting-branches-within-your-repository" class="external">새 브랜치 만들기</a>를 위해 GitHub 웹의 UI를 사용합니다.
2. 편집 할 전자 필기장 파일로 이동하십시오.
3. Google Colab에서 노트북을 엽니 다. URL 스왑 또는 *Open in Colab* Chrome 확장 프로그램을 사용합니다.
4. Colab에서 노트북을 편집하십시오.
5. *파일&gt; GitHub에 사본 저장 ...을 사용하여* Colab에서 리포지토리에 대한 변경 사항을 커밋합니다. 저장 대화 상자는 적절한 저장소 및 분기에 링크되어야합니다. 의미있는 커밋 메시지를 추가합니다.
6. 저장 후, 당신의 리포지토리 혹은<a href="https://github.com/tensorflow/docs" class="external">tensorflow/docs</a> 리포지토리를 찾기 위해, GitHub는 새로운 풀 요청을 할 것을 알려줍니다.
7. 풀 요청은 관리자가 검토합니다.

성공: TensorFlow 문서에 대한 변경 사항이 승인되었습니다.

## 번역

TensorFlow 팀은 tensorflow.org에 대한 번역을 제공하기위해 커뮤니티나 업체와 함께합니다. 노트북에 대한 번역이나 다른 기술적인 내용은 <a class="external" href="https://github.com/tensorflow/docs-l10n">tensorflow/docs-l10n</a> GitHub 리포지토리에 있습니다. <a class="external" href="https://gitlocalize.com/tensorflow/docs-l10n">TensorFlow GitLocalize project</a>를 통해 풀 요청을 제출하시기 바랍니다.

영어 문서는 *source-of-truth*이며 번역된 문서는 가이드를 따라 최대한 가깝게 작성되어야 합니다. 즉, 번역은 텐서플로우를 사용하는 커뮤니티를 위해 작성되어야 합니다. 만약 영어의 술어, 문단, 스타일, 톤이 다른 언어로 바뀌지 않는다면 읽는 이 에게 적절하도록 번역을 해야합니다.

언어 지원은 사이트 지표 및 수요, 커뮤니티 지원, <a class="external" href="https://en.wikipedia.org/wiki/EF_English_Proficiency_Index">영어 능력</a> , 청중 선호도 및 기타 지표를 포함하되 이에 국한되지 않는 여러 요인에 의해 결정됩니다. 지원되는 각 언어에는 비용이 발생하므로 관리되지 않는 언어는 제거됩니다. 새로운 언어에 대한 지원은 <a class="external" href="https://blog.tensorflow.org/">TensorFlow 블로그</a> 또는 <a class="external" href="https://twitter.com/TensorFlow">Twitter</a> 에서 발표됩니다.

선호하는 언어가 지원되지 않는 경우 오픈 소스 기여자를위한 커뮤니티 포크를 유지하는 것을 환영합니다. 이들은 tensorflow.org에 게시되지 않습니다.
