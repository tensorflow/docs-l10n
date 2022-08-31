# TFX クラウドソリューション

TFX を適用して、ニーズを満たすソリューションを構築する方法についての洞察をお探しですか？これらの詳細な記事とガイドが役立つかもしれません！

注意: これらの記事では、TFX が重要な部分であるが、唯一の部分ではない完全なソリューションについて説明しています。ほとんどの場合、これは実際のデプロイに当てはまります。したがって、これらのソリューションを自分で実装するには、TFX のみでなくそれ以上のものが必要になります。主な目標は、他の人があなたと同様の要件を満たす可能性のあるソリューションをどのように実装しているかについての洞察を提供することであり、TFX のクックブックまたは承認されたアプリケーションのリストとしては機能しません。

## ほぼリアルタイムのアイテムマッチングのための機械学習システムのアーキテクチャ

このドキュメントを使用して、アイテムの埋め込みを学習して提供する機械学習（ML）ソリューションのアーキテクチャについて学びます。埋め込みは、顧客が類似していると見なすアイテムを理解するのに役立ちます。これにより、アプリケーションでリアルタイムの「類似アイテム」の提案を提供できます。このソリューションは、データセット内の類似した曲を識別し、この情報を使用して曲の推奨を行う方法を紹介しています。 <a href="https://cloud.google.com/solutions/real-time-item-matching" class="external" target="_blank">続きを読む</a>

## 機械学習のためのデータ前処理: オプションと推奨事項

この 2 部構成の記事では、機械学習（ML）のデータエンジニアリングと特徴エンジニアリングのトピックについて説明します。このパート 1 では、Google Cloud の機械学習パイプラインでデータを前処理するためのベストプラクティスについて説明します。この記事では、TensorFlow とオープンソースの TensorFlow Transform（tf.Transform）ライブラリを使用してデータを準備し、モデルをトレーニングし、予測のためにモデルを提供することに焦点を当てています。このパートでは、機械学習のためにデータを前処理する際の課題に焦点を当て、Google Cloud でデータ変換を効果的に実行するためのオプションとシナリオを示します。<a href="https://cloud.google.com/solutions/machine-learning/data-preprocessing-for-ml-with-tf-transform-pt1" class="external" target="_blank">パート 1</a> <a href="https://cloud.google.com/solutions/machine-learning/data-preprocessing-for-ml-with-tf-transform-pt2" class="external" target="_blank">パート 2</a>

## TFX、Kubeflow パイプライン、Cloud Build を使用した MLOps のアーキテクチャ

このドキュメントでは、TensorFlow Extended（TFX）ライブラリを使用した機械学習（ML）システムの全体的なアーキテクチャについて説明します。また、Cloud Build と Kubeflow パイプラインを使用して、ML システムの継続的インテグレーション（CI）、継続的デリバリー（CD）、および継続的トレーニング（CT）を設定する方法についても説明します。<a href="https://cloud.google.com/solutions/machine-learning/architecture-for-mlops-using-tfx-kubeflow-pipelines-and-cloud-build" class="external" target="_blank">続きを読む</a>

## MLOps: 機械学習における継続的デリバリーと自動化パイプライン

このドキュメントでは、機械学習（ML）システムの継続的インテグレーション（CI）、継続的デリバリー（CD）、および継続的トレーニング（CT）を実装および自動化するための手法について説明します。データサイエンスと ML は、複雑な現実世界の問題を解決し、業界を変革し、すべてのドメインで価値を提供するためのコア機能になりつつあります。<a href="https://cloud.google.com/solutions/machine-learning/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning" class="external" target="_blank">続きを読む</a>

## Google Cloud での MLOps 環境のセットアップ

このリファレンスガイドでは、Google Cloud の機械学習オペレーション（MLOps）環境のアーキテクチャの概要を説明します。**このガイドには、ここで説明する環境のプロビジョニングと構成のプロセスを順を追って説明している GitHub のハンズオンラボが付随**しています。事実上すべての業界が急速に加速するペースで機械学習（ML）を採用しています。 ML から価値を引き出すための重要な課題は、ML システムを効果的にデプロイして運用する方法を作成することです。このガイドは、機械学習（ML）および DevOps エンジニアを対象としています。<a href="https://cloud.google.com/solutions/machine-learning/setting-up-an-mlops-environment" class="external" target="_blank">続きを読む</a>

## MLOps ファンデーションの主な要件

AI 主導の組織は、データと機械学習を使用して最も困難な問題を解決し、成果を得ています。

McKinsey Global Institute によると*、「2025年までに価値を生み出すワークフローで AI を完全に吸収する企業は、+120％ のキャッシュフロー成長で 2030 年の世界経済を支配するでしょう」*と述べています。

しかし、今は簡単ではありません。機械学習（ML）システムには、適切に管理されていない場合に技術的負債を生み出す特別な力があります。<a href="https://cloud.google.com/blog/products/ai-machine-learning/key-requirements-for-an-mlops-foundation" class="external" target="_blank">続きを読む</a>

## Scikit-Learn を使用してクラウドにモデルカードを作成し、デプロイする方法

現在、機械学習モデルは、多くの困難なタスクを実行するために使用されています。 大きな可能性を秘めている ML モデルですが、同様に、使用法、構造、制限について疑問視されています。これらの疑問に対する回答を文書化することは、明確さと共通の理解をもたらすのに役立ちます。これらの目標を前進させるために、Google はモデルカードを導入しました。<a href="https://cloud.google.com/blog/products/ai-machine-learning/create-a-model-card-with-scikit-learn" class="external" target="_blank">続きを読む</a>

## TensorFlow データ検証を使用した機械学習のための大規模なデータの分析と検証

このドキュメントでは、TensorFlow Data Validation（TFDV）ライブラリを使用して、実験中のデータ探索と記述的分析を行う方法について説明します。データサイエンティストと機械学習（ML）エンジニアは、本番 ML システムで TFDV を使用して、連続トレーニング（CT）パイプラインで使用されるデータを検証し、予測サービスのために受信したデータのスキューと異常値を検出できます。**ハンズオンラボ**が含まれています。<a href="https://cloud.google.com/solutions/machine-learning/analyzing-and-validating-data-at-scale-for-ml-using-tfx" class="external" target="_blank">続きを読む</a>
