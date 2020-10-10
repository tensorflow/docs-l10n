# TensorFlow Federated

TensorFlow Federated（TFF）プラットフォームは、2 つのレイヤーで構成されています。

- [フェデレーテッドラーニング（FL）](federated_learning.md): 既存の Keras または非 Keras 機械学習モデルを TFF フレームワークにプラグインする高レベルインターフェース。フェデレーテッドラーニングアルゴリズムの詳細を学習することなく、フェデレーテッドトレーニングや評価などの基本タスクを実行することができます。
- [フェデレーテッドコア（FC）](federated_core.md): 強く型付けされた関数型プログラミング環境内で TensorFlow と分散型通信オペレータを組み合わせることで、カスタムフェデレーテッドアルゴリズムを簡潔に表現するための低レベルインターフェース。

まずは、次のチュートリアルをお読みください。これらのチュートリアルでは、実践的な例を使用しながら主な TFF コンセプトと API が説明されています。[インストール手順](install.md)に従って、TFF と使用するための環境を構成してください。

- [画像分類のフェデレーテッドラーニング](tutorials/federated_learning_for_image_classification.ipynb): フェデレーテッドラーニング（FL）API の主要部分を紹介し、TFF を使用して、MNIST のようなフェデレーテッドデータでフェデレーテッドラーニングをシミュレーションする方法を実演します。
- [テキスト生成のフェデレーテッドラーニング](tutorials/federated_learning_for_text_generation.ipynb): TFF の FL API を使用して、言語モデリングタスク用にシリアル化されたトレーニング済みのモデルを洗練する方法を実演します。
- [カスタムフェデレーテッドアルゴリズム、パート 1:フェデレーテッドコアの基礎](tutorials/custom_federated_algorithms_1.ipynb) および [パート 2: Part 2: Implementing フェデレーテッドアベレージングを実装する](tutorials/custom_federated_algorithms_2.ipynb): フェデレーテッドコア API（FC API）が提供する主なコンセプトとインターフェースを紹介し、単純なフェデレーテッドアベレージング トレーニングアルゴリズムの実装方法とフェデレーテッド評価の実施方法を実演します。
