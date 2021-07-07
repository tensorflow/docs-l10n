# 适用于非 TFX 用户的 Transform 库

Transform 可以作为独立的库使用。

- [开始使用 TensorFlow Transform](/tfx/transform/get_started)
- [TensorFlow Transform API 参考](/tfx/transform/api_docs/python/tft)

`tft` 模块文档是唯一与 TFX 用户有关的模块。`tft_beam` 模块仅在将 Transform 作为独立库使用时才相关。通常情况下，TFX 用户会构造一个 `preprocessing_fn`，其余的 Transform 库调用则由 Transform 组件完成。
