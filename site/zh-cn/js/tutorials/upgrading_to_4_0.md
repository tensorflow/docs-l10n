# 升级到 TensorFlow.js 4.0

## TensorFlow.js 4.0 变化

[此处](https://github.com/tensorflow/tfjs/releases)提供了版本说明。此版本将 TypeScript 升级到 4.8.4，将 `@webgpu/types` 升级到 0.1.21。如果您不使用 TypeScript，则无需阅读此文档即可更新到 4.0。

### 重大变更

此版本对于使用 `typescript<4.4` 的项目是破坏性的。所有其他项目应不受影响。

## 从 3.x 升级代码

### 对于 TypeScript &gt;= 4.4

此版本中没有对 API 进行重大变更，因此使用 `typescript>=4.4` 的项目可以将其视为次要版本，无需进行任何变更即可升级。

### 对于 TypeScript &lt; 4.4

使用 `typescript<4.4` 时，会出现以下错误。

```
node_modules/@webgpu/types/dist/index.d.ts:587:16 - error TS2304: Cannot find name 'PredefinedColorSpace'.

587   colorSpace?: PredefinedColorSpace;
                   ~~~~~~~~~~~~~~~~~~~~
...
```

要解决此问题，请将 TypeScript 升级到 4.4.2 或更高版本，或者将包含以下内容的文件 `predefined_color_space.d.ts`（名称和路径可以更改）添加到您的项目中以定义缺少的类型。在 TypeScript 升级到 4.4 或更高版本后可以移除此文件。

**predefined_color_space.d.ts**

```typescript
type PredefinedColorSpace = "display-p3" | "srgb";
```

### 对于 TypeScript &lt; 3.6

`typescript<3.6` 具有以下附加错误。

```
node_modules/@tensorflow/tfjs-core/dist/engine.d.ts:127:9 - error TS1086: An accessor cannot be declared in an ambient context.

127     get backend(): KernelBackend;
            ~~~~~~~
...
```

启用 [`skipLibCheck`](https://www.typescriptlang.org/tsconfig#skipLibCheck) 以抑制此错误，或者至少升级到 TypeScript 3.6.2 来修正此错误（还需要应用上述对 `PredefinedColorSpace` 的修正）。
