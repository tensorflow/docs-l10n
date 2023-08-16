# TensorFlow.js 4.0 へのアップグレード

## TensorFlow.js 4.0 での変更点

リリースノートは[こちらにあります](https://github.com/tensorflow/tfjs/releases)。このリリースでは、TypeScript が 4.8.4 に、`@webgpu/types` が 0.1.21 にアップグレードされました。TypeScript を使用しない場合は、このドキュメントを読まずに 4.0 に更新できます。

### 重大な変更

このリリースは、`typescript<4.4` を使用するプロジェクトに重大なリリースです。その他すべてのプロジェクトには影響ありません。

## コードを 3.x からアップグレードする

### TypeScript &gt;= 4.4 の場合

このリリースでは、API に重大な変更はありません。`typescript>=4.4` を使用するプロジェクトでは、このリリースをマイナーリリースとして捉え、変更を行わずにアップグレードすることができます。

### TypeScript &lt; 4.4 の場合

`typescript<4.4` を使用する場合に、以下のエラーが発生します。

```
node_modules/@webgpu/types/dist/index.d.ts:587:16 - error TS2304: Cannot find name 'PredefinedColorSpace'.

587   colorSpace?: PredefinedColorSpace;
                   ~~~~~~~~~~~~~~~~~~~~
...
```

これを修正するには、TypeScript を 4.4.2 以上にアップグレードします。または、以下の、欠落している型を定義する内容を含む `predefined_color_space.d.ts` ファイル（名前とパスを変更可能）をプロジェクトに追加し、TypeScript が 4.4 以上にアップグレードされた時点で、このファイルを削除するようにします。

**predefined_color_space.d.ts**

```typescript
type PredefinedColorSpace = "display-p3" | "srgb";
```

### TypeScript &lt; 3.6 の場合

`typescript<3.6` では、追加で以下のエラーが発生します。

```
node_modules/@tensorflow/tfjs-core/dist/engine.d.ts:127:9 - error TS1086: An accessor cannot be declared in an ambient context.

127     get backend(): KernelBackend;
            ~~~~~~~
...
```

このエラーを非表示にするには、[`skipLibCheck`](https://www.typescriptlang.org/tsconfig#skipLibCheck) を有効にします。または、最低限 TypeScript 3.6.2 にアップグレードすると修正されます（上の `PredefinedColorSpace` の修正も適用する必要があります）。
