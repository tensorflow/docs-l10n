# TensorFlow.js 3.0으로 업그레이드

## TensorFlow.js 3.0 변경 사항

릴리스 노트는 [여기에서 확인할 수 있습니다](https://github.com/tensorflow/tfjs/releases). 이 릴리스는 TypeScript를 4.8.4로, `@webgpu/types`를 0.1.21로 업그레이드합니다. TypeScript를 사용하지 않는다면 이 문서를 읽지 않고 4.0으로 업데이트할 수 있습니다.

### 주요 변경 사항

이 릴리스는 `typescript<4.4`를 사용하는 프로젝트에 중요합니다. 모든 다른 프로젝트는 영향을 받지 않아야 합니다.

## 3.x부터 코드 업그레이드

### TypeScript 4.4 이상 버전의 경우

이 릴리스에는 중요한 API 변경 사항이 없으므로 `typescript >=4.4`를 사용하는 프로젝트는 이를 중요하지 않은 릴리스로 여길 수 있으며 변경 사항 없이 업그레이드할 수 있습니다.

### TypeScript 4.4 미만 버전의 경우

`typescript<4.4`를 사용하는 경우, 다음 오류가 발생합니다.

```
node_modules/@webgpu/types/dist/index.d.ts:587:16 - error TS2304: Cannot find name 'PredefinedColorSpace'.

587   colorSpace?: PredefinedColorSpace;
                   ~~~~~~~~~~~~~~~~~~~~
...
```

이를 해결하려면, TypeScript를 4.4.2 이상으로 업그레이드하거나 다음 콘텐츠를 통해 파일 `predefined_color_space.d.ts`(이름 및 경로는 변경될 수 있음)를 추가해 프로젝트가 누락된 유형을 정의하도록 합니다. TypeScript가 4.4 이상으로 업그레이드되면 이 파일을 삭제합니다.

**predefined_color_space.d.ts**

```typescript
type PredefinedColorSpace = "display-p3" | "srgb";
```

### TypeScript 3.6 미만 버전의 경우

`typescript<3.6`는 다음과 같은 추가 오류가 있습니다.

```
node_modules/@tensorflow/tfjs-core/dist/engine.d.ts:127:9 - error TS1086: An accessor cannot be declared in an ambient context.

127     get backend(): KernelBackend;
            ~~~~~~~
...
```

[`skipLibCheck`](https://www.typescriptlang.org/tsconfig#skipLibCheck)를 활성화하여 이 오류를 삭제하거나 적어도 TypeScript 3.6.2로 업그레이드하여 이를 해결합니다(`PredefinedColorSpace`에 대한 위의 해결책 또한 적용되어야 합니다).
