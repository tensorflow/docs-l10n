# Upgrading to TensorFlow.js 4.0

## What’s changed in TensorFlow.js 4.0

Release notes are [available here](https://github.com/tensorflow/tfjs/releases). This release upgrades TypeScript to 4.8.4 and `@webgpu/types` to 0.1.21. If you don't use TypeScript, you can update to 4.0 without reading this doc.

### Breaking Changes

This release is breaking for projects that use `typescript<4.4`. All other projects should be unaffected.

## Upgrading Code from 3.x

### For TypeScript >= 4.4
No breaking API changes were made in this release, so projects that use `typescript>=4.4` can treat this as a minor release and upgrade
without any changes.

### For TypeScript < 4.4

When using `typescript<4.4`, the following error will occur.
```
node_modules/@webgpu/types/dist/index.d.ts:587:16 - error TS2304: Cannot find name 'PredefinedColorSpace'.

587   colorSpace?: PredefinedColorSpace;
                   ~~~~~~~~~~~~~~~~~~~~
...
```

To fix this, upgrade TypeScript to 4.4.2 or greater, or add the file `predefined_color_space.d.ts` (name and path can be changed) with the following contents to your project to define the missing type. Remove this file when TypeScript is upgraded to 4.4 or higher.

**predefined_color_space.d.ts**
```typescript
type PredefinedColorSpace = "display-p3" | "srgb";
```

### For TypeScript < 3.6
`typescript<3.6` has the following additional error.
```
node_modules/@tensorflow/tfjs-core/dist/engine.d.ts:127:9 - error TS1086: An accessor cannot be declared in an ambient context.

127     get backend(): KernelBackend;
            ~~~~~~~
...
```

Enable [`skipLibCheck`](https://www.typescriptlang.org/tsconfig#skipLibCheck) to suppress this error, or upgrade to at least TypeScript 3.6.2 to fix it (the above fix for `PredefinedColorSpace` will also need to be applied).
