# Como fazer upgrade para o TensorFlow.js 4.0

## O que mudou no TensorFlow.js 4.0

As notas de versão estão [disponíveis aqui](https://github.com/tensorflow/tfjs/releases). Essa versão atualiza o TypeScript para a versão 4.8.4 e o `@webgpu/types` para a versão 0.1.21. Se você não usa o TypeScript, pode atualizar para a versão 4.0 sem ler esta documentação.

### Alterações interruptivas

Esta versão é interruptiva para projetos que usem `typescript<4.4`. Nenhum outro projeto deverá ser impactado.

## Como fazer upgrade de código 3.x

### Para TypeScript versão 4.4 ou superior

Nenhuma alteração interruptiva de API foi feita nesta versão. Portanto, projetos que usem `typescript>=4.4` podem considerar esta versão como secundária e fazer o upgrade sem nenhuma mudança.

### Para TypeScript abaixo da versão 4.4

Ao usar `typescript<4.4`, ocorrerá o seguinte erro:

```
node_modules/@webgpu/types/dist/index.d.ts:587:16 - error TS2304: Cannot find name 'PredefinedColorSpace'.

587   colorSpace?: PredefinedColorSpace;
                   ~~~~~~~~~~~~~~~~~~~~
...
```

Para resolver esse problema, faça upgrade do TypeScript para a versão 4.4.2 ou superior, ou então adicione o arquivo `predefined_color_space.d.ts` (o nome e o caminho podem ser alterados) com o conteúdo abaixo ao seu projeto para definir o tipo ausente. Remova esse arquivo quando for feito upgrade do TypeScript para a versão 4.4 ou superior.

**predefined_color_space.d.ts**

```typescript
type PredefinedColorSpace = "display-p3" | "srgb";
```

### Para TypeScript abaixo da versão 3.6

Ao usar `typescript<3.6`, o seguinte erro ocorrerá:

```
node_modules/@tensorflow/tfjs-core/dist/engine.d.ts:127:9 - error TS1086: An accessor cannot be declared in an ambient context.

127     get backend(): KernelBackend;
            ~~~~~~~
...
```

Ative [`skipLibCheck`](https://www.typescriptlang.org/tsconfig#skipLibCheck) para suprimir esse erro ou faça upgrade no mínimo para a versão 3.6.2 do TypeScript para corrigi-lo (a correção acima para `PredefinedColorSpace` também precisará ser aplicada).
