# Cambio a TensorFlow.js 4.0

## Qué cambia en TensorFlow.js 4.0

Las notas de lanzamiento se encuentran [disponibles aquí](https://github.com/tensorflow/tfjs/releases). Con este lanzamiento se actualiza TypeScript a 4.8.4 y `@webgpu/types` a 0.1.21. Si no utiliza TypeScript, puede cambiar a 4.0 sin leer este documento.

### Cambios importantes

Este lanzamiento es importante para proyectos en los que se usa `typescript<4.4`. Ninguno de los demás proyectos debería verse afectado.

## Actualización de código desde 3.x

### Para TypeScript &gt;= 4.4

En este lanzamiento no se han hecho cambios importantes en las API, para que los proyectos en los que se usa `typescript>=4.4` esos cambios sean tratados como menores y la actualización se pueda llevar a cabo sin generar modificaciones.

### Para TypeScript &lt; 4.4

Cuando usamos `typescript<4.4`, se produce el siguiente error.

```
node_modules/@webgpu/types/dist/index.d.ts:587:16 - error TS2304: Cannot find name 'PredefinedColorSpace'.

587   colorSpace?: PredefinedColorSpace;
                   ~~~~~~~~~~~~~~~~~~~~
...
```

Para repararlo, cambie TypeScript a 4.4.2 o a una versión posterior, o agregue el archivo `predefined_color_space.d.ts` (el nombre y la ruta se pueden cambiar) con el siguiente contenido a su proyecto para definir el tipo que falta. Cuando actualice TypeScript a 4.4 o a una versión superior, quite el archivo.

**predefined_color_space.d.ts**

```typescript
type PredefinedColorSpace = "display-p3" | "srgb";
```

### Para TypeScript &lt; 3.6

`typescript<3.6` tiene el siguiente error adicional.

```
node_modules/@tensorflow/tfjs-core/dist/engine.d.ts:127:9 - error TS1086: An accessor cannot be declared in an ambient context.

127     get backend(): KernelBackend;
            ~~~~~~~
...
```

Active [`skipLibCheck`](https://www.typescriptlang.org/tsconfig#skipLibCheck) para eliminar este error o actualice la versión, al menos, a TypeScript 3.6.2 para corregirlo (también deberá aplicar la corrección que figura arriba para `PredefinedColorSpace`).
