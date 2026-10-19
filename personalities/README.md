# Personalidades de Wawi

Cada archivo de esta carpeta es una personalidad: todos los números de cómo
vive Wawi en el escritorio (cuánto duerme, cada cuánto te mira, cómo te
saluda, cuándo juega). Son archivos YAML comunes, para compartir.

| Archivo | Cómo es |
| --- | --- |
| `perra_tranquila.yaml` | Por defecto. Perra adulta: duerme mucho, te mira de reojo, rara vez juega. |
| `perra_atenta.yaml` | La misma, un poco más despierta y pendiente de vos. |
| `cachorra.yaml` | Perra joven: siestas cortas, más activa, juega hasta 3 veces por día. |

## Elegir una

En `config.yaml`:

```yaml
pet:
  personality: perra_atenta     # un nombre de esta carpeta, o la ruta a un archivo
```

El viejo `pet.liveliness` sigue andando como atajo: `calm` = perra_tranquila,
`normal` = perra_atenta, `playful` = cachorra. Si ponés las dos cosas, gana
`personality`. Hay que reiniciar Wawi para que tome el cambio.

## Hacer una nueva

1. Copiá un archivo (por ejemplo `perra_tranquila.yaml`) con otro nombre,
   `mi_perra.yaml`.
2. Cambiá `name`, `description`, `author` y `version`.
3. Cambiá los valores que quieras. Podés borrar los que no toques: lo que
   falte se toma de `perra_tranquila`.
4. Poné `pet.personality: mi_perra`.

## Formato

- Arriba: `name`, `description`, `author`, `version`.
- Después, secciones: `sleep`, `wake`, `rest`, `active`, `habituation`,
  `greeting`, `rhythm`, `play`, `requests`, `context`, `desktop`.
  `perra_tranquila.yaml` las tiene todas, con un comentario cada una.
- La unidad va en el nombre: `_min` minutos, `_s` segundos, `_h` horas.
- Un par `[a, b]` es un rango: cada vez se sortea un valor entre a y b.
- Los valores fuera de rango se recortan al límite y una clave desconocida
  se ignora (las dos cosas quedan en el log); nunca rompe nada. La lista de
  claves con sus límites está en `wawi/pet/personality.py` (`SPEC`).
