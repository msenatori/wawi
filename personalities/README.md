# Wawi personalities

Each file in this folder is a personality: all the numbers that shape how
Wawi lives on your desktop (how much she sleeps, how often she looks at you,
how she greets you, when she plays). They are plain YAML files, meant to be
shared.

The file names are Spanish: *perra tranquila* = calm dog, *perra atenta* =
attentive dog, *cachorra* = puppy.

| File | What she is like |
| --- | --- |
| `perra_tranquila.yaml` | Default. Adult dog: sleeps a lot, watches you out of the corner of her eye, rarely plays. |
| `perra_atenta.yaml` | The same dog, a bit more awake and attentive to you. |
| `cachorra.yaml` | Young dog: short naps, more active, plays up to 3 times a day. |

## Choosing one

In `private/config.yaml`:

```yaml
pet:
  personality: perra_atenta     # a name from this folder, or the path to a file
```

The old `pet.liveliness` still works as a shortcut: `calm` = perra_tranquila,
`normal` = perra_atenta, `playful` = cachorra. If you set both,
`personality` wins.

You can also switch live, without restarting: from the "Personalidad"
submenu in the menu bar (or the right-click menu on Wawi), or by voice —
"Wawi, ponete en modo cachorra" in Spanish mode; in English mode just ask
her. That choice is remembered in `private/pet_local.json` and wins over
the config on every start.

## Making a new one

1. Copy a file (for example `perra_tranquila.yaml`) under another name,
   say `mi_perra.yaml`.
2. Change `name`, `description`, `author` and `version`.
3. Change any values you like. You can delete the ones you don't touch:
   anything missing is taken from `perra_tranquila`.
4. Set `pet.personality: mi_perra`.

## Format

- At the top: `name`, `description`, `author`, `version`.
- Then the sections: `sleep`, `wake`, `rest`, `active`, `habituation`,
  `greeting`, `rhythm`, `play`, `requests`, `context`, `desktop`.
  `perra_tranquila.yaml` has all of them, each with a comment.
- The unit is in the key name: `_min` minutes, `_s` seconds, `_h` hours.
- A pair `[a, b]` is a range: a value between a and b is drawn each time.
- Out-of-range values are clamped to the limit and an unknown key is
  ignored (both are logged); it never breaks anything. The list of keys
  with their limits is in `wawi/pet/personality.py` (`SPEC`).
