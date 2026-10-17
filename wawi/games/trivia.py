"""Trivia de perros: canned questions, three options each, no LLM.

Answered by voice (the option's letter, its number, or a word of it) or by
clicking one of the three buttons on her bubble.
"""

from __future__ import annotations

import random
from typing import Any

from wawi.games.core import DRAW, MAX_MOVE_WORDS, OWNER, WAWI, Button, Game, Reply, norm

#: (question, options, index of the right one, a short fun fact after).
QUESTIONS: tuple[tuple[str, tuple[str, str, str], int, str], ...] = (
    ("¿Cuántos dientes tiene un perro adulto?", ("28", "42", "56"), 1, "Cuarenta y dos, como la respuesta a todo."),
    ("¿Por dónde transpiran los perros?", ("Las patas", "La lengua", "Las orejas"), 0, "Por las almohadillas. Y jadean para enfriarse."),
    ("¿Qué raza es la más rápida?", ("Galgo", "Caniche", "Dálmata"), 0, "El galgo llega a unos 70 km/h."),
    ("¿Cuántas veces mejor que vos huele un perro?", ("10", "1.000", "10.000 o más"), 2, "Diez mil veces o más. Te huelo desde acá."),
    ("¿Qué colores ve mejor un perro?", ("Rojo y verde", "Azul y amarillo", "Ninguno"), 1, "Azul y amarillo. El rojo me parece marrón."),
    ("¿Cuál de estas comidas es peligrosa para un perro?", ("Zanahoria", "Chocolate", "Arroz"), 1, "El chocolate. Ni se te ocurra."),
    ("¿Cómo se llamaba la perra que fue al espacio en 1957?", ("Laika", "Lassie", "Pluto"), 0, "Laika, en el Sputnik 2."),
    ("¿Qué hace un perro cuando mueve la cola hacia la derecha?", ("Está contento", "Tiene miedo", "Tiene hambre"), 0, "Más a la derecha, más contento, dicen los estudios."),
    ("¿Cuántas horas por día duerme un perro, más o menos?", ("4", "8", "12 a 14"), 2, "Doce a catorce. Lo mío es profesional."),
    ("¿De qué animal desciende el perro?", ("Zorro", "Lobo", "Chacal"), 1, "Del lobo. Auuu."),
    ("¿Qué raza tiene la lengua azul?", ("Chow chow", "Beagle", "Bulldog"), 0, "El chow chow, y el shar pei también."),
    ("¿Cuál es la temperatura normal de un perro?", ("36 °C", "38,5 °C", "41 °C"), 1, "Unos 38,5. Somos calentitos."),
    ("¿Para qué se inventó el caniche?", ("Para cazar en el agua", "Para la tele", "Para cuidar ovejas"), 0, "Era un perro de agua. El corte de pelo era para nadar."),
    ("¿Cuál es el perro más chico del mundo?", ("Chihuahua", "Pug", "Yorkshire"), 0, "El chihuahua. Chiquito pero con carácter."),
    ("¿Qué significa un bostezo en un perro?", ("Aburrimiento siempre", "A veces nervios", "Hambre"), 1, "A veces es para calmarse. Como vos con el café."),
)

LETTERS = ("a", "b", "c")
_ORDINALS = {"primera": 0, "primero": 0, "segunda": 1, "segundo": 1, "tercera": 2, "tercero": 2,
             "uno": 0, "dos": 1, "tres": 2, "1": 0, "2": 1, "3": 2}


def parse_option(text: str, options: tuple[str, str, str]) -> int | None:
    t = norm(text)
    ws = t.split()
    if not ws or len(ws) > MAX_MOVE_WORDS:
        return None
    for i, opt in enumerate(options):
        o = norm(opt)
        if o and (o == t or f" {o} " in f" {t} "):
            return i
    if len(ws) <= 3:
        for w in ws:
            if w in LETTERS:
                return LETTERS.index(w)
            if w in _ORDINALS:
                return _ORDINALS[w]
    return None


class Trivia(Game):
    name = "trivia"
    title = "Trivia de perros"

    def __init__(self, rng: random.Random | None = None, count: int = 5, **kw: Any) -> None:
        super().__init__(rng, **kw)
        n = max(1, min(int(count), len(QUESTIONS)))
        self.questions = self.rng.sample(list(QUESTIONS), n)
        self.index = 0
        self.right = 0

    def _ask(self, prefix: str = "") -> Reply:
        q, opts, _ok, _fact = self.questions[self.index]
        self.buttons = [Button(f"opt:{i}", f"{LETTERS[i].upper()}) {o}") for i, o in enumerate(opts)]
        self.line = (prefix + " " if prefix else "") + q
        return Reply(self.line, "tilt")

    def start(self) -> Reply:
        return self._ask(f"Trivia de perros, {len(self.questions)} preguntas.")

    def answer(self, i: int) -> Reply:
        if self.over:
            return Reply("Ya terminó la trivia.", "tilt")
        _q, opts, ok, fact = self.questions[self.index]
        good = i == ok
        self.right += int(good)
        said = ("¡Bien! " if good else f"No, era {opts[ok]}. ") + fact
        self.index += 1
        if self.index >= len(self.questions):
            self.over = True
            self.buttons = []
            n = len(self.questions)
            self.line = f"{said} Hiciste {self.right} de {n}."
            if self.right * 2 > n:
                return Reply(self.line, "party", result=OWNER, done=True, fun=20)
            if self.right * 2 == n:
                return Reply(self.line, "shrug", result=DRAW, done=True, fun=15)
            return Reply(self.line + " Te falta calle canina.", "smug", result=WAWI, done=True, fun=15)
        r = self._ask(said)
        r.expr = "cheer" if good else "eyeroll"
        r.fun = 3
        return r

    def hear(self, text: str) -> Reply | None:
        if self.over:
            return None
        i = parse_option(text, self.questions[self.index][1])
        return None if i is None else self.answer(i)

    def click(self, what: str) -> Reply | None:
        if what.startswith("opt:"):
            try:
                i = int(what[4:])
            except ValueError:
                return None
            if 0 <= i < 3:
                return self.answer(i)
        return None
