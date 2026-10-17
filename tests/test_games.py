"""Games against Wawi (wawi/games): the games, the proposals, the tool.

Everything here is pure: fake clocks, synthetic windows and screens, a real
MascotAnimator where her body is involved, no AppKit."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass

import numpy as np
import pytest

from wawi import games, mascot, tools
from wawi.games import propose as P
from wawi.games.core import DRAW, OWNER, WAWI, GameEnv, is_stop, parse_number
from wawi.games.fetch import BALL_R, Ball, Fetch, screen_for, throw_velocity
from wawi.games.guess import GuessHers, GuessYours, direction
from wawi.games.hide import PEEK_SIDE, PEEK_TOP, HideAndSeek, candidates, choose_spot
from wawi.games.manager import GameManager, resolve
from wawi.games.rps import RPS, PatternLearner, beats, outcome, parse_move, simulate
from wawi.games.settings import GamesSettings
from wawi.games.store import GameStore
from wawi.games.tictactoe import (
    EMPTY,
    O,
    X,
    TicTacToe,
    best_moves,
    choose,
    free,
    minimax,
    parse_cell,
    put,
    winner,
)
from wawi.games.trivia import Trivia, parse_option
from wawi.pet.needs import Needs
from wawi.pet.world import Screen, Win

SCREEN = Screen((0.0, 0.0, 1440.0, 900.0), (0.0, 60.0, 1440.0, 815.0))
SIZE = (124.0, 110.0)


def settings(**kw) -> GamesSettings:
    return GamesSettings.from_dict({"state_file": None, **kw})


def manager(tmp_path=None, **kw) -> tuple[GameManager, list]:
    pushed: list = []
    path = (tmp_path / "games.json") if tmp_path is not None else None
    s = GamesSettings.from_dict({"state_file": str(path) if path else None, **kw})
    m = GameManager(s, store=GameStore(path).load(), rng=random.Random(7), push=pushed.append)
    return m, pushed


# ---------------------------------------------------------------------------
# Piedra, papel o tijera
# ---------------------------------------------------------------------------

class TestRPS:
    def test_rules(self):
        assert beats(0) == 1 and beats(1) == 2 and beats(2) == 0
        assert outcome(1, 0) == OWNER and outcome(0, 1) == WAWI and outcome(2, 2) == DRAW

    def test_parse_move(self):
        assert parse_move("piedra") == 0
        assert parse_move("¡Papel!") == 1
        assert parse_move("tijeras") == 2
        assert parse_move("piedra o papel") is None  # two moves: not a move
        assert parse_move("hola") is None
        # Long sentences are conversation, not a throw.
        assert parse_move("anoche dejé el papel de la impresora en la mesa del living") is None

    def test_learner_beats_a_biased_player(self):
        moves = np.random.default_rng(1).choice(3, size=600, p=[0.6, 0.25, 0.15])
        t = simulate(PatternLearner(np.random.default_rng(2)), moves)
        assert t[WAWI] > 1.6 * t[OWNER]
        assert t[WAWI] / len(moves) > 0.42

    def test_learner_beats_a_cycling_player(self):
        t = simulate(PatternLearner(np.random.default_rng(3)), [i % 3 for i in range(300)])
        assert t[WAWI] / 300 > 0.6

    def test_learner_catches_win_stay_lose_shift(self):
        rng = np.random.default_rng(5)
        learner = PatternLearner(np.random.default_rng(6))
        tally = {OWNER: 0, WAWI: 0, DRAW: 0}
        mine = 0
        for _ in range(500):
            w = learner.choose()
            learner.update(mine, w)
            res = outcome(mine, w)
            tally[res] += 1
            # Stay on a win, move up on anything else, a bit of noise.
            mine = mine if res == OWNER else (mine + 1) % 3
            if rng.random() < 0.1:
                mine = int(rng.integers(3))
        assert tally[WAWI] > tally[OWNER]

    def test_she_stays_beatable_against_random(self):
        moves = np.random.default_rng(4).integers(3, size=3000)
        t = simulate(PatternLearner(np.random.default_rng(9)), moves)
        assert 0.27 < t[OWNER] / 3000 < 0.40
        assert t[WAWI] / 3000 < 0.40

    def test_she_mixes_in_randomness(self):
        # Even against a single repeated move, she does not always counter it.
        learner = PatternLearner(np.random.default_rng(0))
        picks = []
        for _ in range(300):
            w = learner.choose()
            picks.append(w)
            learner.update(0, w)
        late = picks[50:]
        assert 0.6 < late.count(1) / len(late) < 0.98

    def test_history_round_trip(self):
        a = PatternLearner(np.random.default_rng(1))
        for m in [0, 0, 1, 0, 0, 1, 0, 0, 1]:
            a.update(m, 2)
        b = PatternLearner(np.random.default_rng(1))
        b.load(a.to_list())
        assert np.allclose(a.after2, b.after2) and b.history == a.history

    def test_match_first_to_three(self):
        g = RPS(random.Random(1), learner=PatternLearner(np.random.default_rng(1)), first_to=3)
        g.start()
        reply = None
        for _ in range(50):
            reply = g.hear("piedra")
            if reply.done:
                break
        assert reply is not None and reply.done and g.over
        assert max(g.owner, g.wawi) == 3
        assert reply.result in (OWNER, WAWI)
        # Her hand first, then the trophy (or the sulk).
        assert reply.expr in ("rps_rock", "rps_paper", "rps_scissors")
        assert reply.then == ("smug_win" if reply.result == WAWI else "sad")

    def test_buttons_play(self):
        g = RPS(random.Random(1), learner=PatternLearner(np.random.default_rng(1)))
        assert [b.id for b in g.buttons] == ["rps:0", "rps:1", "rps:2"]
        r = g.click("rps:2")
        assert r is not None and r.text
        assert g.click("rps:9") is None


# ---------------------------------------------------------------------------
# Ta-te-tí
# ---------------------------------------------------------------------------

class TestTicTacToe:
    def test_winner(self):
        assert winner("XXX" + EMPTY * 6) == X
        assert winner("O  O  O  ") == O
        assert winner("XOXXOOOXX") == "="
        assert winner(EMPTY * 9) is None

    def test_perfect_play_is_a_draw(self):
        assert minimax(EMPTY * 9, X) == 0

    def test_takes_a_win(self):
        board = "OO XX    "
        assert best_moves(board, O) == [2]

    def test_blocks_a_win(self):
        board = "XX  O    "
        assert best_moves(board, O) == [2]

    def test_dificil_never_loses(self):
        rng = random.Random(3)
        for n in range(150):
            board, turn = EMPTY * 9, X if n % 2 else O
            while winner(board) is None:
                if turn == X:
                    board = put(board, rng.choice(free(board)), X)
                else:
                    board = put(board, choose(board, "dificil", rng), O)
                turn = O if turn == X else X
            assert winner(board) != X

    def _vs_perfect(self, level: str, games_n: int = 150) -> int:
        rng = random.Random(11)
        losses = 0
        for n in range(games_n):
            board, turn = EMPTY * 9, X if n % 2 else O
            while winner(board) is None:
                if turn == X:
                    board = put(board, rng.choice(best_moves(board, X)), X)
                else:
                    board = put(board, choose(board, level, rng), O)
                turn = O if turn == X else X
            losses += winner(board) == X
        return losses

    def test_facil_makes_mistakes_you_can_beat(self):
        assert self._vs_perfect("facil") > 30
        assert self._vs_perfect("dificil") == 0
        assert self._vs_perfect("facil") > self._vs_perfect("normal")

    def test_parse_cell(self):
        assert parse_cell("jugá en el centro") == 4
        assert parse_cell("arriba a la izquierda") == 0
        assert parse_cell("abajo a la derecha") == 8
        assert parse_cell("arriba") == 1
        assert parse_cell("la siete") == 6
        assert parse_cell("qué lindo día") is None

    def test_game_flow_with_thinking_delay(self):
        g = TicTacToe(random.Random(1), level="dificil", owner_starts=True)
        g.start()
        r = g.hear("jugá en el centro")
        assert r is not None and g.board[4] == X and g.turn == O
        assert g.tick(0.5) is None  # still thinking
        g.tick(0.6)
        assert g.board.count(O) == 1 and g.turn == X
        assert g.click("cell:4").expr == "eyeroll"  # taken

    def test_she_starts_every_other_game(self):
        g = TicTacToe(random.Random(1), owner_starts=False)
        g.start()
        g.tick(1.0)
        assert g.board.count(O) == 1


# ---------------------------------------------------------------------------
# Adiviná el número
# ---------------------------------------------------------------------------

class TestGuess:
    @pytest.mark.parametrize("text,n", [
        ("cuarenta y dos", 42), ("es el 7", 7), ("cien", 100), ("veintitrés", 23),
        ("setenta", 70), ("quince", 15), ("noventa y nueve", 99), ("hola", None),
    ])
    def test_parse_number(self, text, n):
        assert parse_number(text) == n

    def test_higher_lower_and_win(self):
        g = GuessHers(random.Random(1), secret=42)
        g.start()
        assert g.hear("cincuenta").text.startswith("Más bajo")
        assert g.hear("treinta").text.startswith("Más alto")
        r = g.hear("cuarenta y dos")
        assert r.done and r.result == OWNER and "42" in r.text

    def test_warm_when_close_and_eyeroll_when_ignoring_hints(self):
        g = GuessHers(random.Random(1), secret=42)
        assert g.hear("cuarenta").expr == "wag"
        r = g.hear("diez")  # she said más alto at 40
        assert r.expr == "eyeroll" and "te lo había dicho" in r.text

    def test_out_of_tries_she_wins(self):
        g = GuessHers(random.Random(1), secret=99)
        r = None
        for _ in range(10):
            r = g.guess(1)
        assert r.done and r.result == WAWI

    def test_long_sentences_are_not_guesses(self):
        g = GuessHers(random.Random(1), secret=5)
        assert g.hear("ayer fui a comprar cinco kilos de papas al mercado del barrio") is None

    def test_direction(self):
        assert direction("más alto") == "up"
        assert direction("mayor") == "up"
        assert direction("más bajo") == "down"
        assert direction("menos") == "down"
        assert direction("¡sí, ese!") == "yes"
        assert direction("más") is None

    @pytest.mark.parametrize("secret", [1, 2, 37, 50, 64, 99, 100])
    def test_she_finds_yours_by_binary_search(self, secret):
        g = GuessYours(random.Random(1))
        g.start()
        for _ in range(8):
            if g.current == secret:
                r = g.hear("sí")
                break
            r = g.hear("más alto" if secret > g.current else "más bajo")
        assert r.done and r.result == WAWI and g.tries <= 7

    def test_she_notices_cheating(self):
        # "More than 50", then "less" every time: the answers cannot all be
        # true, and she says so.
        g = GuessYours(random.Random(1))
        g.start()
        g.answer("up")
        last = None
        while not g.over:
            last = g.answer("down")
        assert last.expr == "eyeroll" and last.result == OWNER


# ---------------------------------------------------------------------------
# Trivia
# ---------------------------------------------------------------------------

class TestTrivia:
    def test_parse_option(self):
        opts = ("Galgo", "Caniche", "Dálmata")
        assert parse_option("galgo", opts) == 0
        assert parse_option("la b", opts) == 1
        assert parse_option("la tercera", opts) == 2
        assert parse_option("dalmata", opts) == 2
        assert parse_option("ni idea", opts) is None

    def test_a_whole_game(self):
        g = Trivia(random.Random(2), count=3)
        g.start()
        r = None
        for _ in range(3):
            ok = g.questions[g.index][2]
            r = g.click(f"opt:{ok}")
        assert r.done and r.result == OWNER and g.right == 3


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def fetch_env(pos=(700.0, 60.0), energy=80.0) -> GameEnv:
    return GameEnv(now=0.0, position=pos, size=SIZE, screens=[SCREEN], energy=energy)


class TestFetchPhysics:
    @pytest.mark.parametrize("seed", range(12))
    def test_any_throw_stays_on_screen_and_comes_to_rest(self, seed):
        rng = random.Random(seed)
        area = SCREEN.visible
        ball = Ball(rng.uniform(100, 1300), rng.uniform(100, 800))
        ball.throw(rng.uniform(-5000, 5000), rng.uniform(-5000, 5000))
        vx, vy, vw, vh = area
        for _ in range(60 * 40):
            ball.step(1 / 60, area)
            assert vx + BALL_R - 1e-6 <= ball.x <= vx + vw - BALL_R + 1e-6
            assert vy + BALL_R - 1e-6 <= ball.y <= vy + vh - BALL_R + 1e-6
            if ball.resting:
                break
        assert ball.resting and ball.y == pytest.approx(vy + BALL_R)

    def test_throws_are_capped(self):
        ball = Ball(500, 500)
        ball.throw(1e6, 0)
        assert math.hypot(ball.vx, ball.vy) <= 2600.0 + 1e-6

    def test_throw_velocity_from_the_last_samples(self):
        samples = [(0.0, 0, 0), (0.05, 50, 10), (0.10, 100, 20)]
        vx, vy = throw_velocity(samples)
        assert vx == pytest.approx(1000.0) and vy == pytest.approx(200.0)
        assert throw_velocity([(0.0, 1, 1)]) == (0.0, 0.0)

    def test_screen_for_picks_the_nearest(self):
        right = Screen((1440.0, 0.0, 1000.0, 800.0), (1440.0, 0.0, 1000.0, 780.0))
        assert screen_for([SCREEN, right], 2000, 300) == right.visible
        assert screen_for([SCREEN, right], -50, 300) == SCREEN.visible


def run_fetch(g: Fetch, env: GameEnv, seconds: float = 12.0):
    replies = []
    for _ in range(int(seconds * 60)):
        r = g.tick(1 / 60, env)
        if r is not None:
            replies.append(r)
        if g.phase in ("ready", "tired") and replies:
            break
    return replies


class TestFetchGame:
    def test_fetch_and_bring_back(self):
        g = Fetch(random.Random(1), max_fetches=3)
        env = fetch_env()
        g.start()
        g.tick(1 / 60, env)
        assert g.phase == "ready" and g.dog[0] == pytest.approx(700.0)
        assert g.grab(600, 300, 0.0)
        g.drag(800, 400, 0.05)
        g.release(900, 450, 0.10)
        assert g.phase == "flying"
        replies = run_fetch(g, env)
        assert g.fetches == 1 and replies[-1].energy < 0 and replies[-1].fun > 0
        # Back where she started, the ball at her feet.
        assert g.dog[0] == pytest.approx(700.0)
        assert abs(g.ball.x - g.dog[0]) < SIZE[0]
        st = g.state()
        assert st["ground"][1] == pytest.approx(SCREEN.visible[1])

    def test_she_runs_and_carries_it_in_her_mouth(self):
        g = Fetch(random.Random(1))
        env = fetch_env()
        g.tick(1 / 60, env)
        g.grab(700, 300, 0.0)
        g.release(700, 300, 0.1)
        seen_trot = seen_carry = False
        for _ in range(60 * 12):
            g.tick(1 / 60, env)
            st = g.state()
            seen_trot |= st["loco"] == "trot"
            if g.phase == "return":
                seen_carry = True
                assert st["ball"] == pytest.approx(g.mouth())
                assert st["grabbable"] is False
            if g.phase == "ready" and g.fetches:
                break
        assert seen_carry and g.fetches == 1
        # Thrown straight down next to her: maybe no running needed, but
        # never anywhere off screen.
        assert seen_trot or g.dog[0] == pytest.approx(700.0)

    def test_she_gets_tired(self):
        g = Fetch(random.Random(1), max_fetches=2)
        env = fetch_env()
        g.tick(1 / 60, env)
        last = None
        for i in range(2):
            g.grab(300 + i * 500, 400, i * 10.0)
            g.release(300 + i * 500, 400, i * 10.0 + 0.1)
            replies = run_fetch(g, env)
            last = replies[-1]
        assert last.done and g.phase == "tired" and g.over

    def test_low_energy_ends_it_sooner(self):
        g = Fetch(random.Random(1), max_fetches=10)
        env = fetch_env(energy=20.0)
        g.tick(1 / 60, env)
        g.grab(300, 400, 0.0)
        g.release(300, 400, 0.1)
        assert run_fetch(g, env)[-1].done

    def test_she_drops_from_a_window_to_the_floor_first(self):
        g = Fetch(random.Random(1))
        env = fetch_env(pos=(700.0, 500.0))
        g.tick(1 / 60, env)
        g.grab(300, 400, 0.0)
        g.release(300, 400, 0.1)
        g.tick(1 / 60, env)
        assert g.state()["loco"] in ("fall", "trot")
        run_fetch(g, env)
        assert g.dog[1] == pytest.approx(SCREEN.visible[1])

    def test_waits_for_her_body(self):
        g = Fetch(random.Random(1))
        g.tick(1 / 60, GameEnv(position=None, screens=[SCREEN]))
        assert g.phase == "wait" and not g.grab(1, 1, 0)


# ---------------------------------------------------------------------------
# Escondidas
# ---------------------------------------------------------------------------

def win(wid, x, y, w, h, owner="Safari"):
    return Win(wid, (float(x), float(y), float(w), float(h)), owner=owner)


def _overlap(a, b):
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


class TestHideSpots:
    def test_nowhere_to_hide_without_windows(self):
        assert choose_spot([], [SCREEN], SIZE, random.Random(1)) is None

    def test_tiny_windows_are_no_hiding_place(self):
        assert choose_spot([win(1, 100, 200, 150, 120)], [SCREEN], SIZE, random.Random(1)) is None

    def test_spots_peek_only_a_little_and_are_otherwise_covered(self):
        windows = [win(1, 600, 300, 500, 400), win(2, 100, 150, 700, 500, "Terminal")]
        spots = candidates(windows, [SCREEN], SIZE)
        assert spots
        w, h = SIZE
        for s in spots:
            target = next(x for x in windows if x.wid == s.wid)
            gx, gy = s.ground
            her = (gx - w / 2, gy, w, h)
            # All of her on the screen.
            vx, vy, vw, vh = SCREEN.visible
            assert vx <= her[0] and her[0] + w <= vx + vw and vy <= gy and gy + h <= vy + vh
            # The peeking part is a sliver of her, outside the window…
            px, py, pw, ph = s.peek
            assert pw * ph == pytest.approx(w * h * (PEEK_TOP if s.kind == "top" else PEEK_SIDE))
            assert not _overlap(s.peek, target.rect) or s.kind == "top" and py >= target.top - 0.5
            # …and not covered by any window in front of hers.
            idx = windows.index(target)
            assert not any(_overlap(s.peek, f.rect) for f in windows[:idx])

    def test_a_window_fully_behind_another_offers_nothing(self):
        front = win(1, 0, 60, 1440, 815)
        back = win(2, 300, 200, 500, 400)
        assert [s for s in candidates([front, back], [SCREEN], SIZE) if s.wid == 2] == []

    def test_she_prefers_spots_far_from_the_cursor(self):
        windows = [win(1, 60, 200, 400, 400), win(2, 980, 200, 400, 400)]
        picks = [choose_spot(windows, [SCREEN], SIZE, random.Random(i), cursor=(150, 500)).wid
                 for i in range(20)]
        assert picks.count(2) >= 15

    def test_side_spots_show_the_tail(self):
        spots = candidates([win(1, 400, 200, 600, 500)], [SCREEN], SIZE)
        kinds = {s.kind: s for s in spots}
        assert {"top", "right", "left"} <= set(kinds)
        assert kinds["right"].facing_right is False and kinds["left"].facing_right is True


def hide_env(windows, cursor=(100.0, 100.0)) -> GameEnv:
    return GameEnv(now=0.0, position=(700.0, 60.0), size=SIZE, screens=[SCREEN],
                   windows=windows, cursor=cursor)


class TestHideGame:
    def _hidden(self, windows, seconds=60.0):
        g = HideAndSeek(random.Random(1), seconds=seconds)
        g.start()
        env = hide_env(windows)
        for _ in range(60 * 6):
            g.tick(1 / 60, env)
        assert g.phase == "hidden"
        return g, env

    def test_count_travel_hide_and_be_found(self):
        g = HideAndSeek(random.Random(1))
        g.start()
        env = hide_env([win(7, 400, 200, 600, 500)])
        g.tick(1.0, env)
        assert g.phase == "count" and g.state()["ground"] is None
        g.tick(3.0, env)
        assert g.phase == "travel" and g.state()["invisible"] is True
        g.tick(2.0, env)
        st = g.state()
        assert g.phase == "hidden" and st["below"] == 7 and st["peek"] in ("top", "right", "left")
        assert st["ground"] is not None and st["invisible"] is False
        assert g.touched() .result == OWNER
        assert g.over

    def test_clicks_before_she_hides_are_not_finding_her(self):
        g = HideAndSeek(random.Random(1))
        g.start()
        assert g.touched() is None

    def test_she_follows_a_moving_window(self):
        g, env = self._hidden([win(7, 400, 200, 600, 500)])
        before = g.state()["ground"]
        env.windows = [win(7, 450, 250, 600, 500)]
        g.tick(1 / 60, env)
        after = g.state()["ground"]
        assert after[0] - before[0] == pytest.approx(50) and after[1] - before[1] == pytest.approx(50)

    def test_closing_her_window_is_a_draw(self):
        g, env = self._hidden([win(7, 400, 200, 600, 500)])
        env.windows = []
        r = g.tick(1 / 60, env)
        assert r.done and r.result == DRAW

    def test_a_hint_then_she_wins_on_time(self):
        g, env = self._hidden([win(7, 400, 200, 600, 500)], seconds=20.0)
        sounds, result = [], None
        for _ in range(60 * 25):
            r = g.tick(1 / 60, env)
            if r is not None and r.sound:
                sounds.append(r.sound)
            if r is not None and r.done:
                result = r
                break
        assert sounds == ["bark"] and result.result == WAWI and "Safari" in result.text

    def test_nowhere_to_hide_ends_the_game(self):
        g = HideAndSeek(random.Random(1))
        r = None
        for _ in range(60 * 5):
            r = g.tick(1 / 60, hide_env([])) or r
        assert r is not None and r.done


# ---------------------------------------------------------------------------
# Proposals: the gate, and what she learns
# ---------------------------------------------------------------------------

def penv(**kw) -> GameEnv:
    base = dict(now=10_000.0, wall=1_800_000_000.0, key_idle=120.0, fun=20.0, quiet=False)
    base.update(kw)
    return GameEnv(**base)


class TestProposalGate:
    def proposer(self, **kw):
        return P.Proposer(settings(**kw), rng=random.Random(1))

    def test_bored_and_present_she_may(self):
        assert self.proposer().blocked(penv()) is None

    def test_quiet_mode_blocks(self):
        assert self.proposer().blocked(penv(quiet=True)) == "quiet"

    def test_unknown_quiet_does_not_block(self):
        assert self.proposer().blocked(penv(quiet=None)) is None

    def test_focus_blocks(self):
        assert self.proposer().blocked(penv(key_idle=3.0)) == "focus"

    def test_empty_room_blocks(self):
        assert self.proposer().blocked(penv(key_idle=3600.0)) == "away"

    def test_content_dog_does_not_ask(self):
        assert self.proposer().blocked(penv(fun=90.0)) == "content"

    def test_busy_blocks(self):
        assert self.proposer().blocked(penv(busy=True)) == "busy"
        assert self.proposer().blocked(penv(), playing=True) == "busy"

    def test_off(self):
        assert self.proposer(propose=False).blocked(penv()) == "off"

    def test_cooldown_and_daily_budget(self):
        pr = self.proposer(propose_every_min=30, max_proposals_per_day=2)
        pr.propose(penv(), ["ppt"])
        pr.answer(False)
        assert pr.blocked(penv(now=10_000.0 + 60)) == "cooldown"
        assert pr.blocked(penv(now=10_000.0 + 31 * 60)) is None
        pr.propose(penv(now=10_000.0 + 31 * 60), ["ppt"])
        pr.answer(False)
        assert pr.blocked(penv(now=10_000.0 + 62 * 60)) == "budget"

    def test_after_a_break_even_when_not_bored(self):
        pr = self.proposer(after_break_min=10)
        pr.observe(penv(now=0.0, key_idle=900.0, fun=90.0))
        pr.observe(penv(now=1.0, key_idle=0.5, fun=90.0))
        assert pr.returned_at == 1.0
        assert pr.blocked(penv(now=60.0, key_idle=60.0, fun=90.0)) is None
        # The window after coming back closes.
        assert pr.blocked(penv(now=1000.0, key_idle=60.0, fun=90.0)) == "content"

    def test_short_pauses_are_not_breaks(self):
        pr = self.proposer(after_break_min=10)
        pr.observe(penv(now=0.0, key_idle=120.0))
        pr.observe(penv(now=1.0, key_idle=0.5))
        assert pr.returned_at is None


class TestProposalLearning:
    def test_accepted_games_get_picked_more(self):
        pr = P.Proposer(settings(), rng=random.Random(1))
        pr.bandit.rng = np.random.default_rng(1)
        env = penv()
        ctx = pr.ctx(env)
        for _ in range(12):
            pr.pending = P.Proposal("trivia", 0, 1, ctx)
            pr.answer(True)
            pr.pending = P.Proposal("ppt", 0, 1, ctx)
            pr.answer(False)
        picks = [pr.pick(["ppt", "trivia", "tateti"], ctx) for _ in range(200)]
        assert picks.count("trivia") > 150
        assert picks.count("ppt") < 5

    def test_ignored_counts_less_than_a_no(self):
        pr = P.Proposer(settings(), rng=random.Random(1))
        pr.pending = P.Proposal("ppt", 0, 1, "tarde:bored")
        pr.answer(None)
        pr.pending = P.Proposal("tateti", 0, 1, "tarde:bored")
        pr.answer(False)
        assert pr.bandit.expected("ppt", "tarde:bored") > pr.bandit.expected("tateti", "tarde:bored")

    def test_a_refused_moment_makes_her_ask_less(self):
        def proposals(refusals: int) -> int:
            pr = P.Proposer(settings(propose_every_min=1), rng=random.Random(3))
            env = penv()
            ctx = pr.ctx(env)
            for _ in range(refusals):
                pr.bandit.update(P.NOW_ARM, ctx, P.REWARD_NO)
            n = 0
            for i in range(200):
                e = penv(now=10_000.0 + i * 3600.0)
                pr.day, pr.today = "", 0
                p = pr.maybe(e, ["ppt"])
                if p is not None:
                    n += 1
                    pr.answer(None)
                    pr.bandit.cells.clear(), pr.bandit.arms.clear()
                    for _ in range(refusals):
                        pr.bandit.update(P.NOW_ARM, ctx, P.REWARD_NO)
            return n

        assert proposals(30) < proposals(0) * 0.6

    def test_timeout_is_ignore(self):
        pr = P.Proposer(settings(proposal_timeout_s=10), rng=random.Random(1))
        pr.propose(penv(), ["ppt"])
        assert not pr.expired(10_005.0)
        assert pr.expired(10_011.0)

    def test_state_round_trip_keeps_the_cooldown(self):
        pr = P.Proposer(settings(), rng=random.Random(1))
        pr.propose(penv(now=500.0, wall=1_800_000_000.0), ["ppt"])
        pr.answer(True)
        raw = json.loads(json.dumps(pr.to_dict(1_800_000_100.0, 600.0)))
        again = P.Proposer(settings(), rng=random.Random(1))
        again.load(raw, 1_800_000_200.0, 50.0)
        # 200 s after the proposal in wall time, whatever the new monotonic clock.
        assert 50.0 - again.last_at == pytest.approx(200.0)
        assert again.bandit.expected("ppt", pr.pending.ctx if pr.pending else
                                     list(pr.bandit.cells)[0].split("|")[1]) > 0.5


# ---------------------------------------------------------------------------
# The manager: voice, clicks, proposals, needs, scores
# ---------------------------------------------------------------------------

class TestManager:
    def test_resolve(self):
        assert resolve("ppt") == "ppt"
        assert resolve("rps") == "ppt"
        assert resolve("piedra, papel o tijera") == "ppt"
        assert resolve("Ta-te-tí") == "tateti"
        assert resolve("adiviná mi número") == "adivino"
        assert resolve("escondidas") == "escondidas"
        assert resolve("ajedrez") is None

    def test_voice_starts_plays_and_stops(self):
        m, pushed = manager()
        assert "Piedra" in m.hear("juguemos a piedra papel o tijera")
        assert m.playing == "ppt"
        said = m.hear("papel")
        assert said and "Vos" in said
        assert any(r.behaviour for r in pushed)
        assert m.hear("¿qué hora es?") is None  # not a move: the brain answers
        assert m.hear("basta") == "Bueno, lo dejamos acá."
        assert m.playing is None
        assert m.hear("papel") is None  # no game: just a word

    def test_nothing_is_consumed_without_a_game(self):
        m, _ = manager()
        for text in ("piedra", "más alto", "sí", "dale", "en el centro", "cuarenta y dos"):
            assert m.hear(text) is None

    def test_otra_replays_the_last_game(self):
        m, _ = manager()
        m.start("numero")
        m.game.secret = 10
        m.hear("diez")
        assert m.playing is None
        assert m.hear("otra") is not None and m.playing == "numero"

    def test_body_games_need_her_body(self):
        m, _ = manager()
        assert "escritorio" in m.start("pelota") and m.playing is None
        m.body_ok = True
        m.start("pelota")
        assert m.playing == "pelota" and m.wants_free()

    def test_proposal_accepted_by_voice(self, monkeypatch):
        signals = []
        monkeypatch.setattr("wawi.learn.signal", lambda kind, now=None: signals.append(kind))
        m, pushed = manager()
        text = m.propose_now("ppt")
        assert text == "¿Jugamos a piedra, papel o tijera?"
        assert any(r.behaviour == "ball_bring" for r in pushed)
        v = m.tick(0.03, penv(now=m._clock()))
        assert v.bubble == text and v.bubble_action == "accept"
        assert [b.id for b in v.buttons] == ["accept", "decline"]
        said = m.hear("¡dale!")
        assert said and m.playing == "ppt"
        assert signals == ["play"]
        ctx = list(m.proposer.bandit.cells)[0].split("|")[1]
        assert m.proposer.bandit.expected("ppt", ctx) > 0.5

    def test_proposal_declined_by_voice(self):
        m, pushed = manager()
        m.propose_now("trivia")
        assert m.hear("no, ahora no") == "Bueno, otro día."
        assert m.playing is None and m.proposer.pending is None
        assert pushed[-1].behaviour == "shrug"
        ctx = list(m.proposer.bandit.cells)[0].split("|")[1]
        assert m.proposer.bandit.expected("trivia", ctx) < 0.5

    def test_proposal_accepted_by_clicking_the_bubble(self):
        m, _ = manager()
        m.propose_now("tateti")
        assert m.bubble_clicked() is True
        assert m.playing == "tateti"

    def test_ignored_proposal_times_out(self):
        m, _ = manager(proposal_timeout_s=10)
        m.propose_now("ppt")
        m.tick(0.03, penv(now=m._clock() + 11.0, fun=90.0))
        assert m.proposer.pending is None and m.playing is None
        ctx = list(m.proposer.bandit.cells)[0].split("|")[1]
        assert m.proposer.bandit.expected("ppt", ctx) < 0.5

    def test_tick_proposes_when_bored(self):
        m, pushed = manager()
        v = m.tick(0.03, penv(now=m._clock(), fun=10.0))
        assert m.proposer.pending is not None and v.bubble

    def test_tick_never_proposes_in_quiet_mode(self):
        m, _ = manager()
        for i in range(10):
            m.tick(30.0, penv(now=m._clock() + i * 30.0, fun=5.0, quiet=True))
        assert m.proposer.pending is None

    def test_playing_raises_her_fun(self):
        m, _ = manager()
        needs = Needs({"fun": 20.0}, now=0.0)
        m.start("ppt")
        for _ in range(20):
            if m.hear("piedra") is None or m.playing is None:
                break
        m.tick(0.03, penv(now=m._clock(), fun=90.0), needs)
        assert needs.snapshot().fun > 30.0

    def test_fetch_tires_her(self):
        m, _ = manager()
        m.body_ok = True
        needs = Needs({"energy": 80.0}, now=0.0)
        m.start("pelota")
        m._gains.append((6.0, -4.0))
        m.tick(0.03, penv(now=m._clock(), fun=90.0), needs)
        assert needs.snapshot().energy < 80.0

    def test_scores_persist(self, tmp_path):
        m, _ = manager(tmp_path)
        m.start("numero")
        m.game.secret = 33
        m.hear("treinta y tres")
        data = json.loads((tmp_path / "games.json").read_text())
        assert data["scores"]["numero"]["owner"] == 1
        m2, _ = manager(tmp_path)
        assert m2.score_line().startswith("Vos 1")

    def test_rps_habits_persist(self, tmp_path):
        m, _ = manager(tmp_path, rps_first_to=2)
        m.start("ppt")
        while m.playing:
            m.hear("piedra")
        m2, _ = manager(tmp_path)
        assert m2.learner.history and m2.learner.freq[0] > 0

    def test_hide_and_seek_found_by_a_click(self):
        m, _ = manager()
        m.body_ok = True
        m.start("escondidas")
        env = penv(now=0.0, position=(700.0, 60.0), size=SIZE, screens=[SCREEN],
                   windows=[win(3, 400, 200, 600, 500)], fun=90.0)
        for i in range(60 * 6):
            env.now = i / 60
            v = m.tick(1 / 60, env)
        assert v.free and v.state["below"] == 3
        assert v.bubble is None  # no bubble giving her away
        assert m.touched() is True and m.playing is None
        assert m.store.score("escondidas")["owner"] == 1

    def test_ttt_board_view(self):
        m, _ = manager()
        m.start("tateti")
        m.click("cell:4")
        v = m.tick(1.0, penv(now=m._clock(), fun=90.0))
        assert v.game == "tateti" and v.state["board"][4] == "X"
        assert v.state["board"].count("O") == 1


# ---------------------------------------------------------------------------
# Her body during a game (the pure half of wawi/games/ui.py)
# ---------------------------------------------------------------------------

@dataclass
class FakeOut:
    ground: tuple | None = None
    hidden: bool = False
    bubble: str | None = None


class TestBody:
    def anim(self):
        return mascot.MascotAnimator(available=mascot.FRAME_NAMES, rng=random.Random(1))

    def test_fetch_moves_and_runs_her(self):
        from wawi.games.manager import View
        from wawi.games.ui import apply_to_body

        anim, out = self.anim(), FakeOut()
        view = View("pelota", None, [], None, {"ground": (300.0, 60.0), "loco": "trot",
                                              "facing_right": True, "clip": None}, free=True)
        apply_to_body(view, out, anim, None)
        assert out.ground == (300.0, 60.0)
        assert anim.locomotion == "trot" and anim.mirrored

    def test_hide_peeks_with_her_ears_or_tail(self):
        from wawi.games.manager import View
        from wawi.games.ui import PEEK_EARS, PEEK_TAIL, apply_to_body

        anim, out = self.anim(), FakeOut()
        top = View("escondidas", None, [], None, {"ground": (1.0, 2.0), "peek": "top",
                                                 "facing_right": False}, free=True)
        assert apply_to_body(top, out, anim, None) == PEEK_EARS.name
        assert anim.act_name == PEEK_EARS.name
        side = View("escondidas", None, [], None, {"ground": (1.0, 2.0), "peek": "left",
                                                  "facing_right": True}, free=True)
        assert apply_to_body(side, out, anim, PEEK_EARS.name) == PEEK_TAIL.name
        assert anim.act_name == PEEK_TAIL.name and anim.mirrored

    def test_travelling_to_her_spot_she_is_invisible(self):
        from wawi.games.manager import View
        from wawi.games.ui import apply_to_body

        out = FakeOut()
        apply_to_body(View("escondidas", None, [], None, {"invisible": True}, free=True),
                      out, self.anim(), None)
        assert out.hidden is True

    def test_not_free_leaves_her_alone(self):
        from wawi.games.manager import View
        from wawi.games.ui import apply_to_body

        out = FakeOut()
        apply_to_body(View("ppt", "x", [], None, {"ground": (1, 1)}, free=False), out,
                      self.anim(), None)
        assert out.ground is None


# ---------------------------------------------------------------------------
# The games' own frames, and their fallbacks
# ---------------------------------------------------------------------------

ALL_FRAMES = frozenset(mascot.FRAME_NAMES)
GAME_FRAMES = frozenset(n for n in mascot.FRAME_NAMES if n.startswith(
    ("run_ball_", "peek_", "giggle_", "rps_", "think_paw_chin", "smug_win", "cover_eyes")))
OLD_FRAMES = ALL_FRAMES - GAME_FRAMES


def _frame_alpha(name):
    """The frame's alpha mask (rows top-down), or skip without art/PIL."""
    pil = pytest.importorskip("PIL.Image")
    path = mascot.frame_path(name)
    if not path.is_file():
        pytest.skip(f"no {name}.png")
    return np.array(pil.open(path).convert("RGBA"))[..., 3] > 20


class TestGameArt:
    def test_art_or_fallback(self):
        from wawi.games.core import art

        assert art("giggle", ALL_FRAMES) == "giggle"
        assert art("giggle", OLD_FRAMES) == "tickle"
        assert art("smug_win", OLD_FRAMES) == "party"
        assert art("rps_rock", OLD_FRAMES) is None  # no hand, no stranger
        assert art("think_paw_chin", OLD_FRAMES) is None
        assert art("cover_eyes", OLD_FRAMES) == "wag"
        assert art("sad", OLD_FRAMES) == "sad" and art(None, ALL_FRAMES) is None

    def test_rps_throw_shows_her_hand_then_her_face(self):
        m, pushed = manager()
        m.tick(0.0, penv(frames=ALL_FRAMES))
        m.start("ppt")
        pushed.clear()
        m.hear("piedra")
        names = [r.behaviour for r in pushed]
        assert len(names) == 2 and names[0] in ("rps_rock", "rps_paper", "rps_scissors")
        assert names[1] in ("smug", "eyeroll", "sad", "shrug")
        assert all(r.priority == 2 for r in pushed)  # the face waits for the hand

    def test_rps_without_hand_frames_just_reacts(self):
        m, pushed = manager()
        m.tick(0.0, penv(frames=OLD_FRAMES))
        m.start("ppt")
        pushed.clear()
        m.hear("papel")
        assert [r.behaviour for r in pushed] in (["smug"], ["eyeroll"], ["sad"], ["shrug"])

    def test_rps_match_won_with_the_trophy(self):
        for frames, want in ((ALL_FRAMES, "smug_win"), (OLD_FRAMES, "party")):
            m, pushed = manager()
            m.tick(0.0, penv(frames=frames))
            m.start("ppt")
            m.learner.choose = lambda: 1  # papel, always
            while m.playing:
                m.hear("piedra")
            assert pushed[-1].behaviour == want

    def test_tateti_thinks_with_her_paw_and_wins_smug(self):
        g = TicTacToe(random.Random(1), level="dificil", owner_starts=True)
        g.start()
        assert g.hear("centro").expr == "think_paw_chin"
        assert TicTacToe(random.Random(1), owner_starts=False).start().expr == "think_paw_chin"
        g = TicTacToe(random.Random(1), level="dificil")
        g.board, g.turn = "OO XX    ", O
        assert g.wawi_move().expr == "smug_win"

    def test_tateti_thinking_without_the_frame_shows_nothing(self):
        m, pushed = manager()
        m.tick(0.0, penv(frames=OLD_FRAMES))
        m.start("tateti")
        pushed.clear()
        m.click("cell:4")
        assert pushed == []

    def test_found_she_giggles(self):
        g = TestHideGame()._hidden([win(1, 400, 200, 600, 500)])[0]
        assert g.touched().expr == "giggle"
        for frames, want in ((ALL_FRAMES, "giggle"), (OLD_FRAMES, "tickle")):
            m, pushed = manager()
            m.body_ok = True
            m.start("escondidas")
            env = penv(now=0.0, position=(700.0, 60.0), size=SIZE, screens=[SCREEN],
                       windows=[win(3, 400, 200, 600, 500)], fun=90.0, frames=frames)
            for i in range(60 * 6):
                env.now = i / 60
                m.tick(1 / 60, env)
            assert m.touched() is True and pushed[-1].behaviour == want

    def test_counting_she_covers_her_eyes(self):
        g = HideAndSeek(random.Random(1))
        assert g.start().expr == "cover_eyes"
        assert g.state()["hold"] == "cover_eyes"
        env = hide_env([win(7, 400, 200, 600, 500)])
        g.tick(4.0, env)
        assert g.phase == "travel" and g.state()["hold"] is None

    def test_clips_exist_for_every_game_frame(self):
        anim = mascot.MascotAnimator(available=mascot.FRAME_NAMES, rng=random.Random(1))
        for name in ("giggle", "cover_eyes", "rps_rock", "rps_paper", "rps_scissors",
                     "think_paw_chin", "smug_win"):
            assert anim.has_act(name), name
        anim.set_locomotion("trot_ball")
        assert anim.locomotion == "trot_ball"
        # Without the art the clips are simply not there (the games fall back).
        old = mascot.MascotAnimator(available=OLD_FRAMES, rng=random.Random(1))
        assert not any(old.has_act(n) for n in ("giggle", "cover_eyes", "rps_rock", "smug_win"))

    def test_acting_falls_back_too(self):
        from wawi.pet.acting import clip_for_behaviour

        assert clip_for_behaviour("giggle", {"tickle", "bark"}) == "tickle"
        assert clip_for_behaviour("smug_win", {"party", "bark"}) == "party"
        assert clip_for_behaviour("think_paw_chin", {"tilt", "bark"}) == "tilt"


class TestPeekGeometry:
    WINDOWS = [win(1, 400, 200, 600, 500)]

    def spots(self, frames):
        spots = candidates(self.WINDOWS, [SCREEN], SIZE, frames)
        assert {s.kind for s in spots} == {"top", "right", "left"}
        return spots

    def test_the_cut_edge_lands_on_the_window_edge(self):
        from wawi.games.hide import ART

        w, h = SIZE
        x, y, ww, wh = self.WINDOWS[0].rect
        for s in self.spots(ALL_FRAMES):
            assert s.art
            gx, gy = s.ground
            if s.kind == "top":
                assert gy + ART.top_cut * h == pytest.approx(y + wh)
                assert s.peek[1] == pytest.approx(y + wh)
            elif s.kind == "right":  # unmirrored: the cut is on her left
                assert gx - w / 2 + ART.side_cut * w == pytest.approx(x + ww)
                assert s.peek[0] == pytest.approx(x + ww) and not s.facing_right
            else:  # mirrored about her centre: the cut is on her right
                assert gx + w / 2 - ART.side_cut * w == pytest.approx(x)
                assert s.peek[0] + s.peek[2] == pytest.approx(x) and s.facing_right
            # What shows is outside the window, the rest inside it.
            assert not _overlap(s.peek, (x + 0.5, y + 0.5, ww - 1, wh - 1))
            hx, hy, hw, hh = s.hidden
            assert x - 1 <= hx and hx + hw <= x + ww + 1 and y - 1 <= hy and hy + hh <= y + wh + 1

    def test_without_the_art_the_old_slices(self):
        w, h = SIZE
        for s in self.spots(OLD_FRAMES):
            assert not s.art
            pw, ph = s.peek[2], s.peek[3]
            assert pw * ph == pytest.approx(w * h * (PEEK_TOP if s.kind == "top" else PEEK_SIDE))

    def test_the_game_places_her_for_the_art(self):
        g = HideAndSeek(random.Random(1))
        g.start()
        env = hide_env(self.WINDOWS)
        env.frames = ALL_FRAMES
        for _ in range(60 * 6):
            g.tick(1 / 60, env)
        assert g.phase == "hidden" and g.state()["peek_art"] is True

    def test_art_constants_match_the_peek_frames(self):
        from wawi.games.hide import ART

        ears = _frame_alpha("peek_ears")
        rows, cols = ears.shape
        cut_row = rows - ART.top_cut * rows  # rows count down from the top
        # Just above the cut: her head, full width; below it: outline stubs.
        assert ears[int(cut_row) - 2].sum() > cols * 0.4
        assert all(ears[r].sum() < cols * 0.12 for r in range(int(cut_row) + 1, rows))
        ys, xs = np.nonzero(ears)
        assert xs.min() / cols == pytest.approx(ART.top_x[0], abs=0.01)
        assert (xs.max() + 1) / cols == pytest.approx(ART.top_x[1], abs=0.01)
        assert (rows - ys.min()) / rows == pytest.approx(ART.top_y1, abs=0.01)
        for name in ("peek_tail_a", "peek_tail_b"):
            tail = _frame_alpha(name)
            rows, cols = tail.shape
            cut_col = ART.side_cut * cols
            # Right of the cut: her rump, full height (the edge is straight).
            assert tail[:, int(cut_col) + 1].sum() > rows * 0.4
            ys, xs = np.nonzero(tail)
            assert (xs.max() + 1) / cols <= ART.side_x1 + 0.01
            assert (rows - ys.min()) / rows <= ART.side_y1 + 0.01
        tail = _frame_alpha("peek_tail_a")
        ys, xs = np.nonzero(tail)
        assert (xs.max() + 1) / tail.shape[1] == pytest.approx(ART.side_x1, abs=0.01)
        assert (tail.shape[0] - ys.min()) / tail.shape[0] == pytest.approx(ART.side_y1, abs=0.01)


class TestBodyArt:
    def anim(self, frames=mascot.FRAME_NAMES):
        return mascot.MascotAnimator(available=frames, rng=random.Random(1))

    def view(self, **state):
        from wawi.games.manager import View

        return View("escondidas", None, [], None, state, free=True)

    def test_peek_frames_when_placed_for_them(self):
        from wawi.games.ui import PEEK_EARS, PEEK_EARS_ART, PEEK_TAIL_ART, apply_to_body

        anim = self.anim()
        assert apply_to_body(self.view(ground=(1, 2), peek="top", peek_art=True), FakeOut(),
                             anim, None) == PEEK_EARS_ART.name
        assert anim.act_name == PEEK_EARS_ART.name
        assert apply_to_body(self.view(ground=(1, 2), peek="left", peek_art=True,
                                       facing_right=True), FakeOut(), anim,
                             PEEK_EARS_ART.name) == PEEK_TAIL_ART.name
        assert anim.mirrored
        # Placed for the old slices (no art when she hid): the old clips.
        assert apply_to_body(self.view(ground=(1, 2), peek="top", peek_art=False), FakeOut(),
                             anim, None) == PEEK_EARS.name

    def test_peek_tail_with_only_one_frame(self):
        from wawi.games.ui import PEEK_TAIL_ONE, peek_clip

        frames = set(mascot.FRAME_NAMES) - {"peek_tail_b"}
        assert peek_clip("right", True, frames) is PEEK_TAIL_ONE

    def test_cover_eyes_held_while_counting(self):
        from wawi.games.ui import apply_to_body

        anim = self.anim()
        last = apply_to_body(self.view(hold="cover_eyes"), FakeOut(), anim, None)
        assert last == "cover_eyes" and anim.act_name == "cover_eyes"
        apply_to_body(self.view(invisible=True), FakeOut(), anim, last)
        assert anim.act_name is None
        # Without the frame: nothing held, nothing broken.
        old = self.anim(OLD_FRAMES)
        assert apply_to_body(self.view(hold="cover_eyes"), FakeOut(), old, None) is None

    def test_a_reaction_after_the_peek_is_not_cut(self):
        from wawi.games.ui import PEEK_EARS_ART, apply_to_body

        anim = self.anim()
        last = apply_to_body(self.view(ground=(1, 2), peek="top", peek_art=True), FakeOut(),
                             anim, None)
        anim.act("giggle")  # found: the reaction takes the stage
        apply_to_body(self.view(), FakeOut(), anim, last)
        assert last == PEEK_EARS_ART.name and anim.act_name == "giggle"

    def test_fetch_carries_the_ball_in_her_mouth(self):
        from wawi.games.manager import View
        from wawi.games.ui import apply_to_body, ball_in_mouth

        st = {"ground": (300.0, 60.0), "loco": "trot", "facing_right": False, "clip": None,
              "carried": "return", "ball": (280.0, 100.0)}
        anim = self.anim()
        apply_to_body(View("pelota", None, [], None, st, free=True), FakeOut(), anim, None)
        assert anim.locomotion == "trot_ball"
        assert ball_in_mouth(st, anim.frames)
        assert ball_in_mouth({"carried": "grab"}, anim.frames)
        # Dropped at her feet: the ball is its own window again.
        assert not ball_in_mouth({**st, "carried": None}, anim.frames)
        # Without the gallop-with-ball frames: the plain gallop, ball riding along.
        old = self.anim(OLD_FRAMES)
        apply_to_body(View("pelota", None, [], None, st, free=True), FakeOut(), old, None)
        assert old.locomotion == "trot" and not ball_in_mouth(st, old.frames)

    def test_fetch_state_says_what_she_carries(self):
        g = Fetch(random.Random(1))
        env = fetch_env()
        g.tick(1 / 60, env)
        g.grab(700, 300, 0.0)
        g.release(700, 300, 0.1)
        seen = set()
        for _ in range(60 * 12):
            g.tick(1 / 60, env)
            seen.add(g.state()["carried"])
            if g.phase == "ready" and g.fetches:
                break
        assert {"grab", "return", None} <= seen


# ---------------------------------------------------------------------------
# The voice tool, the module door, settings
# ---------------------------------------------------------------------------

@pytest.fixture
def installed(monkeypatch):
    m, _ = manager()
    monkeypatch.setattr(games, "_MANAGER", m)
    return m


class TestTool:
    def test_hidden_without_games(self, monkeypatch):
        monkeypatch.setattr(games, "_MANAGER", None)
        monkeypatch.setattr(tools, "_MIND", None)
        assert "play_game" not in {t["name"] for t in tools.active_tools()}
        r = tools.dispatch("play_game", {"name": "ppt"})
        assert r.ok is False

    def test_offered_with_games(self, installed, monkeypatch):
        monkeypatch.setattr(tools, "_MIND", None)
        assert "play_game" in {t["name"] for t in tools.active_tools()}

    def test_one_compact_schema(self):
        assert len(tools.GAME_TOOLS) == 1
        assert len(json.dumps(tools.GAME_TOOLS)) < 400

    def test_start_move_stop(self, installed):
        r = tools.dispatch("play_game", {"name": "tateti"})
        assert r.ok and installed.playing == "tateti"
        r = tools.dispatch("play_game", {"name": "tateti", "move": "en el centro"})
        assert r.ok and installed.game.board[4] == "X"
        r = tools.dispatch("play_game", {"name": "stop"})
        assert r.ok and installed.playing is None

    def test_unknown_game(self, installed):
        r = tools.dispatch("play_game", {"name": "ajedrez"})
        assert "No conozco" in r.speak

    def test_hear_door(self, installed, monkeypatch):
        assert games.hear("juguemos a la trivia")
        assert installed.playing == "trivia"
        monkeypatch.setattr(games, "_MANAGER", None)
        assert games.hear("piedra") is None


class TestSettingsAndStore:
    def test_defaults_and_garbage(self):
        s = GamesSettings.from_dict({"tictactoe_level": "Fácil", "fetch_max": "x",
                                     "propose_every_min": -3})
        assert s.tictactoe_level == "facil"
        assert s.fetch_max == 6
        assert s.propose_every == 60.0
        assert GamesSettings.from_dict({"tictactoe_level": "imposible"}).tictactoe_level == "normal"

    def test_build_respects_enabled(self):
        class Cfg:
            games = {"enabled": False}

        assert games.build(Cfg()) is None

    def test_store_survives_a_broken_file(self, tmp_path):
        p = tmp_path / "g.json"
        p.write_text("{nope")
        st = GameStore(p).load()
        st.record("ppt", "wawi")
        assert st.save(force=True)
        assert GameStore(p).load().score("ppt")["wawi"] == 1

    def test_is_stop(self):
        assert is_stop("basta")
        assert is_stop("no juego más")
        assert not is_stop("piedra")


# ---------------------------------------------------------------------------
# The voice turn: moves skip the brain; the channel stays open mid-game
# ---------------------------------------------------------------------------

class TestVoiceTurn:
    def test_a_move_is_answered_without_the_brain(self, installed, monkeypatch):
        from tests.test_orchestrator import AUDIO, make_turn_orch
        from wawi import orchestrator as orch_mod

        installed.start("ppt")
        o = make_turn_orch()
        o.stt = type("T", (), {"transcribe": staticmethod(lambda *a, **kw: "piedra")})()

        def no_brain(*a, **kw):
            raise AssertionError("the brain must not be asked for a game move")

        monkeypatch.setattr(orch_mod.brain, "ask_sync", no_brain)
        o._do_turn(AUDIO)
        assert installed.learner.history  # the round was played

    def test_other_questions_still_reach_the_brain(self, installed, monkeypatch):
        from tests.test_orchestrator import AUDIO, make_turn_orch
        from wawi import brain
        from wawi import orchestrator as orch_mod

        installed.start("ppt")
        o = make_turn_orch()
        asked = []

        def fake(user_text, **kw):  # noqa: ARG001
            asked.append(user_text)
            return brain.BrainResult(text="Bien.", tool_calls=())

        monkeypatch.setattr(orch_mod.brain, "ask_sync", fake)
        o._do_turn(AUDIO)
        assert asked == ["cómo va"]

    def test_the_channel_stays_open_mid_game(self, installed):
        from tests.test_orchestrator import make_orch

        o = make_orch("Más alto.")
        assert o._follow_up_seconds() == 0.0
        installed.start("numero")
        assert o._follow_up_seconds() == 20.0
