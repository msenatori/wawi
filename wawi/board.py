"""Small, local-first visual board beside Wawi.

The model thread prepares documents; PanelObserver's render tick performs every
AppKit/WebKit operation on the main thread. No web content receives a native
bridge. Network requests are denied by a WebKit content rule list by default.
"""
from __future__ import annotations

import html
import base64
import logging
import re
import tempfile
import threading
import time
import webbrowser
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)
MAX_CONTENT = 200_000
#: Ceiling with embedded images included (one photo of up to 6 MB, base64).
MAX_WITH_IMAGES = 9_000_000
_EMBEDDED_IMAGE = re.compile(r"data:image/(?:jpeg|png|gif|webp);base64,[A-Za-z0-9+/=]+")
HISTORY = 10
ROLL_ART = Path(__file__).parent / "assets" / "wawi" / "prop_abilities_roll.png"


@dataclass(frozen=True)
class BoardPage:
    title: str
    source: str
    format: str
    html: str


_CSS = """
:root{color-scheme:light dark;--paper:#fffbee;--ink:#292f68;--wash:#eee9dc;--muted:#555b75;
  font:15px/1.55 -apple-system,BlinkMacSystemFont,sans-serif}
*{box-sizing:border-box}
html{background:var(--paper)}
body{margin:0;min-height:100vh;padding:20px 23px 36px;color:var(--ink);background:var(--paper);overflow-wrap:break-word}
h1,h2,h3,h4{font-family:"Chalkboard SE","Marker Felt",-apple-system,sans-serif;line-height:1.22;margin:0 0 .55em;font-weight:700}
h1{font-size:1.65rem}h2{font-size:1.36rem}h3{font-size:1.15rem}
p,ul,ol,table,pre,blockquote{margin:0 0 1em}ul,ol{padding-left:1.45em}li{padding-left:.15em;margin:.2em 0}
li::marker{color:var(--ink)}a{color:#354b91;text-decoration-thickness:2px;text-underline-offset:2px}
hr{border:0;border-top:3px solid var(--ink);border-radius:50%;transform:rotate(-.3deg);margin:22px 2px}
pre,code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
pre{overflow:auto;background:var(--wash);border:2px solid var(--ink);border-radius:10px 13px 9px 12px;padding:12px;box-shadow:2px 2px 0 #292f6833}
:not(pre)>code{background:var(--wash);padding:2px 5px;border-radius:5px}
table{border-collapse:separate;border-spacing:0;width:100%;border:2px solid var(--ink);border-radius:9px 12px 8px 11px;overflow:hidden}
th,td{text-align:left;border-bottom:1.5px solid var(--ink);border-right:1.5px solid var(--ink);padding:7px 9px}
tr:last-child td{border-bottom:0}th:last-child,td:last-child{border-right:0}th{background:var(--wash);font-family:"Chalkboard SE","Marker Felt",sans-serif}
blockquote{border-left:4px solid var(--ink);padding:5px 0 5px 13px;color:var(--muted);background:linear-gradient(90deg,var(--wash),transparent)}
button,input,textarea,select{font:inherit;color:var(--ink);background:var(--paper);border:2px solid var(--ink);border-radius:8px 11px 7px 10px;padding:6px 10px}
button{font-family:"Chalkboard SE","Marker Felt",sans-serif;font-weight:700;cursor:pointer;box-shadow:2px 2px 0 #292f6840}
button:hover{background:var(--wash)}button:active{transform:translate(1px,1px);box-shadow:1px 1px 0 #292f6840}
input:focus,textarea:focus,select:focus,button:focus-visible{outline:2px dashed var(--ink);outline-offset:2px}
img,video,canvas,iframe{max-width:100%}
.image-card{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(0,1fr);gap:20px;align-items:start}
.image-frame{margin:0;padding:8px;background:var(--wash);border:3px solid var(--ink);border-radius:12px 9px 14px 10px;box-shadow:3px 4px 0 #292f6833;transform:rotate(-.35deg)}
.image-frame img{display:block;width:100%;max-height:70vh;object-fit:contain;border-radius:5px}
.image-details h1{border-bottom:2px solid var(--ink);padding-bottom:8px}
.image-details p{white-space:pre-line}.image-credit{font-size:.83rem;color:var(--muted);border-top:2px dashed var(--ink);padding-top:10px}
@media(max-width:620px){.image-card{grid-template-columns:1fr}.image-frame img{max-height:52vh}}
@media(prefers-color-scheme:dark){:root{--paper:#263b34;--ink:#f4f0d9;--wash:#344c41;--muted:#d6d2bd}a{color:#d5e7f2}pre,button{box-shadow:2px 2px 0 #101c1880}}
"""


def render_page(title: str, content: str, format: str = "markdown") -> BoardPage:
    title = str(title).strip()[:120] or "Pizarra"
    content = str(content)
    # Embedded pictures (wawi/images.py: validated, at most MAX_IMAGE_BYTES)
    # do not count: a curiosity with its photo was refused as too long.
    text = _EMBEDDED_IMAGE.sub("", content)
    if not text.strip() or len(text) > MAX_CONTENT or len(content) > MAX_WITH_IMAGES:
        raise ValueError("Board content must be non-empty and under 200 KB")
    if format not in {"markdown", "html"}:
        raise ValueError("Board format must be markdown or html")
    if format == "markdown":
        import markdown
        body = markdown.markdown(content, extensions=["tables", "fenced_code"])
    else:
        body = content
    # loadHTMLString with a nil base URL denies local-file resolution. The
    # WebKit rule list below denies network subresources as well as navigation.
    document = ("<!doctype html><html><head><meta charset='utf-8'>"
                "<meta name='viewport' content='width=device-width,initial-scale=1'>"
                f"<title>{html.escape(title)}</title><style>{_CSS}</style></head>"
                f"<body>{body}</body></html>")
    return BoardPage(title, content, format, document)


def render_image_page(card: Any) -> BoardPage:
    """Render only locally embedded, validated image bytes in the offline board."""
    if not card.data_uri.startswith(("data:image/jpeg;base64,", "data:image/png;base64,",
                                     "data:image/gif;base64,", "data:image/webp;base64,")):
        raise ValueError("La imagen no tiene un formato seguro.")
    title = html.escape(card.title)
    source = html.escape(card.source_url, quote=True)
    body = ("<article class='image-card'><figure class='image-frame'>"
            f"<img src='{card.data_uri}' alt='{title}'></figure>"
            f"<section class='image-details'><h1>{title}</h1>"
            f"<p>{html.escape(card.description)}</p>"
            "<div class='image-credit'>"
            f"<p>Crédito: {html.escape(card.credit)}</p>"
            f"<p>Licencia: {html.escape(card.license)}</p>"
            f"<a href='{source}'>Ver fuente</a></div></section></article>")
    document = ("<!doctype html><html><head><meta charset='utf-8'>"
                "<meta name='viewport' content='width=device-width,initial-scale=1'>"
                f"<title>{title}</title><style>{_CSS}</style></head>"
                f"<body>{body}</body></html>")
    return BoardPage(card.title, card.source_url, "image", document)


def render_abilities(groups: list[tuple[str, list[tuple[str, str, str]]]]) -> BoardPage:
    """A scroll of active capabilities; all tool-provided text is escaped."""
    roller = ROLL_ART
    art = ""
    if roller.is_file():
        art = ("<img class='roller-art' alt='' src='data:image/png;base64,"
               + base64.b64encode(roller.read_bytes()).decode("ascii") + "'>")
    sections = []
    for heading, entries in groups:
        cards = "".join(
            "<li><strong>" + html.escape(description) + "</strong>"
            "<small>«" + html.escape(example) + "»</small></li>"
            for _name, description, example in entries)
        sections.append("<section><h2>" + html.escape(heading) + "</h2><ul>" + cards + "</ul></section>")
    body = ("<div class='abilities'><div class='roller" + (" has-art" if art else "")
            + "'>" + art + "</div>"
            "<div class='sheet'><h1>Todo lo que sé hacer</h1>" + "".join(sections)
            + "</div></div>")
    style = """<style>
    html,body{height:100%;overflow:hidden;background:transparent!important}
    body{padding:0!important}.abilities{height:100%;overflow-y:auto;overflow-x:hidden;
      scrollbar-color:#292f68 #fffbee}.roller{position:sticky;top:0;z-index:2;height:75px;
      border-bottom:4px solid #292f68;border-radius:50% 50% 10px 10px;
      background:linear-gradient(#b9a17a,#eee0b1 48%,#b9a17a);box-shadow:0 5px 7px #292f6838}
    .roller-art{position:absolute;left:50%;top:-13px;width:90%;max-width:440px;
      height:72px;object-fit:contain;transform:translateX(-50%)}
    .roller.has-art{background:#fffbee;border:0;box-shadow:none}
    .sheet{margin:0 15px 14px;padding:24px 21px 35px;background:#fffbee;color:#292f68;
      border:3px solid #292f68;border-top:0;border-radius:0 0 14px 12px;
      box-shadow:4px 7px 0 #292f6826;transform-origin:top;animation:unroll 1.25s ease-out both}
    .sheet h1{text-align:center;border-bottom:2px dashed #292f68;padding-bottom:10px}
    .sheet h2{margin:22px 0 8px;font-size:1.15rem}.sheet ul{list-style:none;padding:0}
    .sheet li{padding:7px 0;border-bottom:1px solid #292f6830}.sheet small{display:block;margin-top:2px;color:#555b75}
    .abilities.closing .sheet{animation:reroll .45s ease-in both}
    @keyframes unroll{from{transform:scaleY(.025);opacity:.3}to{transform:scaleY(1);opacity:1}}
    @keyframes reroll{to{transform:scaleY(.025);opacity:0}}
    @media(prefers-reduced-motion:reduce){.sheet,.abilities.closing .sheet{animation:none}}
    </style>"""
    page = render_page("Todo lo que sé hacer", style + body, "html")
    return BoardPage(page.title, page.source, "abilities", page.html)


class Board:
    def __init__(self, *, allow_network: bool = False, directory: str | Path = "board_local") -> None:
        self.allow_network = bool(allow_network)
        self.directory = Path(directory)
        self.history: deque[BoardPage] = deque(maxlen=HISTORY)
        self.current: BoardPage | None = None
        self._pending: BoardPage | bool | None = None
        self._lock = threading.Lock()
        self._panel: Any = None
        self._web: Any = None
        self._controls: list[Any] = []
        self._rule_list: Any = None
        self._closing_abilities = False
        self._abilities_deadline = 0.0
        self._abilities_fallback = ""

    def show(self, title: str, content: str, format: str = "markdown") -> BoardPage:
        page = render_page(title, content, format)
        return self._show_page(page)

    def show_image(self, card: Any) -> BoardPage:
        return self._show_page(render_image_page(card))

    def _show_page(self, page: BoardPage) -> BoardPage:
        with self._lock:
            self.history.append(page)
            self.current = page
            self._pending = page
        self._persist()
        return page

    def show_abilities(self, groups: list[tuple[str, list[tuple[str, str, str]]]],
                       scenic: bool = True) -> BoardPage:
        """Queue the hand-drawn scroll; fall back to the normal board without art."""
        content = "\n\n".join("## " + heading + "\n" + "\n".join(
            f"- {html.escape(description)} — «{html.escape(example)}»" for _, description, example in entries)
            for heading, entries in groups)
        if not scenic or not ROLL_ART.is_file():
            return self.show("Todo lo que sé hacer", content)
        page = render_abilities(groups)
        with self._lock:
            self._abilities_deadline = time.monotonic() + 45.0
            self._abilities_fallback = content
            self.history.append(page)
            self.current = page
            self._pending = page
        self._persist()
        return page

    def previous(self) -> BoardPage | None:
        with self._lock:
            if len(self.history) < 2:
                return None
            page = self.history[-2]
            self.current = page
            self._pending = page
            return page

    def hide(self) -> bool:
        with self._lock:
            visible = self.current is not None
            abilities = visible and self.current.format == "abilities"
            self.current = None
            self._pending = False
            self._closing_abilities = bool(abilities)
        if abilities:
            try:
                from wawi.pet import api
                api.trick("ladder_down")
            except ImportError:
                pass
        return visible

    def _persist(self) -> None:
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            for old in self.directory.glob("board-*.html"):
                old.unlink()
            for index, page in enumerate(self.history):
                (self.directory / f"board-{index:02d}.html").write_text(page.html, encoding="utf-8")
        except OSError:
            log.warning("Could not persist board history", exc_info=True)

    def tick(self, mascot_window: Any = None) -> None:
        """Consume worker-thread requests on the AppKit main thread."""
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("Board.tick must run on the main thread")
        with self._lock:
            pending, self._pending = self._pending, None
        if pending is None:
            return
        if pending is False:
            if self._panel is not None:
                if self._closing_abilities and self._web is not None:
                    self._web.evaluateJavaScript_completionHandler_(
                        "document.querySelector('.abilities')?.classList.add('closing')", None)
                    self._closing_abilities = False
                    threading.Timer(.48, self._queue_hide).start()
                    return
                self._panel.orderOut_(None)
            return
        if pending.format == "abilities":
            from wawi.pet import api
            if not api.at_window():
                if time.monotonic() < self._abilities_deadline:
                    with self._lock:
                        if self.current is pending:
                            self._pending = pending
                    return
                pending = self.show("Todo lo que sé hacer", self._abilities_fallback)
                with self._lock:
                    self._pending = None
        self._show_native(pending, mascot_window)

    def _queue_hide(self) -> None:
        with self._lock:
            if self.current is None:
                self._pending = False

    def _show_native(self, page: BoardPage, mascot_window: Any) -> None:
        try:
            from AppKit import (NSBackingStoreBuffered, NSColor, NSFloatingWindowLevel, NSTextField, NSFont,
                NSMakeRect, NSPanel, NSViewWidthSizable, NSViewHeightSizable,
                NSWindowCollectionBehaviorCanJoinAllSpaces,
                NSWindowCollectionBehaviorFullScreenAuxiliary,
                NSWindowStyleMaskBorderless, NSWindowStyleMaskResizable, NSScreen, NSViewMinXMargin, NSViewMinYMargin)
            from WebKit import WKWebView, WKWebViewConfiguration, WKWebsiteDataStore, WKContentRuleListStore
        except ImportError:
            self._open_browser(page)
            return
        if self._panel is None:
            frame = NSMakeRect(400, 300, 570, 490)
            panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
                frame, NSWindowStyleMaskBorderless | NSWindowStyleMaskResizable,
                NSBackingStoreBuffered, False)
            panel.setBackgroundColor_(NSColor.clearColor())
            panel.setOpaque_(False)
            panel.setHasShadow_(True)
            panel.setMinSize_((360, 260))
            panel.setLevel_(NSFloatingWindowLevel)
            panel.setMovableByWindowBackground_(True)
            panel.setHidesOnDeactivate_(False)
            panel.setReleasedWhenClosed_(False)
            panel.setCollectionBehavior_(NSWindowCollectionBehaviorCanJoinAllSpaces |
                                         NSWindowCollectionBehaviorFullScreenAuxiliary)
            frame_view = _BoardFrame.alloc().initWithFrame_(NSMakeRect(0, 0, 570, 490))
            frame_view.setAutoresizingMask_(NSViewWidthSizable | NSViewHeightSizable)
            panel.contentView().addSubview_(frame_view)
            config = WKWebViewConfiguration.alloc().init()
            config.setWebsiteDataStore_(WKWebsiteDataStore.nonPersistentDataStore())
            web = WKWebView.alloc().initWithFrame_configuration_(NSMakeRect(8, 17, 554, 419), config)
            web.setAutoresizingMask_(NSViewWidthSizable | NSViewHeightSizable)
            web.setWantsLayer_(True)
            web.layer().setCornerRadius_(9)
            web.layer().setMasksToBounds_(True)
            panel.contentView().addSubview_(web)
            title_view = NSTextField.alloc().initWithFrame_(NSMakeRect(22, 451, 374, 28))
            title_view.setStringValue_(page.title)
            title_view.setFont_(NSFont.fontWithName_size_("Chalkboard SE", 17) or NSFont.boldSystemFontOfSize_(16))
            title_view.setTextColor_(_board_ink(panel))
            title_view.setEditable_(False)
            title_view.setBordered_(False)
            title_view.setDrawsBackground_(False)
            title_view.setAutoresizingMask_(NSViewMinYMargin)
            panel.contentView().addSubview_(title_view)
            self._controls.append(title_view)
            target = _ButtonTarget.alloc().initWithBoard_(self)
            self._controls.append(target)
            for symbol, tooltip, action, x in (("copy", "Copiar", "copy:", 414),
                                                ("open", "Abrir", "open:", 450),
                                                ("save", "Guardar", "save:", 486),
                                                ("close", "Cerrar", "close:", 532)):
                button = _BoardButton.alloc().initWithFrame_(NSMakeRect(x, 448, 30, 30))
                button.setTitle_(symbol)
                button.setToolTip_(tooltip)
                button.setTarget_(target)
                button.setAction_(action)
                button.setAutoresizingMask_(NSViewMinXMargin | NSViewMinYMargin)
                panel.contentView().addSubview_(button)
                self._controls.append(button)
            grip = _ResizeGrip.alloc().initWithFrame_(NSMakeRect(548, 0, 22, 22))
            grip.setAutoresizingMask_(NSViewMinXMargin)
            panel.contentView().addSubview_(grip)
            self._controls.append(grip)
            self._panel, self._web = panel, web
            # Always deny document navigation outside the injected about:blank
            # page. In offline mode also deny every network subresource.
            import json
            network_types = (["document"] if self.allow_network else
                             ["document", "image", "style-sheet", "script", "font", "media", "raw"])
            rules = json.dumps([
                {"trigger": {"url-filter": "^" + scheme + "://", "resource-type": types},
                 "action": {"type": "block"}}
                for scheme, types in (("http", network_types), ("https", network_types),
                                      ("file", ["document", "image", "style-sheet", "script", "font", "media", "raw"]),
                                      ("ftp", ["document", "image", "style-sheet", "script", "font", "media", "raw"]))
            ])
            def compiled(rule, error):
                if rule is None:
                    log.error("Board content blocker failed: %s", error)
                    self._panel.orderOut_(None)
                    return
                self._rule_list = rule
                config.userContentController().addContentRuleList_(rule)
                self._web.loadHTMLString_baseURL_(self.current.html if self.current else page.html, None)
            WKContentRuleListStore.defaultStore().compileContentRuleListForIdentifier_encodedContentRuleList_completionHandler_(
                "wawi-board-network-" + str(self.allow_network), rules, compiled)
        else:
            self._web.loadHTMLString_baseURL_(page.html, None)
            self._controls[0].setStringValue_(page.title)
            self._controls[0].setTextColor_(_board_ink(self._panel))
        if mascot_window is not None:
            m = mascot_window.frame()
            screen = NSScreen.mainScreen().visibleFrame()
            if page.format == "abilities":
                x = min(max(m.origin.x + m.size.width * .5 - 285, screen.origin.x),
                        screen.origin.x + screen.size.width - 570)
                y = min(max(m.origin.y - 430, screen.origin.y),
                        screen.origin.y + screen.size.height - 490)
            else:
                x = min(max(m.origin.x + m.size.width + 14, screen.origin.x),
                        screen.origin.x + screen.size.width - 570)
                y = min(max(m.origin.y, screen.origin.y), screen.origin.y + screen.size.height - 490)
            self._panel.setFrameOrigin_((x, y))
        self._panel.orderFrontRegardless()

    def copy(self) -> None:
        from AppKit import NSPasteboard, NSPasteboardTypeString
        if self.current:
            pasteboard = NSPasteboard.generalPasteboard()
            pasteboard.clearContents()
            pasteboard.setString_forType_(self.current.source, NSPasteboardTypeString)

    def _open_browser(self, page: BoardPage | None = None) -> Path | None:
        page = page or self.current
        if page is None:
            return None
        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", prefix="wawi-board-",
                                         encoding="utf-8", delete=False) as file:
            file.write(page.html)
            path = Path(file.name)
        webbrowser.open(path.as_uri())
        return path

    def save(self) -> Path | None:
        if not self.current:
            return None
        self.directory.mkdir(parents=True, exist_ok=True)
        name = re.sub(r"[^a-zA-Z0-9_-]+", "-", self.current.title).strip("-")[:50] or "board"
        path = self.directory / f"saved-{name}.html"
        path.write_text(self.current.html, encoding="utf-8")
        return path


try:
    import objc
    from Foundation import NSObject
    from AppKit import (NSView, NSButton, NSColor, NSBezierPath, NSMakeRect, NSAppearanceNameDarkAqua,
                        NSAppearanceNameAqua)

    def _board_dark(view):
        return view.effectiveAppearance().bestMatchFromAppearancesWithNames_(
            [NSAppearanceNameAqua, NSAppearanceNameDarkAqua]) == NSAppearanceNameDarkAqua

    def _board_ink(view):
        rgb = (0.96, 0.94, 0.85) if _board_dark(view) else (0.16, 0.18, 0.41)
        return NSColor.colorWithSRGBRed_green_blue_alpha_(*rgb, 1.0)

    class _BoardFrame(NSView):
        """Paper/chalkboard surround, drawn with Wawi's doubled navy crayon line."""

        def isOpaque(self):
            return False

        def drawRect_(self, rect):
            b = self.bounds()
            w, h = b.size.width, b.size.height
            dark = _board_dark(self)
            fill = NSColor.colorWithSRGBRed_green_blue_alpha_(
                *((0.15, 0.23, 0.20) if dark else (1.0, 0.985, 0.94)), 1.0)
            ink = _board_ink(self)
            outline = NSBezierPath.bezierPath()
            outline.moveToPoint_((13, 5))
            outline.curveToPoint_controlPoint1_controlPoint2_((w - 14, 6), (w * .32, 2), (w * .72, 8))
            outline.curveToPoint_controlPoint1_controlPoint2_((w - 5, 20), (w - 7, 5), (w - 4, 9))
            outline.lineToPoint_((w - 5, h - 17))
            outline.curveToPoint_controlPoint1_controlPoint2_((w - 18, h - 5), (w - 5, h - 7), (w - 7, h - 5))
            outline.curveToPoint_controlPoint1_controlPoint2_((17, h - 6), (w * .66, h - 8), (w * .33, h - 3))
            outline.curveToPoint_controlPoint1_controlPoint2_((7, h - 18), (6, h - 6), (5, h - 9))
            outline.lineToPoint_((8, 97))
            outline.lineToPoint_((1, 77))  # little speech-bubble tail toward Wawi
            outline.lineToPoint_((9, 70))
            outline.lineToPoint_((9, 17))
            outline.curveToPoint_controlPoint1_controlPoint2_((13, 5), (8, 9), (9, 6))
            outline.closePath()
            fill.setFill()
            outline.fill()
            ink.setStroke()
            outline.setLineWidth_(3.2)
            outline.stroke()
            ink.colorWithAlphaComponent_(0.36).setStroke()
            outline.setLineWidth_(1.0)
            outline.stroke()
            # A slightly slanted divider echoes the two-pass speech bubble.
            divider = NSBezierPath.bezierPath()
            divider.moveToPoint_((10, h - 49))
            divider.curveToPoint_controlPoint1_controlPoint2_((w - 9, h - 50),
                                                              (w * .35, h - 47), (w * .7, h - 52))
            ink.setStroke()
            divider.setLineWidth_(2.2)
            divider.stroke()

    class _BoardButton(NSButton):
        """Compact crayon controls; the save action never depends on label width."""

        def isOpaque(self):
            return False

        def drawRect_(self, rect):
            ink = _board_ink(self)
            paper = NSColor.colorWithSRGBRed_green_blue_alpha_(
                *((0.20, 0.30, 0.25) if _board_dark(self) else (1.0, 0.985, 0.94)), 1.0)
            box = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                NSMakeRect(2, 2, self.bounds().size.width - 4, self.bounds().size.height - 4), 7, 8)
            paper.setFill()
            box.fill()
            ink.setStroke()
            box.setLineWidth_(2.0)
            box.stroke()
            path = NSBezierPath.bezierPath()
            symbol = self.title()
            if symbol == "close":
                path.moveToPoint_((10, 10)); path.lineToPoint_((21, 20))
                path.moveToPoint_((20, 10)); path.lineToPoint_((10, 20))
            elif symbol == "copy":
                path.moveToPoint_((10, 11)); path.lineToPoint_((10, 21))
                path.lineToPoint_((19, 21)); path.lineToPoint_((19, 11)); path.closePath()
                path.moveToPoint_((13, 9)); path.lineToPoint_((22, 9))
                path.lineToPoint_((22, 18))
            elif symbol == "open":
                path.moveToPoint_((9, 18)); path.lineToPoint_((9, 10))
                path.lineToPoint_((21, 10)); path.lineToPoint_((21, 15))
                path.moveToPoint_((14, 16)); path.lineToPoint_((22, 22))
                path.moveToPoint_((16, 22)); path.lineToPoint_((22, 22)); path.lineToPoint_((22, 16))
            else:  # save
                path.moveToPoint_((10, 22)); path.lineToPoint_((20, 22))
                path.lineToPoint_((22, 19)); path.lineToPoint_((22, 9))
                path.lineToPoint_((9, 9)); path.lineToPoint_((9, 22)); path.closePath()
                path.moveToPoint_((12, 22)); path.lineToPoint_((12, 17))
                path.lineToPoint_((18, 17)); path.lineToPoint_((18, 22))
                path.moveToPoint_((12, 10)); path.lineToPoint_((12, 14))
                path.lineToPoint_((19, 14)); path.lineToPoint_((19, 10))
            path.setLineCapStyle_(1)
            path.setLineJoinStyle_(1)
            path.setLineWidth_(2.0)
            path.stroke()
    class _ResizeGrip(NSView):
        def drawRect_(self, rect):
            NSColor.tertiaryLabelColor().setStroke()
            for offset in (5, 10, 15):
                path = NSBezierPath.bezierPath()
                path.moveToPoint_((offset, 3))
                path.lineToPoint_((19, 22 - offset))
                path.stroke()

        def mouseDown_(self, event):
            self._start = event.locationInWindow()
            self._size = self.window().frame().size

        def mouseDragged_(self, event):
            point = event.locationInWindow()
            frame = self.window().frame()
            frame.size.width = max(360, self._size.width + point.x - self._start.x)
            frame.size.height = max(260, self._size.height + self._start.y - point.y)
            self.window().setFrame_display_(frame, True)

    class _ButtonTarget(NSObject):
        def initWithBoard_(self, board):
            self = objc.super(_ButtonTarget, self).init()
            if self is not None:
                self.board = board
            return self
        def close_(self, sender):
            self.board.hide()
            self.board.tick()
        def copy_(self, sender):
            self.board.copy()
        def open_(self, sender):
            self.board._open_browser()
        def save_(self, sender):
            self.board.save()

except ImportError:
    pass
