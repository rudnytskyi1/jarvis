"""The owner's web panel: FastAPI + Jinja, overlay network only (ТЗ F-705).

The panel shows what the owner needs to see and nothing that can be stolen: the
client list carries the *hash* of each token, never the token, and the panel
itself refuses every request that did not come from the overlay network —
including the login form, which is also a piece of information.

Login is a password from the environment, exchanged for a short-lived signed
session cookie. Signing uses the standard library (HMAC), so no extra
dependency enters the hub for this.
"""
from __future__ import annotations

import hashlib
import hmac
import ipaddress
import logging
import os
import secrets
import sqlite3
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from hub import labelling
from hub.audit import AuditLog

log = logging.getLogger(__name__)

TEMPLATES = Path(__file__).resolve().parent / "templates"
COOKIE = "rowan_admin"
#: The panel knows one password, not which member typed it, so the audit says
#: honestly where the click came from (see ``DECISIONS.md``, P2-26).
PANEL_ACTOR = "web-panel"


class WebAdminAuth:
    """Password from the environment, exchanged for a signed cookie."""

    def __init__(self, *, password_env: str, session_minutes: int = 60,
                 secret: bytes | None = None, clock: Any = time.time) -> None:
        self.password_env, self.session_s = password_env, session_minutes * 60
        self.secret = secret or secrets.token_bytes(32)
        self.clock = clock

    def password(self) -> str:
        value = os.environ.get(self.password_env, "")
        if not value:
            raise RuntimeError(f"{self.password_env} is not set")
        return value

    def verify(self, given: str) -> bool:
        try:
            expected = self.password()
        except RuntimeError:
            return False
        return bool(given) and hmac.compare_digest(str(given), expected)

    def issue(self) -> str:
        expires = int(self.clock()) + self.session_s
        nonce = secrets.token_urlsafe(12)
        body = f"{expires}:{nonce}"
        return body + ":" + self._sign(body)

    def valid(self, cookie: str | None) -> bool:
        if not cookie:
            return False
        parts = str(cookie).split(":")
        if len(parts) != 3:
            return False
        body = f"{parts[0]}:{parts[1]}"
        if not hmac.compare_digest(parts[2], self._sign(body)):
            return False
        try:
            return int(parts[0]) >= int(self.clock())
        except ValueError:
            return False

    def _sign(self, body: str) -> str:
        return hmac.new(self.secret, body.encode("utf-8"), hashlib.sha256).hexdigest()[:32]


def overlay_only(allowed: list[str]):
    """The network gate: everything outside the overlay never sees the panel."""
    networks = [ipaddress.ip_network(str(item), strict=False) for item in allowed]

    def allowed_from(host: str | None) -> bool:
        if not host:
            return False
        try:
            address = ipaddress.ip_address(str(host))
        except ValueError:
            return False
        if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
            address = address.ipv4_mapped
        return any(address in network for network in networks)

    return allowed_from


class WebAdminData:
    """Read-only views of the hub database for the panel (ТЗ F-705).

    Every read opens its own short-lived connection: the panel is served from
    whatever thread the web server happens to use, and the hub's own connection
    belongs to the server loop (see ``hub.decision_log``). WAL makes the reads
    free of the writer.
    """

    def __init__(self, path: str | Path, *, root: str | Path | None = None) -> None:
        self.path = str(path)
        #: Crops may only be served from here: the database stores a path, and a
        #: path that escaped the data directory must not reach the web server.
        self.root = Path(root) if root is not None else Path(self.path).resolve().parent

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=5.0)

    def _read(self, sql: str, params: tuple[Any, ...] = ()) -> list[Any]:
        conn = self._connect()
        try:
            return conn.execute(sql, params).fetchall()
        finally:
            conn.close()

    def homes(self) -> list[dict[str, Any]]:
        rows = self._read(
            "SELECT home_id, name, tz, owner_person_id, config_rev,"
            " (SELECT COUNT(*) FROM clients WHERE clients.home_id = homes.home_id) AS clients,"
            " (SELECT COUNT(*) FROM memberships WHERE memberships.home_id = homes.home_id) AS people"
            " FROM homes ORDER BY name")
        return [{"home_id": row[0], "name": row[1], "tz": row[2], "owner": row[3] or "",
                 "config_rev": row[4], "clients": row[5], "people": row[6]} for row in rows]

    def clients(self) -> list[dict[str, Any]]:
        """Token *hashes* only: the panel must never show a usable token."""
        rows = self._read(
            "SELECT client_id, home_id, kind, token_hash, version, hw, last_seen"
            " FROM clients ORDER BY home_id, client_id")
        return [{"client_id": row[0], "home_id": row[1], "kind": row[2],
                 "token": _hash_hint(row[3]), "version": row[4] or "unknown",
                 "hw": row[5] or "", "last_seen": row[6] or "never"} for row in rows]

    def people(self) -> list[dict[str, Any]]:
        rows = self._read(
            "SELECT p.person_id, p.display_name, m.home_id, m.role, m.share_identity,"
            " m.share_presence FROM persons p LEFT JOIN memberships m ON m.person_id = p.person_id"
            " ORDER BY p.display_name, m.home_id")
        return [{"person_id": row[0], "name": row[1], "home_id": row[2] or "",
                 "role": row[3] or "not a member",
                 "shares": _shares(row[4], row[5])} for row in rows]

    def counts(self) -> dict[str, int]:
        return {"homes": len(self.homes()), "clients": len(self.clients()),
                "people": len({row["person_id"] for row in self.people()}),
                "pending": len(self.unknown_tracks())}

    def audit(self, limit: int = 50) -> list[dict[str, Any]]:
        """The newest privileged actions of the hub (ТЗ F-706)."""
        rows = self._read(
            "SELECT ts, actor_person_id, home_id, action, target, result, detail_json"
            f" FROM audit ORDER BY ts DESC, rowid DESC LIMIT {int(limit)}")
        return [{"when": time.strftime("%Y-%m-%d %H:%M", time.localtime(row[0])),
                 "actor": row[1] or "—", "home_id": row[2] or "—", "action": row[3],
                 "target": row[4] or "—", "result": row[5] or "ok",
                 "detail": row[6] or "{}"} for row in rows]

    # --- manual labelling (ТЗ F-216) ---------------------------------------

    def unknown_tracks(self, *, day: str | None = None, home_id: str | None = None,
                       limit: int = labelling.QUEUE_LIMIT) -> list[dict[str, Any]]:
        """The tracks of one day that nobody could name, with their evidence."""
        conn = self._connect()
        try:
            tracks = labelling.queue(conn, home_id=home_id, day=day, limit=limit)
        finally:
            conn.close()
        return [self._track_view(track) for track in tracks]

    def labelling_people(self) -> list[dict[str, Any]]:
        """Who the track may be bound to: the click names a person, not a name."""
        rows = self._read("SELECT person_id, display_name FROM persons"
                          " ORDER BY display_name, person_id")
        return [{"person_id": row[0], "name": row[1] or row[0]} for row in rows]

    def save_label(self, track_id: str, person_id: str, *, day: str | None = None,
                   crop_id: str = "") -> dict[str, Any]:
        """One click: name the track, link that day's samples, keep the label."""
        conn = self._connect()
        try:
            result = labelling.label(conn, track_id, person_id, day=day, crop_id=crop_id,
                                     actor=PANEL_ACTOR, source="admin", audit=AuditLog(conn))
        except sqlite3.Error as exc:
            log.warning("The panel could not label track %s (%s)", track_id, exc)
            return {"ok": False, "track_id": str(track_id), "person_id": str(person_id),
                    "note": f"database error: {exc}"}
        finally:
            conn.close()
        return result.summary()

    def crop_path(self, crop_id: str) -> Path | None:
        """The JPEG of a crop, and only when it lives under the data directory."""
        conn = self._connect()
        try:
            return labelling.crop_file(conn, crop_id, self.root)
        finally:
            conn.close()

    @staticmethod
    def _track_view(track: labelling.UnknownTrack) -> dict[str, Any]:
        belief = dict(track.belief)
        sources = belief.get("sources") if isinstance(belief.get("sources"), dict) else {}
        return {"track_id": track.track_id, "home_id": track.home_id or "—",
                "first_seen": track.first_seen, "last_seen": track.last_seen,
                "faces": track.faces, "bodies": track.bodies, "voices": track.voices,
                "quality": f"{track.quality:.2f}",
                "belief_p": f"{float(belief.get('p') or 0.0):.2f}",
                "belief_person": belief.get("person_id") or "—",
                "sources": ", ".join(f"{name} {float(value):.2f}"
                                     for name, value in sorted(sources.items())) or "—",
                "crops": [{"crop_id": crop.crop_id} for crop in track.crops]}


def _hash_hint(token_hash: str) -> str:
    """A fingerprint of the stored hash: enough to compare, useless to present."""
    digest = hashlib.sha256(str(token_hash).encode("utf-8")).hexdigest()
    return f"{digest[:6]}…{digest[-4:]}"


def _shares(identity: Any, presence: Any) -> str:
    words = []
    if identity:
        words.append("identity")
    if presence:
        words.append("presence")
    return ", ".join(words) or "nothing shared"


def build_router(*, cfg: Any, data: WebAdminData | None, auth: WebAdminAuth) -> APIRouter:
    """The panel's routes; every one of them sits behind the overlay gate."""
    settings = cfg.server.web_admin
    allowed_from = overlay_only(list(settings.allowed_networks))
    templates = Jinja2Templates(directory=str(TEMPLATES))
    router = APIRouter()

    def denied(request: Request):
        if not allowed_from(request.client.host if request.client else None):
            log.warning("The web panel refused %s", request.client.host if request.client else "?")
            return HTMLResponse("<h1>404</h1><p>This panel is only reachable from the "
                                "overlay network.</p>", status_code=404)
        return None

    def signed_in(request: Request) -> bool:
        return auth.valid(request.cookies.get(COOKIE))

    @router.get("/admin", response_class=HTMLResponse)
    async def home(request: Request):
        if (blocked := denied(request)) is not None:
            return blocked
        if not signed_in(request):
            return templates.TemplateResponse(request, "admin/login.html",
                                              {"error": "Sign in to open the panel."})
        if data is None:
            return templates.TemplateResponse(request, "admin/login.html",
                                              {"error": "The hub database is unavailable."})
        return templates.TemplateResponse(request, "admin/dashboard.html",
                                          {"counts": data.counts()})

    @router.post("/admin/login", response_class=HTMLResponse)
    async def login(request: Request, password: str = Form("")):  # noqa: S107 - a form field
        if (blocked := denied(request)) is not None:
            return blocked
        if not auth.verify(password):
            log.warning("A web panel login failed from %s",
                        request.client.host if request.client else "?")
            return templates.TemplateResponse(request, "admin/login.html",
                                              {"error": "Wrong password."}, status_code=403)
        answer = RedirectResponse("/admin/homes", status_code=303)
        answer.set_cookie(COOKIE, auth.issue(), httponly=True, samesite="lax",
                          max_age=auth.session_s)
        return answer

    @router.get("/admin/logout")
    async def logout(request: Request):
        if (blocked := denied(request)) is not None:
            return blocked
        answer = RedirectResponse("/admin", status_code=303)
        answer.delete_cookie(COOKIE)
        return answer

    def page(template: str, rows_key: str, rows_fn):
        async def view(request: Request):
            if (blocked := denied(request)) is not None:
                return blocked
            if not signed_in(request):
                return RedirectResponse("/admin", status_code=303)
            rows = rows_fn() if data is not None else []
            return templates.TemplateResponse(request, template,
                                              {rows_key: rows, "counts": data.counts() if data else {}})

        return view

    router.add_api_route("/admin/homes", page("admin/homes.html", "homes",
                                              lambda: data.homes() if data else []),
                         methods=["GET"], response_class=HTMLResponse)
    router.add_api_route("/admin/clients", page("admin/clients.html", "clients",
                                                lambda: data.clients() if data else []),
                         methods=["GET"], response_class=HTMLResponse)
    router.add_api_route("/admin/people", page("admin/people.html", "people",
                                               lambda: data.people() if data else []),
                         methods=["GET"], response_class=HTMLResponse)
    router.add_api_route("/admin/audit", page("admin/audit.html", "events",
                                              lambda: data.audit(50) if data else []),
                         methods=["GET"], response_class=HTMLResponse)

    def tracks_page(request: Request, day: str = ""):
        """ТЗ F-216: the unrecognized tracks of a day, with crops and numbers."""
        if (blocked := denied(request)) is not None:
            return blocked
        if not signed_in(request):
            return RedirectResponse("/admin", status_code=303)
        wanted = str(day)[:10] or labelling.day_of()
        return templates.TemplateResponse(request, "admin/tracks.html", {
            "tracks": data.unknown_tracks(day=wanted) if data is not None else [],
            "people": data.labelling_people() if data is not None else [],
            "day": wanted, "message": request.query_params.get("message", ""),
            "counts": data.counts() if data is not None else {}})

    router.add_api_route("/admin/tracks", tracks_page, methods=["GET"],
                         response_class=HTMLResponse)

    @router.post("/admin/label")
    async def apply_label(request: Request, track_id: str = Form(""),
                          person_id: str = Form(""), day: str = Form(""),
                          crop_id: str = Form("")):
        if (blocked := denied(request)) is not None:
            return blocked
        if not signed_in(request):
            return RedirectResponse("/admin", status_code=303)
        wanted = str(day)[:10]
        query: dict[str, str] = {"day": wanted} if wanted else {}
        if data is None or not track_id or not person_id:
            query["message"] = "Nothing changed: pick a track and a person."
        else:
            result = data.save_label(track_id, person_id, day=wanted or None, crop_id=crop_id)
            query["message"] = (f"Track {track_id} is now {result['name']}."
                                if result.get("ok")
                                else f"Nothing changed: {result.get('note', 'failed')}")
        return RedirectResponse("/admin/tracks?" + urlencode(query), status_code=303)

    @router.get("/admin/crop/{crop_id}.jpg")
    async def crop(request: Request, crop_id: str):
        """The crop of a track, from the data directory and nowhere else."""
        if (blocked := denied(request)) is not None:
            return blocked
        path = data.crop_path(crop_id) if (data is not None and signed_in(request)) else None
        if path is None:
            return HTMLResponse("<h1>404</h1><p>No such crop.</p>", status_code=404)
        return Response(content=path.read_bytes(), media_type="image/jpeg")

    return router


def mount(app: Any, *, cfg: Any, data: WebAdminData | None) -> Any:
    """Attach the panel to the hub's FastAPI app when it is switched on."""
    settings = getattr(cfg.server, "web_admin", None)
    if settings is None or not settings.enabled:
        return None
    auth = WebAdminAuth(password_env=settings.password_env,
                        session_minutes=settings.session_minutes)
    if not os.environ.get(settings.password_env):
        log.warning("The web panel is enabled but %s is not set: nobody can sign in",
                    settings.password_env)
    app.include_router(build_router(cfg=cfg, data=data, auth=auth))
    log.info("The owner web panel listens on http://%s:%s/admin (overlay only)",
             settings.host, settings.port)
    return auth


__all__ = ["COOKIE", "PANEL_ACTOR", "TEMPLATES", "WebAdminAuth", "WebAdminData", "build_router",
           "mount", "overlay_only", "quote"]
