"""Web backend: REST + SSE. No application-layer auth: the network layer (loopback + Tailscale ACLs)
is the trust boundary; this module enforces a bind-address policy, a Host allowlist and an Origin
check on writes. Duck-typed on the controller; never imports bot.py."""
import argparse
import asyncio
import base64
import contextlib
import getpass
import hashlib
import hmac
import ipaddress
import json
import logging
import os
import re
import secrets
import subprocess
import sys
import threading
import time
import unicodedata
from urllib.parse import unquote, urlsplit

from aiohttp import web

from config_store import ConflictError, atomic_write_json
from features import FeatureStore
from messages import MAX_MESSAGE_CHARS, MessageError
from voices import VOICE_NAME_RE, BadVoiceError, SayError, VoiceLoadError

log = logging.getLogger("webui")

SESSION_COOKIE = "tts_session"
SESSION_LIFETIME = 7 * 86400.0  # 7 days
MAX_LOGIN_FAILURES = 5
LOGIN_RATE_WINDOW = 60.0  # seconds
AUTH_MANAGER_KEY = web.AppKey("auth_manager", object)
FEATURE_STORE_KEY = web.AppKey("feature_store", object)
USER_KEY = web.RequestKey("user", str)


# ---------- password hashing (stdlib scrypt) ----------
SCRYPT_N, SCRYPT_R, SCRYPT_P, SCRYPT_DKLEN = 2 ** 14, 8, 1, 32
SCRYPT_MAXMEM = 64 * 1024 * 1024
MIN_PASSWORD_LEN, MAX_PASSWORD_LEN = 10, 200
# Passwords that shipped in examples/docs: never accepted, whatever is stored.
KNOWN_BAD_PASSWORDS = frozenset({"changeme", "testerpassword", "password", "admin"})


def hash_password(password: str) -> str:
    """`scrypt$N$r$p$<salt_b64>$<hash_b64>` with a fresh 16-byte random salt."""
    salt = secrets.token_bytes(16)
    dk = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P,
                        maxmem=SCRYPT_MAXMEM, dklen=SCRYPT_DKLEN)
    return "$".join(["scrypt", str(SCRYPT_N), str(SCRYPT_R), str(SCRYPT_P),
                     base64.b64encode(salt).decode("ascii"), base64.b64encode(dk).decode("ascii")])


def _parse_int(s: str) -> int | None:
    return int(s) if s.isascii() and s.isdigit() and len(s) <= 9 else None


def verify_hash(stored: str, password: str) -> bool:
    """Constant-time check against a `scrypt$...` value. Strict parsing; any malformed or
    out-of-range (tampered) value returns False. Never raises."""
    try:
        parts = stored.split("$")
        if len(parts) != 6 or parts[0] != "scrypt":
            return False
        n, r, p = (_parse_int(x) for x in parts[1:4])
        if n is None or r is None or p is None:
            return False
        if not (2 ** 10 <= n <= 2 ** 17 and n & (n - 1) == 0 and 1 <= r <= 16 and 1 <= p <= 4):
            return False
        salt = base64.b64decode(parts[4], validate=True)
        want = base64.b64decode(parts[5], validate=True)
        if not (8 <= len(salt) <= 64 and 16 <= len(want) <= 64):
            return False
        got = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=n, r=r, p=p,
                             maxmem=SCRYPT_MAXMEM, dklen=len(want))
        return hmac.compare_digest(got, want)
    except Exception:
        return False


def password_policy_error(password) -> str | None:
    """Return an error code for a password the Admin tab / CLI must refuse, else None."""
    if not isinstance(password, str) or not (MIN_PASSWORD_LEN <= len(password) <= MAX_PASSWORD_LEN):
        return "weak_password"
    if password.strip().lower() in KNOWN_BAD_PASSWORDS:
        return "weak_password"
    return None


# Verified against when the username is unknown, so both paths cost one scrypt.
_DUMMY_HASH = hash_password(secrets.token_urlsafe(16))


class PasswordRequired(ValueError):
    pass


def _split_raw(raw):
    """Return (records, rebuild) for any supported users.json layout. `records` is a list of
    entries (dicts for users); `rebuild(records)` gives the JSON value to write back, keeping the
    wrapper object and its other fields when there is one."""
    if isinstance(raw, list):
        return raw, lambda recs: recs
    if not isinstance(raw, dict):
        raise ValueError("users file must hold a JSON list or object")
    wrapped = "users" in raw
    inner = raw["users"] if wrapped else raw
    if isinstance(inner, list):
        return inner, lambda recs: {**raw, "users": recs}
    if not isinstance(inner, dict):
        raise ValueError("users file has an unsupported layout")
    recs = []
    for k, v in inner.items():
        if isinstance(v, str):
            recs.append({"username": str(k), "password": v})
        elif isinstance(v, dict):
            recs.append({**v, "username": str(k)})
    return recs, (lambda r: {**raw, "users": r}) if wrapped else (lambda r: r)


def _read_users_file(path: str):
    """(records, rebuild) of the file on disk; ([], identity) when it does not exist yet.
    Raises (OSError / ValueError) rather than guess when it exists but cannot be understood."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        return [], (lambda recs: recs)
    return _split_raw(raw)


def _groups_of(rec: dict) -> set[str]:
    groups: set[str] = set()
    if rec.get("role"):
        groups.add(str(rec["role"]).strip().lower())
    if rec.get("is_admin") is True:
        groups.add("admin")
    if isinstance(rec.get("groups"), list):
        groups.update(str(g).strip().lower() for g in rec["groups"] if g)
    return groups


def _guilds_of(rec: dict) -> set[str]:
    """Server ids a non-admin account may manage (decimal strings); anything else is dropped."""
    raw = rec.get("guilds")
    if not isinstance(raw, list):
        return set()
    return {str(g) for g in raw if isinstance(g, (str, int)) and not isinstance(g, bool)
            and str(g).isascii() and str(g).isdigit() and len(str(g)) <= 25}


def _is_user_rec(rec, name_lower: str) -> bool:
    return isinstance(rec, dict) and str(rec.get("username", "")).strip().lower() == name_lower


class AuthManager:
    """Session-based authentication manager for Web UI.

    Loads credentials from a JSON file (users.json).
    Supports:
      - [{"username": "...", "password": "..."}, ...]
      - {"users": [{"username": "...", "password": "..."}, ...]}
      - {"users": {"<user>": "<pass>", ...}}
      - {"<user>": "<pass>", ...}
      - {"<user>": {"password": "<pass>"}, ...}
    Passwords can be plaintext or "sha256:<hex>".
    """

    def __init__(self, users_file: str | None = None):
        self.users_file = users_file
        self.sessions: dict[str, dict] = {}
        self.failed_attempts: dict[str, list[float]] = {}
        self._cached_users: dict[str, str] = {}
        self._cached_admins: set[str] = set()
        self._cached_groups: dict[str, set[str]] = {}
        self._cached_guilds: dict[str, set[str]] = {}
        self._cached_mtime: tuple | float = -1.0
        self._reported: set[str] = set()
        self._lock = threading.RLock()

    @property
    def enabled(self) -> bool:
        """Fail closed: authentication is on whenever a users-file path is configured, whether or
        not the file exists, parses or holds users. Only no path at all (library/dev use of
        make_app, or WEBUI_AUTH=off in bot.py) turns it off."""
        return bool(self.users_file)

    def _problem(self, kind: str, detail: str) -> None:
        """One ERROR per distinct condition, not per request; re-armed by the next good load."""
        if kind not in self._reported:
            self._reported.add(kind)
            log.error("web UI users file %s: %s; nobody can log in until it is fixed", self.users_file, detail)

    def load_users(self) -> dict[str, str]:
        if not self.enabled:
            return {}
        with self._lock:
            try:
                st = os.stat(self.users_file)
            except FileNotFoundError:
                self._problem("missing", "does not exist")
                return self._cached_users  # empty if it never loaded: nobody can log in
            except OSError as e:
                self._problem("unreadable", f"cannot be read ({type(e).__name__})")
                return self._cached_users
            stamp = (st.st_mtime_ns, st.st_size)
            if stamp == self._cached_mtime:
                return self._cached_users
            try:
                with open(self.users_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                records, _ = _split_raw(data)
            except (OSError, UnicodeDecodeError) as e:
                self._problem("unreadable", f"cannot be read ({type(e).__name__})")
                return self._cached_users
            except ValueError as e:  # json.JSONDecodeError is a ValueError
                self._problem("invalid", f"is not valid ({type(e).__name__})")
                return self._cached_users
            users: dict[str, str] = {}
            groups_map: dict[str, set[str]] = {}
            guilds_map: dict[str, set[str]] = {}
            for rec in records:
                if isinstance(rec, dict) and "username" in rec and "password" in rec:
                    uname = str(rec["username"]).strip()
                    if not uname:
                        continue
                    users[uname] = str(rec["password"])
                    groups_map[uname.lower()] = _groups_of(rec)
                    guilds_map[uname.lower()] = _guilds_of(rec)
            for u in users:
                u_lower = u.lower()
                # Legacy files: an account literally named "admin" with no group info was the admin.
                if u_lower == "admin" and not groups_map[u_lower]:
                    groups_map[u_lower] = {"admin"}
                if not groups_map.get(u_lower):
                    groups_map[u_lower] = {"user"}
            self._cached_users = users
            self._cached_groups = groups_map
            self._cached_guilds = guilds_map
            self._cached_admins = {u for u, grps in groups_map.items() if "admin" in grps}
            self._cached_mtime = stamp
            self._reported.clear()
            if not users:
                self._problem("empty", "contains no users")
            return users

    def get_user_groups(self, username: str | None) -> set[str]:
        if not self.enabled:
            return {"admin"}
        if not username:
            return set()
        self.load_users()
        return self._cached_groups.get(username.strip().lower(), set())

    def in_group(self, username: str | None, group: str) -> bool:
        if not self.enabled:
            return True
        if not username:
            return False
        return group.strip().lower() in self.get_user_groups(username)

    def user_exists(self, username: str | None) -> bool:
        if not username:
            return False
        self.load_users()
        return username.strip().lower() in self._cached_groups

    def is_admin(self, username: str | None) -> bool:
        return self.in_group(username, "admin")

    def allowed_guilds(self, username: str | None) -> set[str] | None:
        """None = every server (platform admin, or auth off). Otherwise the account's server ids
        (empty set = none; fail closed for unknown users)."""
        if not self.enabled or self.is_admin(username):
            return None
        if not username:
            return set()
        self.load_users()
        return set(self._cached_guilds.get(username.strip().lower(), set()))

    def list_users(self) -> list[dict]:
        self.load_users()
        result = []
        for uname in sorted(self._cached_users.keys()):
            grps = sorted(list(self.get_user_groups(uname)))
            result.append({"username": uname, "groups": grps,
                           "guilds": sorted(self._cached_guilds.get(uname.lower(), set()))})
        return result

    def _read_records(self):
        return _read_users_file(self.users_file)

    def _write_records(self, rebuild, records) -> None:
        for rec in records:  # persist the legacy "admin without groups" migration
            if _is_user_rec(rec, "admin") and not _groups_of(rec):
                rec["groups"] = ["admin"]
        atomic_write_json(rebuild(records), self.users_file)
        self._cached_mtime = -1.0
        self.load_users()

    def save_user(self, username: str, password: str | None, groups: list[str],
                  guilds: list[str] | None = None) -> bool:
        """Create/update a user. `password` is PLAINTEXT here and is stored only as a scrypt hash;
        None/"" keeps the existing hash (a new user needs one). Callers enforce the password policy."""
        if not self.users_file:
            return False
        uname = username.strip()
        uname_lower = uname.lower()
        grps = [str(g).strip().lower() for g in groups if str(g).strip()] or ["user"]
        new_hash = hash_password(password) if password else None  # slow: outside the lock
        with self._lock:
            records, rebuild = self._read_records()
            for item in records:
                if _is_user_rec(item, uname_lower):
                    if new_hash:
                        item["password"] = new_hash
                    item.pop("role", None)
                    item.pop("is_admin", None)
                    item["groups"] = grps
                    if guilds is not None:
                        item["guilds"] = sorted(set(guilds))
                    break
            else:
                if not new_hash:
                    raise PasswordRequired("a password is required for a new user")
                new_rec = {"username": uname, "password": new_hash, "groups": grps}
                if guilds is not None:
                    new_rec["guilds"] = sorted(set(guilds))
                records.append(new_rec)
            self._write_records(rebuild, records)
        return True

    def delete_user(self, username: str) -> bool:
        if not self.users_file or not os.path.exists(self.users_file):
            return False
        uname_lower = username.strip().lower()
        if uname_lower == "admin":
            return False
        try:
            with self._lock:
                records, rebuild = self._read_records()
                kept = [r for r in records if not _is_user_rec(r, uname_lower)]
                if len(kept) == len(records):
                    return False
                self._write_records(rebuild, kept)
            return True
        except Exception as e:
            log.warning("failed to delete user %s: %s", username, e)
            return False

    def _upgrade_hash(self, uname_lower: str, old_stored: str, password: str) -> None:
        """Replace a legacy plaintext/sha256 value by a scrypt hash after a successful login.
        Best effort: a failure is logged and never fails the login."""
        try:
            new_hash = hash_password(password)
            with self._lock:
                records, rebuild = self._read_records()
                for item in records:
                    if _is_user_rec(item, uname_lower) and str(item.get("password", "")) == old_stored:
                        item["password"] = new_hash
                        self._write_records(rebuild, records)
                        log.info("upgraded stored password of user %r to scrypt", uname_lower)
                        return
        except Exception as e:
            log.warning("could not upgrade stored password of user %r: %s", uname_lower, type(e).__name__)

    def verify_password(self, username: str, password: str) -> bool:
        users = self.load_users()
        uname_clean = username.strip().lower()
        matched_pass = None
        for u, p in users.items():
            if u.lower() == uname_clean:
                matched_pass = p
                break
        if matched_pass is None:
            verify_hash(_DUMMY_HASH, password)  # same cost as a real check; result discarded
            return False
        if password.strip().lower() in KNOWN_BAD_PASSWORDS:
            log.error("login refused for user %r: credential is on the known-default list", username)
            return False
        if matched_pass.startswith("scrypt$"):
            return verify_hash(matched_pass, password)
        if matched_pass.startswith("sha256:"):
            h = hashlib.sha256(password.encode("utf-8")).hexdigest()
            ok = hmac.compare_digest(matched_pass[7:].lower().encode("utf-8"), h.encode("ascii"))
        else:
            ok = hmac.compare_digest(matched_pass.encode("utf-8"), password.encode("utf-8"))
        if ok:
            self._upgrade_hash(uname_clean, matched_pass, password)
        return ok

    def check_rate_limit(self, ip: str) -> bool:
        now = time.monotonic()
        recent = [t for t in self.failed_attempts.get(ip, []) if now - t < LOGIN_RATE_WINDOW]
        self.failed_attempts[ip] = recent
        return len(recent) < MAX_LOGIN_FAILURES

    def record_failure(self, ip: str) -> None:
        self.failed_attempts.setdefault(ip, []).append(time.monotonic())

    def record_success(self, ip: str) -> None:
        self.failed_attempts.pop(ip, None)

    def create_session(self, username: str) -> str:
        now = time.time()
        expired = [sid for sid, s in self.sessions.items() if now > s["expires_at"]]
        for sid in expired:
            self.sessions.pop(sid, None)
        sid = secrets.token_urlsafe(32)
        self.sessions[sid] = {
            "username": username,
            "created_at": now,
            "expires_at": now + SESSION_LIFETIME,
        }
        return sid

    def get_session(self, session_id: str | None) -> str | None:
        if not session_id or session_id not in self.sessions:
            return None
        s = self.sessions[session_id]
        if time.time() > s["expires_at"]:
            self.sessions.pop(session_id, None)
            return None
        return s["username"]

    def revoke_session(self, session_id: str | None) -> None:
        if session_id:
            self.sessions.pop(session_id, None)


def _session_id_from(request) -> str | None:
    sid = request.cookies.get(SESSION_COOKIE) or request.cookies.get("session")
    if sid:
        return sid
    auth_hdr = request.headers.get("Authorization", "")
    if auth_hdr.startswith("Bearer "):
        return auth_hdr[7:].strip()
    return None


MAX_BODY = 16384  # 2000 non-ASCII chars JSON-escaped (\uXXXX) are ~12 KB
MAX_SAY_CHARS = 200
RETRY_INTERVAL = 15.0
ALLOWED_HOSTS_KEY = web.AppKey("allowed_hosts", set)
SAFE_METHODS = frozenset({"GET", "HEAD"})
# Bind policy: loopback, Tailscale CGNAT, Tailscale ULA. Nothing else may ever be listened on.
BIND_NETWORKS = tuple(ipaddress.ip_network(n) for n in (
    "127.0.0.0/8", "::1/128", "100.64.0.0/10", "fd7a:115c:a1e0::/48"))
CSP = ("default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; "
       "base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
STATIC_FILES = (
    ("/", "index.html", "text/html"),
    ("/app.js", "app.js", "application/javascript"),
    ("/app.css", "app.css", "text/css"),
)
DEFAULT_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


def _in_policy(ip) -> bool:
    return any(ip.version == n.version and ip in n for n in BIND_NETWORKS)


def _normalize_ip(entry: str) -> str | None:
    """Canonical text of `entry` if it is an IP literal inside the address policy, else None."""
    if "%" in entry:
        return None
    try:
        ip = ipaddress.ip_address(entry)
    except ValueError:
        return None
    return str(ip) if _in_policy(ip) else None


def peer_allowed(peername) -> bool:
    """Is this TCP peer (asyncio `peername`) loopback or Tailscale? Same ranges as the bind policy.
    IPv4-mapped IPv6 peers are unwrapped first; a missing/odd peername is refused."""
    try:
        host = str(peername[0]).split("%")[0]
        ip = ipaddress.ip_address(host)
    except (TypeError, IndexError, ValueError):
        return False
    if ip.version == 6 and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return _in_policy(ip)


def _peer_of(request):
    t = request.transport
    return t.get_extra_info("peername") if t is not None else None


def validate_bind_hosts(hosts: list[str]) -> list[str]:
    """Fail-closed bind policy. Returns normalized, de-duplicated addresses or raises ValueError."""
    out: list[str] = []
    for raw in hosts:
        entry = str(raw).strip()
        addr = "127.0.0.1" if entry.lower() == "localhost" else _normalize_ip(entry)
        if addr is None:
            raise ValueError(f"bind address {entry!r} is not loopback or a Tailscale address")
        if addr not in out:
            out.append(addr)
    if not out:
        raise ValueError("no bind addresses given")
    return out


def parse_tailscale_ip(text: str) -> list[str]:
    """Parse `tailscale ip` output (one address per line); keeps only policy-compliant ones."""
    out = []
    for line in text.splitlines():
        addr = _normalize_ip(line.strip())
        if addr and addr not in out:
            out.append(addr)
    return out


def parse_ip_addr(text: str) -> list[str]:
    """Parse `ip -o addr show dev tailscale0` output (inet/inet6 fields); policy-compliant only."""
    out = []
    parts = text.split()
    for i, word in enumerate(parts[:-1]):
        if word in ("inet", "inet6"):
            addr = _normalize_ip(parts[i + 1].split("/")[0])
            if addr and addr not in out:
                out.append(addr)
    return out


_last_run_error = ""
_detect_found: bool | None = None  # last detection outcome, so INFO logs fire only on change


def _run(cmd: list[str]) -> str | None:
    global _last_run_error
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3, check=True)
    except (OSError, subprocess.SubprocessError) as e:
        _last_run_error = f"{cmd[0]}: {type(e).__name__}"
        log.debug("%s failed: %s", cmd[0], e)
        return None
    return r.stdout


def detect_tailscale_addrs() -> list[str]:
    """Blocking. This host's Tailscale addresses, or [] (never raises). Logs at INFO only when the
    outcome changes (first call included), so periodic re-detection does not spam."""
    global _detect_found, _last_run_error
    addrs: list[str] = []
    _last_run_error = ""
    try:
        for cmd, parse in ((["tailscale", "ip"], parse_tailscale_ip),
                           (["ip", "-o", "addr", "show", "dev", "tailscale0"], parse_ip_addr)):
            out = _run(cmd)
            addrs = parse(out) if out else []
            if addrs:
                break
    except Exception as e:
        _last_run_error = f"{type(e).__name__}"
        log.debug("tailscale address detection failed", exc_info=True)
        addrs = []
    if bool(addrs) != _detect_found:
        _detect_found = bool(addrs)
        if addrs:
            log.info("Tailscale address(es) detected: %s", ", ".join(addrs))
        else:
            log.info("no Tailscale address detected%s", f" ({_last_run_error})" if _last_run_error else "")
    return addrs


def host_of(value: str) -> str | None:
    """Lowercased hostname from a Host header value ('h', 'h:port', '[v6]:port'); IPs canonicalized."""
    v = (value or "").strip().lower()
    if not v:
        return None
    if v.startswith("["):
        end = v.find("]")
        if end < 0:
            return None
        h = v[1:end]
    elif v.count(":") == 1:
        h = v.split(":")[0]
    else:
        h = v
    h = h.rstrip(".")
    try:
        return str(ipaddress.ip_address(h))
    except ValueError:
        return h or None


_HOSTNAME_RE = re.compile(r"^[a-z0-9]([a-z0-9._-]*[a-z0-9])?$")
_BRACKET_RE = re.compile(r"^\[([0-9a-fA-F:.]+)\](:\d{1,5})?$")


def _allowed_entry(item: str) -> str | None:
    """Canonical host for one WEBUI_ALLOWED_HOSTS entry, or None if it is not a plain host[:port]."""
    item = item.strip()
    if not item or any(c.isspace() for c in item) or "://" in item or any(c in item for c in "*/\\@?#"):
        return None
    m = _BRACKET_RE.match(item)
    if m:
        try:
            ip = ipaddress.ip_address(m.group(1))
        except ValueError:
            return None
        return str(ip) if ip.version == 6 else None
    if "[" in item or "]" in item or item.count(":") > 1:
        return None
    host, _, port = item.partition(":")
    if _ and not (port.isdigit() and len(port) <= 5):
        return None
    host = host.lower().rstrip(".")
    try:
        return str(ipaddress.ip_address(host))
    except ValueError:
        pass
    return host if _HOSTNAME_RE.match(host) else None


def parse_allowed_hosts(raw: str | None) -> list[str]:
    """WEBUI_ALLOWED_HOSTS: comma-separated hostnames (optional :port, ignored). URLs, paths,
    wildcards and malformed entries are dropped with a warning."""
    out = []
    for item in (raw or "").split(","):
        if not item.strip():
            continue
        h = _allowed_entry(item)
        if h is None:
            log.warning("ignoring invalid WEBUI_ALLOWED_HOSTS entry %r (plain hostnames only: no URL, path or wildcard)",
                        item.strip())
            continue
        out.append(h)
    return out


def sanitize_editor(value) -> str | None:
    if not isinstance(value, str):
        return None
    s = " ".join(value.split())
    s = "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")
    s = " ".join(s.split())[:40].strip()
    return s or None


def _s(v):
    return None if v is None else str(v)


def wire_snapshot(snap: dict) -> dict:
    """Snapshot with channel ids as strings (JS-safe)."""
    return {**snap, "guilds": {g: {**e, "channel_id": _s(e.get("channel_id")),
                                   "speed": e.get("speed", 1.0),
                                   "volume": e.get("volume", 1.0)}
                               for g, e in snap["guilds"].items()}}


def wire_event(ev: dict) -> dict:
    if ev.get("type") == "config":
        ch = dict(ev["change"])
        ch["guild_id"] = _s(ch["guild_id"])  # None (global change, e.g. voice) stays JSON null
        if ch.get("field") == "channel_id":
            ch["old"], ch["new"] = _s(ch["old"]), _s(ch["new"])
        return {**ev, "snapshot": wire_snapshot(ev["snapshot"]), "change": ch}
    return ev


def _json(data, status=200):
    return web.json_response(data, status=status)


def _make_shielded_callback(task_name: str, expected_exceptions: tuple):
    """Returns a done-callback for shielded tasks that logs unexpected exceptions.

    Use when a shielded task (cancelled mid-write) needs exception tracking that doesn't emit
    "Task exception was never retrieved" warnings for expected failures.
    Example:
        task = asyncio.create_task(coro)
        task.add_done_callback(_make_shielded_callback("my_task", (OSError, ConflictError)))
    """
    def callback(fut):
        if fut.cancelled():
            return
        exc = fut.exception()
        if exc and not isinstance(exc, expected_exceptions):
            log.error("%s failed after client disconnect", task_name, exc_info=exc)
    return callback


def _err(code, status):
    return _json({"error": code}, status)


def make_app(config, controller, *, allowed_hosts, status_interval: float = 3.0,
             keepalive_interval: float = 15.0, static_dir: str | None = None,
             users_file: str | None = None,
             features_file: str | None = None) -> web.Application:
    allowed = {"localhost", "127.0.0.1", "::1"}
    allowed.update(h for h in (host_of(x) for x in allowed_hosts) if h)
    streams: set[asyncio.Task] = set()
    peer_logged: dict[str, float] = {}
    static_dir = static_dir or DEFAULT_STATIC_DIR
    auth_mgr = AuthManager(users_file or None)
    auth_mgr.load_users()  # surface a missing/broken users file in the log at startup
    hash_slots = asyncio.Semaphore(4)  # concurrent scrypt operations (16 MiB each)
    features_path = features_file or os.getenv("FEATURES_FILE") or (
        os.path.join(os.path.dirname(config.path), "features.json") if hasattr(config, "path") and config.path else "features.json"
    )
    feature_store = FeatureStore(features_path)

    @web.middleware
    async def security_headers(request, handler):
        try:
            resp = await handler(request)
        except web.HTTPException as e:
            _apply_headers(request, e)
            raise
        _apply_headers(request, resp)
        return resp

    def _apply_headers(request, resp):
        resp.headers["Content-Security-Policy"] = CSP
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Referrer-Policy"] = "no-referrer"
        resp.headers["Cache-Control"] = "no-store" if request.path.startswith("/api/") else "no-cache"

    @web.middleware
    async def peer_guard(request, handler):
        # Runs first, on every route. A legitimate tailnet connection always has a tailnet source
        # address; a LAN host abusing the weak-host model arrives with its LAN source address.
        peer = _peer_of(request)
        if not peer_allowed(peer):
            ip = str(peer[0]) if peer else "?"
            now = time.monotonic()
            if now - peer_logged.get(ip, -1e9) > 60:
                if len(peer_logged) > 256:
                    peer_logged.clear()
                peer_logged[ip] = now
                log.warning("refused connection from non-loopback, non-Tailscale peer %s", ip)
            return _err("bad_peer", 403)
        return await handler(request)

    @web.middleware
    async def host_guard(request, handler):
        # DNS-rebinding defense: a page on evil.example resolving to our IP sends Host: evil.example.
        if host_of(request.headers.get("Host", "")) not in allowed:
            return _err("bad_host", 403)
        cross = request.headers.get("Sec-Fetch-Site", "").lower() == "cross-site"
        if cross and request.path.startswith("/api/"):
            return _err("bad_origin", 403)  # also GET: a no-cors cross-site fetch must not open SSE
        if request.method not in SAFE_METHODS:
            # Browsers always send Origin on cross-origin writes; curl/scripts send none, so absent is OK.
            origin = request.headers.get("Origin")
            if origin is not None:
                try:
                    oh = host_of(urlsplit(origin).netloc.rpartition("@")[2]) if origin != "null" else None
                except ValueError:
                    oh = None
                if oh not in allowed:
                    return _err("bad_origin", 403)
            if cross:
                return _err("bad_origin", 403)
        return await handler(request)

    PUBLIC_PATHS = frozenset({"/", "/app.js", "/app.css", "/api/session", "/api/login", "/api/logout"})

    @web.middleware
    async def auth_guard(request, handler):
        if not auth_mgr.enabled:
            return await handler(request)
        if request.path in PUBLIC_PATHS:
            return await handler(request)
        sid = _session_id_from(request)
        user = auth_mgr.get_session(sid)
        if user and not auth_mgr.user_exists(user):  # removed from the file: the session dies with it
            auth_mgr.revoke_session(sid)
            user = None
        if not user:
            return _err("unauthorized", 401)
        request[USER_KEY] = user
        return await handler(request)

    app = web.Application(middlewares=[security_headers, peer_guard, host_guard, auth_guard], client_max_size=MAX_BODY)

    # --- static ---
    def static_handler(name, ctype):
        try:
            with open(os.path.join(static_dir, name), "rb") as f:
                body = f.read()
        except OSError:
            log.error("static file %s missing", name)
            body = None

        async def h(request):
            if body is None:
                return web.Response(status=500, text="static file missing")
            return web.Response(body=body, content_type=ctype, charset="utf-8")
        return h

    app[ALLOWED_HOSTS_KEY] = allowed  # mutable: start_webui adds late-detected Tailscale addresses

    for path, name, ctype in STATIC_FILES:
        app.router.add_get(path, static_handler(name, ctype))

    # --- helpers ---
    def scope_of(request):
        """None = all servers; else the set of server ids this session may touch."""
        return auth_mgr.allowed_guilds(request.get(USER_KEY))

    def require_platform_admin(request):
        if auth_mgr.enabled and not auth_mgr.is_admin(request.get(USER_KEY)):
            raise web.HTTPForbidden(text=json.dumps({"error": "forbidden"}), content_type="application/json")

    def scoped_snapshot(snap, scope):
        if scope is None:
            return snap
        return {**snap, "guilds": {g: e for g, e in snap["guilds"].items() if g in scope}}

    def scoped_guilds(guilds, scope):
        return guilds if scope is None else [g for g in guilds if g["id"] in scope]

    def known_guild(request):
        gid = request.match_info["gid"]
        if not (gid.isascii() and gid.isdigit()):
            raise web.HTTPNotFound()
        if gid not in {g["id"] for g in controller.guilds_view()}:
            raise web.HTTPNotFound()
        scope = scope_of(request)
        if scope is not None and gid not in scope:
            raise web.HTTPNotFound()  # same answer as an unknown server: no existence leak
        return int(gid)

    def editor_of(request):
        ed = sanitize_editor(unquote(request.headers.get("X-Editor") or ""))
        if ed is None:
            user = request.get(USER_KEY)
            if user:
                ed = sanitize_editor(user)
        if ed is None:
            raise web.HTTPBadRequest(text=json.dumps({"error": "editor_required"}),
                                     content_type="application/json")
        return ed

    async def body_of(request) -> dict:
        try:
            body = json.loads(await request.read())
        except ValueError:
            body = None
        if not isinstance(body, dict):
            raise web.HTTPBadRequest(text=json.dumps({"error": "bad_request"}),
                                     content_type="application/json")
        rev = body.get("rev")
        if rev is not None and (isinstance(rev, bool) or not isinstance(rev, int)):
            raise web.HTTPBadRequest(text=json.dumps({"error": "bad_request"}),
                                     content_type="application/json")
        return body

    async def write(coro):
        # shield: a client disconnect must not cancel a half-finished persist
        task = asyncio.create_task(coro)
        task.add_done_callback(_make_shielded_callback("config_write", (ConflictError, OSError)))
        try:
            snap = await asyncio.shield(task)
        except ConflictError as e:
            return _json({"error": "conflict", "snapshot": wire_snapshot(e.snapshot)}, 409)
        except OSError:
            log.exception("config persist failed")
            return _err("persist_failed", 500)
        return _json({"snapshot": wire_snapshot(snap)})

    def voice_summary():
        fn = getattr(controller, "voice_summary", None)
        return fn() if fn else None

    async def apply_voice(request):
        require_platform_admin(request)  # the voice/engine is global: affects every server
        editor = editor_of(request)
        body = await body_of(request)
        if "name" not in body:
            return _err("bad_request", 400)
        name = body["name"]
        if name is not None:
            if not isinstance(name, str):
                return _err("bad_request", 400)
            if VOICE_NAME_RE.fullmatch(name) is None:
                return _err("bad_voice", 400)
        extra = {}
        if "engine" in body:
            eng = body["engine"]
            if not isinstance(eng, str) or eng not in ("piper", "kokoro"):
                return _err("bad_engine", 400)
            extra["engine_name"] = eng
        task = asyncio.create_task(controller.apply_voice(name, body.get("rev"), editor, **extra))
        task.add_done_callback(_make_shielded_callback("apply_voice",
                                                       (ConflictError, BadVoiceError, VoiceLoadError, OSError)))
        try:
            # shield: a client disconnect must not cancel a half-finished engine swap
            snap = await asyncio.shield(task)
        except ConflictError as e:
            return _json({"error": "conflict", "snapshot": wire_snapshot(e.snapshot)}, 409)
        except BadVoiceError:
            return _err("bad_voice", 400)
        except VoiceLoadError:
            return _err("voice_load_failed", 422)
        except OSError:
            log.exception("config persist failed (voice)")
            return _err("persist_failed", 500)
        except Exception:
            log.exception("voice change failed")
            return _err("internal", 500)
        return _json({"snapshot": wire_snapshot(snap)})

    async def list_voices(request):
        eng = request.query.get("engine")
        if eng is None:
            return _json(await controller.list_voices())
        if eng not in ("piper", "kokoro"):
            return _err("bad_engine", 400)
        return _json(await controller.list_voices(eng))

    async def say(request):
        gid = known_guild(request)
        editor = editor_of(request)
        body = await body_of(request)
        text = body.get("text")
        if not isinstance(text, str) or not text.strip():
            return _err("bad_request", 400)
        if len(text) > MAX_SAY_CHARS:
            return _err("too_long", 400)
        voice = body.get("voice")
        if voice is not None:
            if not isinstance(voice, str):
                return _err("bad_request", 400)
            voice = voice.strip()
            if not voice:
                voice = None
            elif VOICE_NAME_RE.fullmatch(voice) is None:
                return _err("bad_voice", 400)
        try:
            try:
                return _json(controller.say(gid, text, editor=editor, voice=voice))
            except TypeError:
                try:
                    return _json(controller.say(gid, text, editor=editor))
                except TypeError:
                    return _json(controller.say(gid, text))
        except SayError as e:
            if e.code == "bad_text":
                return _err("bad_request", 400)
            return _err(e.code, 409)

    MESSAGE_STATUS = {"bad_request": 400, "bad_mention": 400, "missing_permission": 403, "no_channel": 409,
                      "rate_limited": 429, "discord_error": 502}

    def message_error(e):
        status = MESSAGE_STATUS.get(e.code)
        if status is None:
            return _err("internal", 500)
        if e.code == "rate_limited":
            return _json({"error": "rate_limited", "retry_after": int(e.retry_after or 1)}, 429)
        return _err(e.code, status)

    async def members(request):
        gid = known_guild(request)
        q = request.query.get("q", "")[:100]
        cid_raw = request.query.get("channel_id")
        cid = int(cid_raw) if cid_raw and cid_raw.isdigit() else None
        kwargs = {"channel_id": cid} if cid is not None else {}
        try:
            return _json(await controller.members_for(gid, q, **kwargs))
        except MessageError as e:
            return message_error(e)
        except Exception:
            log.exception("member lookup failed")
            return _err("internal", 500)

    async def roles(request):
        gid = known_guild(request)
        q = request.query.get("q", "")[:100]
        cid_raw = request.query.get("channel_id")
        cid = int(cid_raw) if cid_raw and cid_raw.isdigit() else None
        kwargs = {"channel_id": cid} if cid is not None else {}
        try:
            fn = getattr(controller, "roles_for", None)
            if not fn:
                return _json({"roles": [], "truncated": False})
            res = fn(gid, q, **kwargs)
            if asyncio.iscoroutine(res):
                res = await res
            return _json(res)
        except MessageError as e:
            return message_error(e)
        except Exception:
            log.exception("role lookup failed")
            return _err("internal", 500)

    async def send_message(request):
        gid = known_guild(request)
        editor = editor_of(request)
        body = await body_of(request)
        content = body.get("content")
        cid_val = body.get("channel_id")
        if cid_val is not None:
            if not (isinstance(cid_val, (int, str)) and str(cid_val).isascii() and str(cid_val).isdigit()):
                return _err("bad_request", 400)
            cid = int(cid_val)
        else:
            cid = None
        if not isinstance(content, str) or not 1 <= len(content.strip()) <= MAX_MESSAGE_CHARS:
            return _err("bad_request", 400)
        kwargs = {"channel_id": cid} if cid is not None else {}
        task = asyncio.create_task(controller.send_message(gid, content, editor, **kwargs))
        task.add_done_callback(_make_shielded_callback("send_message", (MessageError,)))
        try:
            # shield: a client disconnect must not cancel the send (nor invite a retry-duplicate)
            return _json(await asyncio.shield(task))
        except MessageError as e:
            return message_error(e)
        except Exception:
            log.exception("message send failed")
            return _err("internal", 500)

    # --- API ---
    async def state(request):
        scope = scope_of(request)
        return _json({"config": wire_snapshot(scoped_snapshot(config.snapshot(), scope)),
                      "guilds": scoped_guilds(controller.guilds_view(), scope)})

    async def put_channel(request):
        gid = known_guild(request)
        editor = editor_of(request)
        body = await body_of(request)
        ch = body.get("channel_id")
        if ch is None:
            cid = None
        elif isinstance(ch, str) and ch.isascii() and ch.isdigit():
            cid = int(ch)
            if not controller.is_text_channel(gid, cid):
                return _err("bad_channel", 400)
        else:
            return _err("bad_channel", 400)
        return await write(config.set_channel(gid, cid, expected_rev=body.get("rev"), editor=editor))

    async def put_enabled(request):
        gid = known_guild(request)
        editor = editor_of(request)
        body = await body_of(request)
        enabled = body.get("enabled")
        if not isinstance(enabled, bool):
            return _err("bad_request", 400)
        return await write(config.set_enabled(gid, enabled, expected_rev=body.get("rev"), editor=editor))

    async def put_speed(request):
        gid = known_guild(request)
        editor = editor_of(request)
        body = await body_of(request)
        speed = body.get("speed")
        if not isinstance(speed, (int, float)) or isinstance(speed, bool):
            return _err("bad_request", 400)
        speed_val = float(speed)
        if not (0.5 <= speed_val <= 2.0):
            return _err("bad_request", 400)
        return await write(config.set_speed(gid, speed_val, expected_rev=body.get("rev"), editor=editor))

    async def put_volume(request):
        gid = known_guild(request)
        editor = editor_of(request)
        body = await body_of(request)
        volume = body.get("volume")
        if not isinstance(volume, (int, float)) or isinstance(volume, bool):
            return _err("bad_request", 400)
        vol_val = float(volume)
        if not (0.1 <= vol_val <= 2.0):
            return _err("bad_request", 400)
        return await write(config.set_volume(gid, vol_val, expected_rev=body.get("rev"), editor=editor))

    async def get_watched_users(request):
        gid = known_guild(request)
        fn = getattr(controller, "watched_users", None)
        users = fn(gid) if fn else []
        return _json({"users": users})

    async def put_user_voice(request):
        gid = known_guild(request)
        uid_str = request.match_info.get("uid", "")
        if not (uid_str.isascii() and uid_str.isdigit()):
            return _err("bad_request", 400)
        uid = int(uid_str)
        body = await body_of(request)
        voice = body.get("voice")
        if voice is not None:
            if not isinstance(voice, str):
                return _err("bad_request", 400)
            voice = voice.strip()
            if not voice:
                voice = None
            elif VOICE_NAME_RE.fullmatch(voice) is None:
                return _err("bad_voice", 400)
        fn = getattr(controller, "set_user_voice", None)
        if not fn:
            return _err("internal", 500)
        try:
            res = fn(gid, uid, voice)
            if asyncio.iscoroutine(res):
                await res
        except BadVoiceError:
            return _err("bad_voice", 400)
        except Exception:
            log.exception("setting user voice failed")
            return _err("internal", 500)
        return _json({"ok": True, "guild_id": str(gid), "user_id": str(uid), "voice": voice})

    async def skip(request):
        gid = known_guild(request)
        editor_of(request)
        return _json({"skipped": bool(controller.skip(gid))})

    async def clear(request):
        gid = known_guild(request)
        editor_of(request)
        return _json({"cleared": int(controller.clear_queue(gid))})

    async def events(request):
        resp = web.StreamResponse(headers={
            "Content-Type": "text/event-stream; charset=utf-8",
            "X-Accel-Buffering": "no",
        })
        _apply_headers(request, resp)  # must precede prepare(); middleware re-sets harmlessly
        await resp.prepare(request)
        q = config.hub.subscribe()
        streams.add(asyncio.current_task())
        scope = scope_of(request)

        async def send(event, data):
            await resp.write(f"event: {event}\ndata: {json.dumps(data)}\n\n".encode())

        try:
            await send("hello", {"snapshot": wire_snapshot(scoped_snapshot(config.snapshot(), scope)),
                                 "guilds": scoped_guilds(controller.guilds_view(), scope),
                                 "voices": voice_summary()})
            loop = asyncio.get_running_loop()
            next_status = loop.time() + status_interval
            next_keep = loop.time() + keepalive_interval
            last_status = None
            while True:
                now = loop.time()
                timeout = max(0.0, min(next_status, next_keep) - now)
                try:
                    ev = await asyncio.wait_for(q.get(), timeout)
                except asyncio.TimeoutError:
                    ev = None
                if ev is not None:
                    if scope is not None and ev.get("type") == "config":
                        gid_ev = ev["change"].get("guild_id")
                        if gid_ev is not None and str(gid_ev) not in scope:
                            ev = None
                        else:
                            ev = {**ev, "snapshot": scoped_snapshot(ev["snapshot"], scope)}
                if ev is not None:
                    ev = wire_event(ev)
                    await send(ev.get("type", "config"), ev)
                now = loop.time()
                if now >= next_status:
                    next_status = now + status_interval
                    st = controller.status_view()
                    if scope is not None:
                        st = {g: v for g, v in st.items() if g in scope}
                    if st != last_status:
                        last_status = st
                        await send("status", st)
                if now >= next_keep:
                    next_keep = now + keepalive_interval
                    await resp.write(b": keepalive\n\n")
        except (ConnectionResetError, ConnectionError):
            pass
        except Exception:
            log.exception("SSE stream failed")
        finally:
            config.hub.unsubscribe(q)
            streams.discard(asyncio.current_task())
        return resp

    async def session_status(request):
        if not auth_mgr.enabled:
            return _json({"authenticated": True, "auth_required": False, "username": "admin", "is_admin": True})
        sid = _session_id_from(request)
        user = auth_mgr.get_session(sid) if sid else None
        if user:
            return _json({
                "authenticated": True,
                "auth_required": True,
                "username": user,
                "is_admin": auth_mgr.is_admin(user),
            })
        return _json({"authenticated": False, "auth_required": True})

    async def login(request):
        if not auth_mgr.enabled:
            return _json({"ok": True, "auth_required": False, "username": "admin", "is_admin": True})
        peer = _peer_of(request)
        ip = str(peer[0]) if peer else "unknown"
        if not auth_mgr.check_rate_limit(ip):
            return _err("rate_limited", 429)
        body = await body_of(request)
        username = str(body.get("username", "")).strip()
        password = str(body.get("password", ""))
        if not username or not password:
            return _err("bad_request", 400)
        loop = asyncio.get_running_loop()
        async with hash_slots:  # scrypt is CPU/memory heavy: off the event loop, bounded
            ok = await loop.run_in_executor(None, auth_mgr.verify_password, username, password)
        if not ok:
            auth_mgr.record_failure(ip)
            log.warning("failed login attempt for user %r from %s", username, ip)
            return _err("invalid_credentials", 401)
        auth_mgr.record_success(ip)
        sid = auth_mgr.create_session(username)
        resp = _json({"ok": True, "username": username, "is_admin": auth_mgr.is_admin(username)})
        resp.set_cookie(SESSION_COOKIE, sid, max_age=int(SESSION_LIFETIME), path="/", httponly=True, samesite="Lax")
        return resp

    async def logout(request):
        sid = _session_id_from(request)
        if sid:
            auth_mgr.revoke_session(sid)
        resp = _json({"ok": True})
        resp.del_cookie(SESSION_COOKIE, path="/")
        resp.del_cookie("session", path="/")
        return resp

    async def get_features(request):
        return _json({"features": feature_store.get_features()})

    async def post_feature(request):
        body = await body_of(request)
        name = body.get("name")
        desc = body.get("description")
        if not isinstance(name, str) or not isinstance(desc, str):
            return _err("bad_request", 400)
        user = request.get(USER_KEY) or editor_of(request) or "anonymous"
        try:
            feat = await feature_store.add_feature(name, desc, user)
            return _json({"ok": True, "feature": feat}, 201)
        except ValueError:
            return _err("bad_request", 400)
        except Exception:
            log.exception("failed to add feature")
            return _err("internal", 500)

    async def delete_feature(request):
        user = request.get(USER_KEY) or editor_of(request)
        if auth_mgr.enabled and not auth_mgr.is_admin(user):
            return _err("forbidden", 403)
        fid = request.match_info.get("fid", "")
        if not fid:
            return _err("bad_request", 400)
        try:
            ok = await feature_store.delete_feature(fid)
            if not ok:
                return _err("not_found", 404)
            return _json({"ok": True})
        except Exception:
            log.exception("failed to delete feature")
            return _err("internal", 500)

    async def get_admin_users(request):
        user = request.get(USER_KEY) or editor_of(request)
        if auth_mgr.enabled and not auth_mgr.is_admin(user):
            return _err("forbidden", 403)
        return _json({"users": auth_mgr.list_users()})

    async def save_admin_user(request):
        user = request.get(USER_KEY) or editor_of(request)
        if auth_mgr.enabled and not auth_mgr.is_admin(user):
            return _err("forbidden", 403)
        body = await body_of(request)
        uname = str(body.get("username", "")).strip()
        passw = body.get("password")
        if passw is not None and not isinstance(passw, str):
            return _err("bad_request", 400)
        if passw == "":
            passw = None
        if passw is not None and password_policy_error(passw):
            return _err("weak_password", 400)
        raw_groups = body.get("groups", ["user"])
        if isinstance(raw_groups, str):
            raw_groups = [g.strip() for g in raw_groups.split(",") if g.strip()]
        elif not isinstance(raw_groups, list):
            raw_groups = [str(raw_groups)]
        groups = [str(g).strip().lower() for g in raw_groups if str(g).strip()]
        raw_guilds = body.get("guilds")
        guilds = None
        if raw_guilds is not None:
            if not isinstance(raw_guilds, list) or len(raw_guilds) > 200 or not all(
                    isinstance(g, str) and g.isascii() and g.isdigit() and len(g) <= 25 for g in raw_guilds):
                return _err("bad_request", 400)
            guilds = list(raw_guilds)
        if not (1 <= len(uname) <= 40) or not re.match(r"^[A-Za-z0-9_.-]+$", uname):
            return _err("bad_request", 400)
        try:
            loop = asyncio.get_running_loop()
            async with hash_slots:
                await loop.run_in_executor(None, auth_mgr.save_user, uname, passw, groups, guilds)
            return _json({"ok": True, "username": uname, "groups": groups, "guilds": guilds or []})
        except PasswordRequired:
            return _err("password_required", 400)
        except Exception:
            log.exception("failed to save user")
            return _err("internal", 500)

    async def delete_admin_user(request):
        user = request.get(USER_KEY) or editor_of(request)
        if auth_mgr.enabled and not auth_mgr.is_admin(user):
            return _err("forbidden", 403)
        target = request.match_info.get("uname", "").strip()
        if not target:
            return _err("bad_request", 400)
        if user and target.lower() == str(user).lower():
            return _err("cannot_delete_self", 400)
        if target.lower() == "admin":
            return _err("cannot_delete_admin", 400)
        try:
            loop = asyncio.get_running_loop()
            ok = await loop.run_in_executor(None, auth_mgr.delete_user, target)
            if not ok:
                return _err("not_found", 404)
            return _json({"ok": True})
        except Exception:
            log.exception("failed to delete user")
            return _err("internal", 500)

    async def end_streams(app):
        for t in list(streams):
            t.cancel()

    app.on_shutdown.append(end_streams)
    app[AUTH_MANAGER_KEY] = auth_mgr
    app[FEATURE_STORE_KEY] = feature_store
    app.router.add_get("/api/session", session_status)
    app.router.add_post("/api/login", login)
    app.router.add_post("/api/logout", logout)
    app.router.add_get("/api/features", get_features)
    app.router.add_post("/api/features", post_feature)
    app.router.add_delete("/api/features/{fid}", delete_feature)
    app.router.add_get("/api/admin/users", get_admin_users)
    app.router.add_post("/api/admin/users", save_admin_user)
    app.router.add_delete("/api/admin/users/{uname}", delete_admin_user)
    app.router.add_get("/api/state", state)
    app.router.add_put("/api/guilds/{gid}/channel", put_channel)
    app.router.add_put("/api/guilds/{gid}/enabled", put_enabled)
    app.router.add_put("/api/guilds/{gid}/speed", put_speed)
    app.router.add_put("/api/guilds/{gid}/volume", put_volume)
    app.router.add_get("/api/guilds/{gid}/watched", get_watched_users)
    app.router.add_put("/api/guilds/{gid}/watched/{uid}/voice", put_user_voice)
    app.router.add_post("/api/guilds/{gid}/skip", skip)
    app.router.add_post("/api/guilds/{gid}/clear", clear)
    app.router.add_get("/api/voices", list_voices)
    app.router.add_put("/api/voice", apply_voice)
    app.router.add_post("/api/guilds/{gid}/say", say)
    app.router.add_get("/api/guilds/{gid}/members", members)
    app.router.add_get("/api/guilds/{gid}/roles", roles)
    app.router.add_post("/api/guilds/{gid}/send", send_message)
    app.router.add_get("/api/events", events)
    return app


class WebUIRunner(web.AppRunner):
    """AppRunner whose cleanup also stops the bind-retry task."""

    retry_task: asyncio.Task | None = None

    async def cleanup(self) -> None:
        t, self.retry_task = self.retry_task, None
        try:
            if t:
                t.cancel()
                await asyncio.wait({t})
                if not t.cancelled():
                    t.exception()  # mark retrieved; the loop already logged it
        finally:
            await super().cleanup()


async def _bind_site(runner: web.AppRunner, host: str, port: int) -> None:
    site = web.TCPSite(runner, host, port)
    try:
        await site.start()
    except BaseException:
        await site.stop()  # unregister the half-started site
        raise


async def _retry_binds(runner, pending: list[str], port: int, interval: float,
                       detect, known: set[str], tailnet_found: bool) -> None:
    """Retry failed binds; in auto mode (`detect` given) also re-detect Tailscale addresses until
    one is found. Only cancellation ends the loop early; other errors are logged and retried."""
    while pending or (detect is not None and not tailnet_found):
        await asyncio.sleep(interval)
        try:
            if detect is not None and not tailnet_found:
                for addr in await asyncio.to_thread(detect):
                    addr = _normalize_ip(addr)
                    if addr and addr not in known:
                        known.add(addr)
                        pending.append(addr)
                        tailnet_found = True
                        runner.app[ALLOWED_HOSTS_KEY].add(host_of(addr))
            for host in list(pending):
                try:
                    await _bind_site(runner, host, port)
                except OSError as e:
                    log.debug("web UI still cannot bind %s:%s (%s)", host, port, e)
                else:
                    pending.remove(host)
                    log.info("web UI now listening on %s:%s", host, port)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("web UI bind retry failed; will try again")


async def start_webui(app: web.Application, hosts: list[str], port: int, *,
                      retry_interval: float = RETRY_INTERVAL, detect=None) -> WebUIRunner:
    """Bind every policy-compliant address independently; retry failed ones in the background.
    `detect` (auto mode only): blocking callable returning Tailscale addresses, re-run every
    `retry_interval` until one is found. Raises ValueError (policy) or OSError (nothing bound)."""
    hosts = validate_bind_hosts(hosts)
    runner = WebUIRunner(app, shutdown_timeout=5.0)
    await runner.setup()
    pending: list[str] = []
    last: OSError | None = None
    try:
        for host in hosts:
            try:
                await _bind_site(runner, host, port)
            except OSError as e:
                last = e
                pending.append(host)
                log.warning("web UI cannot bind %s:%s (%s); will retry", host, port, e)
            else:
                log.info("web UI listening on %s:%s", host, port)
        if len(pending) == len(hosts):
            raise OSError(f"web UI could not bind any of {hosts} on port {port}") from last
    except BaseException:
        await runner.cleanup()
        raise
    tailnet_found = any(not ipaddress.ip_address(h).is_loopback for h in hosts)
    if pending or (detect is not None and not tailnet_found):
        runner.retry_task = asyncio.create_task(
            _retry_binds(runner, pending, port, retry_interval, detect, set(hosts), tailnet_found),
            name="webui-bind-retry")
    return runner


# ---------- user bootstrap CLI ----------
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,40}$")


def main_cli(argv=None, *, getpass_fn=getpass.getpass, environ=None, stdout=None, stderr=None) -> int:
    """`python webui.py add-user|list-users|remove-user ...` -- manage the web UI users file.
    Passwords are only ever read from an interactive prompt (never argv or the environment)."""
    out = stdout if stdout is not None else sys.stdout
    err = stderr if stderr is not None else sys.stderr
    env = os.environ if environ is None else environ

    def say(msg, stream=out):
        print(msg, file=stream)

    parser = argparse.ArgumentParser(prog="webui.py", description="Manage web UI users.")
    parser.add_argument("--users-file", help="users file (default: $USERS_FILE)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_add = sub.add_parser("add-user", help="create a user or change a user's password")
    p_add.add_argument("username")
    p_add.add_argument("--admin", action="store_true", help="put the user in the admin group")
    sub.add_parser("list-users", help="list user names and groups")
    p_rm = sub.add_parser("remove-user", help="delete a user")
    p_rm.add_argument("username")
    # global option is accepted before or after the sub-command
    for p in (p_add, p_rm, sub.choices["list-users"]):
        p.add_argument("--users-file", dest="users_file_sub", default=None, help=argparse.SUPPRESS)
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        return int(e.code or 0) if isinstance(e.code, int) else 2

    path = args.users_file_sub or args.users_file or env.get("USERS_FILE") or ""
    if not path:
        say("error: no users file: pass --users-file PATH or set USERS_FILE", err)
        return 2
    try:
        records, rebuild = _read_users_file(path)
    except (OSError, ValueError) as e:
        say(f"error: cannot use {path}: {type(e).__name__}; refusing to overwrite it", err)
        return 1

    if args.cmd == "list-users":
        if not os.path.exists(path):
            say(f"error: {path} does not exist", err)
            return 1
        for rec in records:
            if isinstance(rec, dict) and rec.get("username"):
                groups = sorted(_groups_of(rec)) or ["user"]
                say(f"{rec['username']}\t{','.join(groups)}")
        return 0

    name = args.username.strip()
    lower = name.lower()
    if not USERNAME_RE.match(name):
        say("error: username must be 1-40 characters of A-Z a-z 0-9 _ . -", err)
        return 1

    if args.cmd == "remove-user":
        kept = [r for r in records if not _is_user_rec(r, lower)]
        if len(kept) == len(records):
            say(f"error: no such user: {name}", err)
            return 1
        atomic_write_json(rebuild(kept), path)
        say(f"removed {name}")
        return 0

    # add-user
    pw1 = getpass_fn("Password: ")
    pw2 = getpass_fn("Repeat password: ")
    if pw1 != pw2:
        say("error: passwords do not match", err)
        return 1
    if password_policy_error(pw1):
        say(f"error: password must be {MIN_PASSWORD_LEN} to {MAX_PASSWORD_LEN} characters and not a well-known default", err)
        return 1
    new_hash = hash_password(pw1)
    for rec in records:
        if _is_user_rec(rec, lower):
            groups = _groups_of(rec) or ({"admin"} if lower == "admin" else {"user"})
            if args.admin:
                groups.add("admin")
            rec.pop("role", None)
            rec.pop("is_admin", None)
            rec["groups"] = sorted(groups)
            rec["password"] = new_hash
            verb = "updated"
            break
    else:
        records.append({"username": name, "password": new_hash, "groups": ["admin"] if args.admin else ["user"]})
        verb = "created"
    try:
        atomic_write_json(rebuild(records), path)  # new files are created 0600
    except OSError as e:
        say(f"error: cannot write {path}: {type(e).__name__}", err)
        return 1
    say(f"{verb} {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main_cli())
