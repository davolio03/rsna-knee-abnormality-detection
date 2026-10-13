"""Local Sign in with ChatGPT profile management.

This module deliberately exposes profile metadata, never OAuth tokens. Token
transport and credential storage are private implementation details so callers
cannot accidentally log a credential by serializing a returned session.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import threading
import uuid
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Protocol
from urllib.parse import urlencode, urlparse, parse_qs


AUTHORIZATION_ENDPOINT = "https://auth.openai.com/api/accounts/authorize"
TOKEN_ENDPOINT = "https://auth.openai.com/api/accounts/oauth/token"
JWKS_ENDPOINT = "https://auth.openai.com/.well-known/jwks.json"
ISSUER = "https://auth.openai.com"
RESOURCE = "https://api.openai.com/v1"
PLAN_SCOPE = "chatgpt.tokens.use.direct"
SCOPES = "openid profile email offline_access resource.invoke " + PLAN_SCOPE
_KEYRING_SERVICE = "rsna-knee-siwc"
_SAFE_KEYRING_MODULES = {
    "keyring.backends.Windows",
    "keyring.backends.macOS",
    "keyring.backends.SecretService",
    "keyring.backends.kwallet",
    "keyring.backends.libsecret",
}


class AuthError(RuntimeError):
    """A safe authentication failure that contains no OAuth response data."""


class OAuthClient(Protocol):
    def exchange(self, *, client_id: str, code: str, verifier: str,
                 redirect_uri: str) -> Mapping[str, object]: ...

    def refresh(self, *, client_id: str,
                refresh_token: str) -> Mapping[str, object]: ...


class CredentialStore(Protocol):
    def save(self, profile_id: str, record: Mapping[str, object]) -> None: ...
    def load(self, profile_id: str) -> Mapping[str, object] | None: ...


@dataclass(frozen=True)
class AuthorizationRequest:
    authorization_url: str
    redirect_uri: str


@dataclass(frozen=True)
class Profile:
    profile_id: str
    client_id: str
    subject: str
    email: str | None
    plan_usage_enabled: bool
    expires_at: float


@dataclass
class _PendingAuthorization:
    state: str
    nonce: str
    verifier: str
    client_id: str
    existing_subject: str | None


class KeyringCredentialStore:
    """Store serialized secrets only in an explicitly supported OS keyring."""

    def __init__(self, *, keyring_module=None):
        if keyring_module is None:
            try:
                import keyring as keyring_module
            except ImportError:
                raise AuthError("OS credential storage is unavailable.") from None
        try:
            backend = keyring_module.get_keyring()
            module = type(backend).__module__
            priority = float(getattr(backend, "priority", 0))
        except Exception:
            raise AuthError("OS credential storage is unavailable.") from None
        if module not in _SAFE_KEYRING_MODULES or priority <= 0:
            raise AuthError("A supported secure OS credential backend is required.")
        self._keyring = keyring_module

    def save(self, profile_id: str, record: Mapping[str, object]) -> None:
        try:
            payload = json.dumps(dict(record), separators=(",", ":"), sort_keys=True)
            self._keyring.set_password(_KEYRING_SERVICE, profile_id, payload)
        except Exception:
            raise AuthError("OAuth credentials could not be stored securely.") from None

    def load(self, profile_id: str) -> Mapping[str, object] | None:
        try:
            payload = self._keyring.get_password(_KEYRING_SERVICE, profile_id)
            if payload is None:
                return None
            record = json.loads(payload)
            if not isinstance(record, dict):
                raise ValueError
            return record
        except Exception:
            raise AuthError("OAuth credentials could not be read securely.") from None


def load_or_create_host_id(path: str | Path) -> str:
    """Persist the non-secret host ID separately from keyring-only tokens."""
    host_file = Path(path)
    try:
        host_file.parent.mkdir(parents=True, exist_ok=True)
        if host_file.is_symlink():
            raise OSError
        if host_file.exists():
            value = host_file.read_text(encoding="ascii").strip()
            parsed = uuid.UUID(value.removeprefix("urn:uuid:"))
            if not value.startswith("urn:uuid:"):
                raise ValueError
            return f"urn:uuid:{parsed}"
        value = f"urn:uuid:{uuid.uuid4()}"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
        descriptor = os.open(host_file, flags, 0o600)
        try:
            os.write(descriptor, value.encode("ascii"))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return value
    except FileExistsError:
        # A second local process won creation; use its stable ID.
        return load_or_create_host_id(host_file)
    except Exception:
        raise AuthError("The local host identifier could not be safely stored.") from None


class _AuthlibOAuthClient:
    """Authlib-backed public OAuth client; no client secret or API key is used."""

    @staticmethod
    def _session(client_id: str, redirect_uri: str | None = None):
        try:
            from authlib.integrations.requests_client import OAuth2Session
            return OAuth2Session(client_id=client_id, redirect_uri=redirect_uri,
                                 scope=SCOPES,
                                 token_endpoint_auth_method="none")
        except ImportError:
            raise AuthError("OAuth support is unavailable.") from None

    def exchange(self, *, client_id: str, code: str, verifier: str,
                 redirect_uri: str) -> Mapping[str, object]:
        try:
            return self._session(client_id, redirect_uri).fetch_token(
                TOKEN_ENDPOINT, grant_type="authorization_code", code=code,
                code_verifier=verifier, redirect_uri=redirect_uri,
                resource=RESOURCE, include_client_id=True,
            )
        except Exception:
            raise AuthError("Authorization could not be completed.") from None

    def refresh(self, *, client_id: str,
                refresh_token: str) -> Mapping[str, object]:
        try:
            return self._session(client_id).refresh_token(
                TOKEN_ENDPOINT, refresh_token=refresh_token, resource=RESOURCE,
                include_client_id=True,
            )
        except Exception:
            raise AuthError("The saved authorization could not be refreshed.") from None


def _verify_openai_id_token(token: str, client_id: str,
                           nonce: str | None) -> Mapping[str, object]:
    """Verify a signed OpenAI ID token using Authlib and the published JWKS."""
    try:
        from authlib.jose import JsonWebToken
        from urllib.request import urlopen

        with urlopen(JWKS_ENDPOINT, timeout=10) as response:
            jwks = json.loads(response.read().decode("utf-8"))
        claim_rules = {
            "iss": {"essential": True, "value": ISSUER},
            "aud": {"essential": True, "value": client_id},
        }
        if nonce is not None:
            claim_rules["nonce"] = {"essential": True, "value": nonce}
        claims = JsonWebToken([
            "RS256", "RS384", "RS512", "PS256", "PS384", "PS512",
            "ES256", "ES384", "ES512",
        ]).decode(token, jwks, claims_options=claim_rules)
        claims.validate(now=int(time.time()))
        return dict(claims)
    except Exception:
        raise AuthError("The identity token could not be validated.") from None


class SIWCAuthenticator:
    """Implement Sign in with ChatGPT without exposing credentials to callers."""

    def __init__(self, *, store: CredentialStore, host_id: str,
                 redirect_uri: str, oauth: OAuthClient | None = None,
                 id_token_verifier: Callable[[str, str, str], Mapping[str, object]] | None = None,
                 clock: Callable[[], float] = time.time,
                 random_value: Callable[[int], bytes] = secrets.token_bytes,
                 agent_name: str = "RSNA Knee"):
        _validate_redirect_uri(redirect_uri)
        if not isinstance(host_id, str) or not host_id.startswith("urn:uuid:"):
            raise AuthError("A persistent host identifier is required.")
        self._store = store
        self._host_id = host_id
        self._redirect_uri = redirect_uri
        self._oauth = oauth or _AuthlibOAuthClient()
        self._verify_id_token = id_token_verifier or _verify_openai_id_token
        self._clock = clock
        self._random = random_value
        self._agent_name = agent_name
        self._pending: dict[str, _PendingAuthorization] = {}
        self._refresh_lock_guard = threading.Lock()
        self._refresh_locks: dict[str, threading.Lock] = {}

    def begin_sign_in(self, profile_id: str | None = None) -> AuthorizationRequest:
        record = self._store.load(profile_id) if profile_id is not None else None
        if profile_id is not None and record is None:
            raise AuthError("The selected account profile is unavailable.")
        existing = record is not None
        client_id = str(record["client_id"]) if existing else "dynamic_agent_client"
        verifier = _b64url(self._random(32))
        challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
        state = _b64url(self._random(32))
        nonce = _b64url(self._random(32))
        values = {
            "client_id": client_id,
            "ext_agent_host_id": self._host_id,
            "response_type": "code",
            "redirect_uri": self._redirect_uri,
            "scope": SCOPES,
            "resource": RESOURCE,
            "state": state,
            "nonce": nonce,
            "code_challenge_method": "S256",
            "code_challenge": challenge,
        }
        if existing:
            if record.get("id_token"):
                values["id_token_hint"] = str(record["id_token"])
            if record.get("email"):
                values["login_hint"] = str(record["email"])
            subject = str(record["subject"])
        else:
            values["agent_name_hint"] = self._agent_name
            subject = None
        self._pending[state] = _PendingAuthorization(
            state, nonce, verifier, client_id, subject,
        )
        return AuthorizationRequest(
            AUTHORIZATION_ENDPOINT + "?" + urlencode(values), self._redirect_uri,
        )

    def complete_sign_in(self, callback_url: str) -> Profile:
        state, query = _callback_query(callback_url, self._redirect_uri)
        pending = self._pending.pop(state, None)
        if pending is None:
            raise AuthError("The authorization callback did not match a pending sign-in.")
        if "error" in query:
            raise AuthError("Sign-in was declined or could not be completed.")
        code = _one(query, "code")
        if not code:
            raise AuthError("The authorization callback was incomplete.")
        returned_client = _one(query, "client_id")
        if pending.client_id == "dynamic_agent_client":
            if not returned_client or returned_client == "dynamic_agent_client":
                raise AuthError("Client registration was incomplete.")
            client_id = returned_client
        else:
            if returned_client and returned_client != pending.client_id:
                raise AuthError("The callback did not match the selected account.")
            client_id = pending.client_id
        tokens = self._oauth.exchange(
            client_id=client_id, code=code, verifier=pending.verifier,
            redirect_uri=self._redirect_uri,
        )
        profile = self._validated_profile(tokens, client_id, pending.nonce,
                                          pending.existing_subject)
        record = _record_from_tokens(tokens, client_id, profile.subject,
                                     profile.email, self._clock())
        # Preserve the validated token verbatim as the next optional OIDC hint.
        self._store.save(profile.profile_id, record)
        return profile

    def refresh_if_needed(self, profile_id: str) -> Profile:
        with self._refresh_lock_guard:
            profile_lock = self._refresh_locks.setdefault(profile_id, threading.Lock())
        with profile_lock:
            return self._refresh_if_needed_locked(profile_id)

    def _refresh_if_needed_locked(self, profile_id: str) -> Profile:
        record = self._store.load(profile_id)
        if record is None:
            raise AuthError("The selected account profile is unavailable.")
        if float(record.get("expires_at", 0)) > self._clock() + 30:
            return _profile(record)
        refresh_token = record.get("refresh_token")
        if not isinstance(refresh_token, str) or not refresh_token:
            raise AuthError("The saved authorization must be renewed.")
        tokens = dict(self._oauth.refresh(client_id=str(record["client_id"]),
                                          refresh_token=refresh_token))
        # OAuth refresh responses may omit an unchanged scope and may omit an
        # ID token. In that case retain the previously verified identity and
        # grants rather than treating an unchanged grant as a new grant.
        if "scope" not in tokens and "scopes" not in tokens:
            tokens["scope"] = record.get("scopes", [])
        if "id_token" in tokens:
            profile = self._validated_profile(tokens, str(record["client_id"]), None,
                                              str(record["subject"]))
            subject, email = profile.subject, profile.email
        else:
            if PLAN_SCOPE not in _granted_scopes(tokens):
                raise AuthError("ChatGPT plan usage permission was not granted.")
            expires_in = tokens.get("expires_in")
            if (not isinstance(tokens.get("access_token"), str)
                    or not tokens.get("access_token")
                    or not isinstance(tokens.get("refresh_token"), str)
                    or not tokens.get("refresh_token")
                    or isinstance(expires_in, bool)
                    or not isinstance(expires_in, (int, float)) or expires_in <= 0):
                raise AuthError("The refreshed authorization response was incomplete.")
            tokens["id_token"] = record["id_token"]
            subject, email = str(record["subject"]), record.get("email")
            profile = Profile(profile_id, str(record["client_id"]), subject,
                              email if isinstance(email, str) else None, True,
                              self._clock() + float(expires_in))
        refreshed = _record_from_tokens(tokens, str(record["client_id"]),
                                        subject, email if isinstance(email, str) else None,
                                        self._clock())
        self._store.save(profile.profile_id, refreshed)
        return profile

    def _validated_profile(self, tokens: Mapping[str, object], client_id: str,
                           nonce: str | None, expected_subject: str | None) -> Profile:
        id_token = tokens.get("id_token")
        if not isinstance(id_token, str) or not id_token:
            raise AuthError("The identity token is missing.")
        try:
            claims = self._verify_id_token(id_token, client_id, nonce)
        except AuthError:
            raise
        except Exception:
            raise AuthError("The identity token could not be validated.") from None
        audience = claims.get("aud")
        audience_matches = (audience == client_id if isinstance(audience, str)
                            else isinstance(audience, (list, tuple)) and client_id in audience)
        if (claims.get("iss") != ISSUER or not audience_matches
                or (nonce is not None and claims.get("nonce") != nonce)
                or not isinstance(claims.get("exp"), (int, float))
                or float(claims["exp"]) <= self._clock()
                or isinstance(claims.get("iat"), bool)
                or not isinstance(claims.get("iat"), (int, float))
                or float(claims["iat"]) > self._clock() + 5
                or not isinstance(claims.get("sub"), str) or not claims.get("sub")):
            raise AuthError("The identity token could not be validated.")
        subject = str(claims["sub"])
        if expected_subject is not None and subject != expected_subject:
            raise AuthError("The sign-in identity did not match the selected account.")
        scopes = _granted_scopes(tokens)
        if PLAN_SCOPE not in scopes:
            raise AuthError("ChatGPT plan usage permission was not granted.")
        if not all(isinstance(tokens.get(field), str) and tokens.get(field)
                   for field in ("access_token", "refresh_token")):
            raise AuthError("The authorization response was incomplete.")
        expires_in = tokens.get("expires_in")
        if isinstance(expires_in, bool) or not isinstance(expires_in, (int, float)) or expires_in <= 0:
            raise AuthError("The authorization response was incomplete.")
        return Profile(client_id, client_id, subject,
                       claims.get("email") if isinstance(claims.get("email"), str) else None,
                       True, self._clock() + float(expires_in))


def _record_from_tokens(tokens: Mapping[str, object], client_id: str, subject: str,
                        email: str | None, now: float) -> dict[str, object]:
    expires_in = float(tokens["expires_in"])
    return {
        "client_id": client_id,
        "subject": subject,
        "email": email,
        "issuer": ISSUER,
        "access_token": tokens["access_token"],
        "refresh_token": tokens["refresh_token"],
        "id_token": tokens["id_token"],
        "token_type": str(tokens.get("token_type", "Bearer")),
        "scopes": sorted(_granted_scopes(tokens)),
        "expires_at": now + expires_in,
    }


def _profile(record: Mapping[str, object]) -> Profile:
    scopes = record.get("scopes", [])
    return Profile(str(record["client_id"]), str(record["client_id"]),
                   str(record["subject"]),
                   record.get("email") if isinstance(record.get("email"), str) else None,
                   isinstance(scopes, list) and PLAN_SCOPE in scopes,
                   float(record.get("expires_at", 0)))


def _granted_scopes(tokens: Mapping[str, object]) -> set[str]:
    value = tokens.get("scope", tokens.get("scopes", ""))
    if isinstance(value, str):
        return set(value.split())
    if isinstance(value, (list, tuple, set)) and all(isinstance(item, str) for item in value):
        return set(value)
    return set()


def _callback_query(callback_url: str, redirect_uri: str):
    _validate_redirect_uri(redirect_uri)
    try:
        callback = urlparse(callback_url)
        expected = urlparse(redirect_uri)
        if (callback.scheme != "http" or callback.hostname != "127.0.0.1"
                or callback.port != expected.port or callback.path != expected.path
                or callback.username is not None or callback.password is not None
                or callback.fragment):
            raise ValueError
        query = parse_qs(callback.query, keep_blank_values=True, strict_parsing=True)
        state = _one(query, "state")
        if not state:
            raise ValueError
        return state, query
    except Exception:
        raise AuthError("The authorization callback was invalid.") from None


def _validate_redirect_uri(uri: str) -> None:
    try:
        parsed = urlparse(uri)
        if (parsed.scheme != "http" or parsed.hostname != "127.0.0.1"
                or parsed.port is None or parsed.path != "/auth/callback"
                or parsed.username is not None or parsed.password is not None
                or parsed.query or parsed.fragment):
            raise ValueError
    except Exception:
        raise AuthError("A valid 127.0.0.1 callback URI is required.") from None


def _one(query: Mapping[str, list[str]], key: str) -> str | None:
    values = query.get(key)
    if not values:
        return None
    if len(values) != 1:
        raise AuthError("The authorization callback was invalid.")
    return values[0]


def _b64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")
