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
import re
import secrets
import ssl
import sys
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
_WINDOWS_CREDENTIAL_BLOB_LIMIT_BYTES = 2560
_SAFE_KEYRING_MODULES = {
    "keyring.backends.Windows",
    "keyring.backends.macOS",
    "keyring.backends.SecretService",
    "keyring.backends.kwallet",
    "keyring.backends.libsecret",
}
_SAFE_OAUTH_ERROR_CODES = {
    "access_denied", "invalid_client", "invalid_grant", "invalid_refresh_token",
    "invalid_request", "invalid_scope", "refresh_token_expired",
    "refresh_token_invalidated", "refresh_token_reused", "server_error",
    "temporarily_unavailable", "token_expired", "unauthorized_client",
    "unsupported_grant_type",
}
_SAFE_REQUEST_ID = re.compile(r"req_[A-Za-z0-9_-]{1,64}\Z")
_SAFE_EXCEPTION_TYPE_NAME = re.compile(r"\A[A-Za-z][A-Za-z0-9_]{0,63}\Z", re.ASCII)
_PERSISTED_CREDENTIAL_FIELDS = frozenset({
    "client_id", "subject", "email", "issuer", "refresh_token",
    "token_type", "scopes", "expires_at",
})
_SAFE_OAUTH_FAILURE_CATEGORIES = {
    "exchange_timeout", "exchange_connection_error", "exchange_tls_error",
    "exchange_invalid_json_response", "unknown_exchange_error",
}
_SAFE_OAUTH_EXCEPTION_FAMILIES = {"authlib", "requests", "urllib3", "stdlib", "other"}


class AuthError(RuntimeError):
    """A safe authentication failure that contains no OAuth response data."""

    def __init__(self, message: str, *, _diagnostics: Mapping[str, object] | None = None):
        super().__init__(message)
        # Diagnostics are private and fixed-schema; never put them in repr/message.
        self._diagnostics = dict(_diagnostics or {})

    def __str__(self) -> str:
        message = super().__str__()
        summary = _format_safe_auth_diagnostics(self._diagnostics)
        return f"{message} Diagnostics: {summary}" if summary else message


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
        self._backend_module = module

    def save(self, profile_id: str, record: Mapping[str, object]) -> None:
        payload = None
        try:
            safe_record = {
                key: value for key, value in record.items()
                if key in _PERSISTED_CREDENTIAL_FIELDS
            }
            payload = json.dumps(safe_record, separators=(",", ":"), sort_keys=True)
            self._keyring.set_password(_KEYRING_SERVICE, profile_id, payload)
        except Exception:
            diagnostics = {}
            if payload is not None and self._backend_module == "keyring.backends.Windows":
                blob_bytes = len(payload.encode("utf-16-le"))
                diagnostics = {
                    "windows_credential_blob_bytes": blob_bytes,
                    "windows_blob_limit_bytes": _WINDOWS_CREDENTIAL_BLOB_LIMIT_BYTES,
                    "exceeds_windows_blob_limit": (
                        blob_bytes > _WINDOWS_CREDENTIAL_BLOB_LIMIT_BYTES
                    ),
                }
            raise AuthError(
                "OAuth credentials could not be stored securely.",
                _diagnostics=diagnostics,
            ) from None

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
        except Exception as exc:
            raise AuthError(
                "Authorization could not be completed.",
                _diagnostics=_safe_oauth_failure_diagnostics(exc),
            ) from None

    def refresh(self, *, client_id: str,
                refresh_token: str) -> Mapping[str, object]:
        try:
            return self._session(client_id).refresh_token(
                TOKEN_ENDPOINT, refresh_token=refresh_token, resource=RESOURCE,
                scope=None,
            )
        except Exception:
            raise AuthError("The saved authorization could not be refreshed.") from None


def _safe_oauth_failure_diagnostics(error: Exception) -> dict[str, object]:
    """Extract fixed-schema OAuth diagnostics without retaining payload text."""
    response = getattr(error, "response", None)
    failure_category = _safe_oauth_failure_category(error)
    diagnostics: dict[str, object] = {"failure_category": failure_category}
    if failure_category == "unknown_exchange_error":
        diagnostics["exception_family"] = _safe_oauth_exception_family(error)
        exception_type = _safe_oauth_exception_type_name(error)
        if exception_type is not None:
            diagnostics["exception_type"] = exception_type
    if response is not None:
        status = getattr(response, "status_code", None)
        if isinstance(status, int) and not isinstance(status, bool) and 100 <= status <= 599:
            diagnostics["http_status"] = status

        headers = getattr(response, "headers", None)
        try:
            request_id = next(
                value for name, value in headers.items()
                if isinstance(name, str) and name.lower() == "x-request-id"
            )
        except (AttributeError, StopIteration, TypeError):
            request_id = None
        if isinstance(request_id, str) and _SAFE_REQUEST_ID.fullmatch(request_id):
            diagnostics["request_id"] = request_id

        body_unavailable = object()
        try:
            body = response.json()
        except Exception:
            body = body_unavailable
        if isinstance(body, dict):
            diagnostics["response_body_shape"] = "json_object"
            error_value = body.get("error")
            error_code = error_value.get("code") if isinstance(error_value, dict) else error_value
        elif isinstance(body, list):
            diagnostics["response_body_shape"] = "json_array"
            error_code = None
        elif body is None or isinstance(body, (str, int, float, bool)):
            diagnostics["response_body_shape"] = "json_scalar"
            error_code = None
        else:
            error_code = None
        if isinstance(error_code, str) and error_code in _SAFE_OAUTH_ERROR_CODES:
            diagnostics["provider_error_code"] = error_code

    if "provider_error_code" not in diagnostics:
        error_code = getattr(error, "error", None)
        if isinstance(error_code, str) and error_code in _SAFE_OAUTH_ERROR_CODES:
            diagnostics["provider_error_code"] = error_code
    return diagnostics


def _safe_oauth_failure_category(error: Exception) -> str:
    """Classify stable transport categories without exposing type names."""
    request_timeout_types: tuple[type[BaseException], ...] = ()
    request_tls_types: tuple[type[BaseException], ...] = ()
    request_connection_types: tuple[type[BaseException], ...] = ()
    request_json_error_types: tuple[type[BaseException], ...] = ()
    try:
        from requests.exceptions import (
            ConnectionError as RequestsConnectionError,
            SSLError as RequestsSSLError,
            Timeout as RequestsTimeout,
        )
        try:
            from requests.exceptions import JSONDecodeError as RequestsJSONDecodeError
            request_json_error_types = (RequestsJSONDecodeError,)
        except ImportError:
            pass
        request_timeout_types = (RequestsTimeout,)
        request_tls_types = (RequestsSSLError,)
        request_connection_types = (RequestsConnectionError,)
    except ImportError:
        pass

    if (isinstance(error, TimeoutError)
            or isinstance(error, request_timeout_types)):
        return "exchange_timeout"
    if (isinstance(error, ssl.SSLError)
            or isinstance(error, request_tls_types)):
        return "exchange_tls_error"
    if (isinstance(error, ConnectionError)
            or isinstance(error, request_connection_types)):
        return "exchange_connection_error"
    if isinstance(error, request_json_error_types):
        return "exchange_invalid_json_response"
    return "unknown_exchange_error"


def _safe_oauth_exception_family(error: Exception) -> str:
    """Return only a coarse, fixed family label for otherwise unknown errors."""
    module = type(error).__module__
    if isinstance(module, str):
        for family in ("authlib", "requests", "urllib3"):
            if module == family or module.startswith(family + "."):
                return family
        root = module.partition(".")[0]
        if root == "builtins" or root in sys.stdlib_module_names:
            return "stdlib"
    return "other"


def _safe_oauth_exception_type_name(error: Exception) -> str | None:
    """Return an exception class name only when it is a bounded ASCII identifier."""
    name = type(error).__name__
    return (name if type(name) is str
            and _SAFE_EXCEPTION_TYPE_NAME.fullmatch(name) else None)


def _format_safe_auth_diagnostics(diagnostics: Mapping[str, object]) -> str:
    """Render only known diagnostic fields that pass their strict validators."""
    fields: list[str] = []
    category = diagnostics.get("failure_category")
    if isinstance(category, str) and category in _SAFE_OAUTH_FAILURE_CATEGORIES:
        fields.append(f"failure category: {category}")
    exception_family = diagnostics.get("exception_family")
    if (category == "unknown_exchange_error"
            and isinstance(exception_family, str)
            and exception_family in _SAFE_OAUTH_EXCEPTION_FAMILIES):
        fields.append(f"exception family: {exception_family}")
    exception_type = diagnostics.get("exception_type")
    if (category == "unknown_exchange_error"
            and type(exception_type) is str
            and _SAFE_EXCEPTION_TYPE_NAME.fullmatch(exception_type)):
        fields.append(f"exception type: {exception_type}")

    status = diagnostics.get("http_status")
    if type(status) is int and 100 <= status <= 599:
        fields.append(f"HTTP status: {status}")

    error_code = diagnostics.get("provider_error_code")
    if isinstance(error_code, str) and error_code in _SAFE_OAUTH_ERROR_CODES:
        fields.append(f"provider error code: {error_code}")

    request_id = diagnostics.get("request_id")
    if isinstance(request_id, str) and _SAFE_REQUEST_ID.fullmatch(request_id):
        fields.append(f"request ID: {request_id}")

    body_shape = diagnostics.get("response_body_shape")
    if isinstance(body_shape, str) and body_shape in {"json_object", "json_array", "json_scalar"}:
        fields.append(f"response body shape: {body_shape}")

    blob_bytes = diagnostics.get("windows_credential_blob_bytes")
    blob_limit = diagnostics.get("windows_blob_limit_bytes")
    exceeds_limit = diagnostics.get("exceeds_windows_blob_limit")
    if (type(blob_bytes) is int and 0 <= blob_bytes <= sys.maxsize
            and type(blob_limit) is int
            and blob_limit == _WINDOWS_CREDENTIAL_BLOB_LIMIT_BYTES
            and type(exceeds_limit) is bool
            and exceeds_limit == (blob_bytes > blob_limit)):
        exceeds_text = "yes" if exceeds_limit else "no"
        fields.append(
            f"Windows credential blob: {blob_bytes} bytes; Windows blob limit: "
            f"{blob_limit} bytes; exceeds limit: {exceeds_text}"
        )

    return "; ".join(fields)


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
        self._access_tokens: dict[str, tuple[str, float, int]] = {}

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
        with self._profile_lock(profile.profile_id):
            self._store.save(profile.profile_id, record)
            self._cache_access_token(profile.profile_id,
                                     str(tokens["access_token"]),
                                     profile.expires_at)
        return profile

    def refresh_if_needed(self, profile_id: str) -> Profile:
        profile_lock = self._profile_lock(profile_id)
        with profile_lock:
            return self._refresh_if_needed_locked(profile_id)

    def _profile_lock(self, profile_id: str) -> threading.Lock:
        with self._refresh_lock_guard:
            return self._refresh_locks.setdefault(profile_id, threading.Lock())

    def with_plan_access_token(self, profile_id: str,
                               operation: Callable[[str], object]) -> object:
        """Run one operation with a refreshed plan-use token, without returning it.

        Callers receive only the operation result; the OAuth token remains inside
        this narrow credential boundary.
        """
        if not callable(operation):
            raise AuthError("An authorized operation is required.")
        request_started = time.monotonic_ns()
        with self._profile_lock(profile_id):
            self._refresh_if_needed_locked(
                profile_id, allow_cache_created_since=request_started
            )
            with self._refresh_lock_guard:
                cached = self._access_tokens.get(profile_id)
            if cached is None or cached[1] <= self._clock():
                raise AuthError("The selected account authorization is unavailable.")
            record = self._store.load(profile_id)
            if record is None:
                raise AuthError("The selected account profile is unavailable.")
            if not _profile(record).plan_usage_enabled:
                raise AuthError("ChatGPT plan usage permission was not granted.")
            token = cached[0]
        return operation(token)

    def _refresh_if_needed_locked(
            self, profile_id: str, *,
            allow_cache_created_since: float | None = None) -> Profile:
        record = self._store.load(profile_id)
        if record is None:
            raise AuthError("The selected account profile is unavailable.")
        with self._refresh_lock_guard:
            cached = self._access_tokens.get(profile_id)
        if (cached is not None and cached[1] > self._clock() + 30
                and float(record.get("expires_at", 0)) > self._clock() + 30):
            return _profile(record)
        # Overlapping callbacks may share a token refreshed after they began;
        # an older cached token still follows the 30-second refresh buffer.
        if (cached is not None and allow_cache_created_since is not None
                and cached[2] > allow_cache_created_since
                and cached[1] > self._clock()):
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
            subject, email = str(record["subject"]), record.get("email")
            profile = Profile(profile_id, str(record["client_id"]), subject,
                              email if isinstance(email, str) else None, True,
                              self._clock() + float(expires_in))
        refreshed = _record_from_tokens(tokens, str(record["client_id"]),
                                        subject, email if isinstance(email, str) else None,
                                        self._clock())
        self._store.save(profile.profile_id, refreshed)
        self._cache_access_token(profile.profile_id, str(tokens["access_token"]),
                                 profile.expires_at)
        return profile

    def _cache_access_token(self, profile_id: str, token: str,
                            expires_at: float) -> None:
        with self._refresh_lock_guard:
            self._access_tokens[profile_id] = (
                token, expires_at, time.monotonic_ns()
            )

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
        "refresh_token": tokens["refresh_token"],
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
