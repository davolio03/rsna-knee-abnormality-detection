import json
import hashlib
import sys
import threading
import tempfile
import time
import unittest
from datetime import datetime, timezone
from types import ModuleType
from unittest.mock import Mock
from urllib.parse import parse_qs, urlparse
from requests.exceptions import (
    ConnectionError as RequestsConnectionError,
    JSONDecodeError as RequestsJSONDecodeError,
    RequestException as RequestsException,
    SSLError as RequestsSSLError,
    Timeout as RequestsTimeout,
)

from rsna_knee.labels.siwc_auth import (
    PLAN_SCOPE,
    AuthError,
    KeyringCredentialStore,
    SIWCAuthenticator,
    _AuthlibOAuthClient,
    load_or_create_host_id,
)


class FakeOAuth:
    def __init__(self):
        self.authorization = None
        self.exchange_result = None
        self.refresh_result = None
        self.exchange_args = None
        self.refresh_args = None

    def exchange(self, *, client_id, code, verifier, redirect_uri):
        self.exchange_args = (client_id, code, verifier, redirect_uri)
        return dict(self.exchange_result)

    def refresh(self, *, client_id, refresh_token):
        self.refresh_args = (client_id, refresh_token)
        return dict(self.refresh_result)


class MemoryStore:
    def __init__(self):
        self.values = {}

    def save(self, profile_id, record):
        self.values[profile_id] = dict(record)

    def load(self, profile_id):
        value = self.values.get(profile_id)
        return None if value is None else dict(value)


class SIWCTest(unittest.TestCase):
    def setUp(self):
        self.oauth = FakeOAuth()
        self.store = MemoryStore()
        self.now = lambda: datetime(2026, 10, 13, tzinfo=timezone.utc).timestamp()
        self.auth = SIWCAuthenticator(
            oauth=self.oauth,
            store=self.store,
            id_token_verifier=lambda token, client_id, nonce: {
                "iss": "https://auth.openai.com", "sub": "subject-1",
                "aud": client_id, "nonce": nonce, "exp": self.now() + 3600,
                "iat": self.now(),
                "email": "synthetic@example.test",
            },
            clock=self.now,
            host_id="urn:uuid:synthetic-host",
            redirect_uri="http://127.0.0.1:1455/auth/callback",
            random_value=lambda n: (b"A" * n),
        )

    def complete(self, request, *, client_id="oaiapp-synthetic", scopes=None):
        self.oauth.exchange_result = {
            "access_token": "access-secret", "refresh_token": "refresh-secret",
            "id_token": "id-secret", "token_type": "Bearer", "expires_in": 3600,
            "scope": " ".join(scopes or ["openid", "profile", "email", "offline_access", "resource.invoke", PLAN_SCOPE]),
        }
        query = parse_qs(urlparse(request.authorization_url).query)
        callback = ("http://127.0.0.1:1455/auth/callback?code=synthetic-code"
                    f"&state={query['state'][0]}&client_id={client_id}")
        return self.auth.complete_sign_in(callback)

    def test_new_registration_uses_dynamic_client_and_required_oauth_parameters(self):
        request = self.auth.begin_sign_in()
        query = parse_qs(urlparse(request.authorization_url).query)
        self.assertEqual(query["client_id"], ["dynamic_agent_client"])
        self.assertEqual(query["ext_agent_host_id"], ["urn:uuid:synthetic-host"])
        self.assertEqual(query["response_type"], ["code"])
        self.assertEqual(query["code_challenge_method"], ["S256"])
        expected_verifier = __import__("base64").urlsafe_b64encode(b"A" * 32).rstrip(b"=").decode("ascii")
        expected_challenge = __import__("base64").urlsafe_b64encode(
            hashlib.sha256(expected_verifier.encode("ascii")).digest()
        ).rstrip(b"=").decode("ascii")
        self.assertEqual(query["code_challenge"], [expected_challenge])
        self.assertTrue(query["state"][0])
        self.assertTrue(query["nonce"][0])
        self.assertIn(PLAN_SCOPE, query["scope"][0].split())
        self.assertEqual(query["resource"], ["https://api.openai.com/v1"])

    def test_authorization_code_exchange_uses_issued_client_and_same_redirect(self):
        request = self.auth.begin_sign_in()
        self.complete(request, client_id="oaiapp-issued")
        self.assertEqual(self.oauth.exchange_args, (
            "oaiapp-issued", "synthetic-code",
            __import__("base64").urlsafe_b64encode(b"A" * 32).rstrip(b"=").decode("ascii"),
            "http://127.0.0.1:1455/auth/callback",
        ))

    def test_exchange_failure_exposes_only_sanitized_provider_diagnostics(self):
        response = Mock(
            status_code=400,
            headers={"x-request-id": "req_safe123"},
        )
        response.json.return_value = {
            "error": "invalid_grant",
            "error_description": "AUTH_CODE and access-secret must never escape",
        }

        class SyntheticOAuthFailure(Exception):
            def __init__(self):
                super().__init__("AUTH_CODE access-secret raw exception")
                self.response = response

        class SyntheticSession:
            def __init__(self, **kwargs):
                pass

            def fetch_token(self, *args, **kwargs):
                raise SyntheticOAuthFailure()

        authlib = ModuleType("authlib")
        integrations = ModuleType("authlib.integrations")
        requests_client = ModuleType("authlib.integrations.requests_client")
        requests_client.OAuth2Session = SyntheticSession

        with unittest.mock.patch.dict(sys.modules, {
            "authlib": authlib,
            "authlib.integrations": integrations,
            "authlib.integrations.requests_client": requests_client,
        }):
            with self.assertRaises(AuthError) as raised:
                _AuthlibOAuthClient().exchange(
                    client_id="oaiapp-synthetic", code="AUTH_CODE",
                    verifier="synthetic-verifier",
                    redirect_uri="http://127.0.0.1:1455/auth/callback",
                )

        error = raised.exception
        self.assertTrue(str(error).startswith("Authorization could not be completed. Diagnostics: "))
        self.assertEqual(error._diagnostics, {
            "failure_category": "unknown_exchange_error",
            "exception_family": "other",
            "exception_type": "SyntheticOAuthFailure",
            "http_status": 400,
            "provider_error_code": "invalid_grant",
            "request_id": "req_safe123",
            "response_body_shape": "json_object",
        })
        for sensitive in ("AUTH_CODE", "access-secret", "raw exception", "error_description"):
            self.assertNotIn(sensitive, repr(error))
            self.assertNotIn(sensitive, repr(error._diagnostics))

    def test_exchange_failure_omits_untrusted_provider_error_values(self):
        response = Mock(
            status_code=400,
            headers={"x-request-id": "AUTH_CODE"},
        )
        response.json.return_value = {"error": "access-secret"}

        class SyntheticOAuthFailure(Exception):
            def __init__(self):
                super().__init__("raw exception with access-secret")
                self.response = response

        class SyntheticSession:
            def __init__(self, **kwargs):
                pass

            def fetch_token(self, *args, **kwargs):
                raise SyntheticOAuthFailure()

        authlib = ModuleType("authlib")
        integrations = ModuleType("authlib.integrations")
        requests_client = ModuleType("authlib.integrations.requests_client")
        requests_client.OAuth2Session = SyntheticSession

        with unittest.mock.patch.dict(sys.modules, {
            "authlib": authlib,
            "authlib.integrations": integrations,
            "authlib.integrations.requests_client": requests_client,
        }):
            with self.assertRaises(AuthError) as raised:
                _AuthlibOAuthClient().exchange(
                    client_id="oaiapp-synthetic", code="AUTH_CODE",
                    verifier="synthetic-verifier",
                    redirect_uri="http://127.0.0.1:1455/auth/callback",
                )

        self.assertEqual(raised.exception._diagnostics, {
            "failure_category": "unknown_exchange_error",
            "exception_family": "other",
            "exception_type": "SyntheticOAuthFailure",
            "http_status": 400,
            "response_body_shape": "json_object",
        })

    def test_authlib_oauth_error_shape_preserves_only_allowlisted_error_code(self):
        class AuthlibOAuthErrorShape(Exception):
            def __init__(self):
                self.error = "invalid_grant"
                self.description = "PRIVATE_AUTH_CODE access-token description"
                self.uri = "https://provider.invalid/error?token=PRIVATE_TOKEN"
                super().__init__(f"{self.error}: {self.description}")
        AuthlibOAuthErrorShape.__module__ = "authlib.oauth2.rfc6749.errors"

        class SyntheticSession:
            def __init__(self, **kwargs):
                pass

            def fetch_token(self, *args, **kwargs):
                raise AuthlibOAuthErrorShape()

        authlib = ModuleType("authlib")
        integrations = ModuleType("authlib.integrations")
        requests_client = ModuleType("authlib.integrations.requests_client")
        requests_client.OAuth2Session = SyntheticSession

        with unittest.mock.patch.dict(sys.modules, {
            "authlib": authlib,
            "authlib.integrations": integrations,
            "authlib.integrations.requests_client": requests_client,
        }):
            with self.assertRaises(AuthError) as raised:
                _AuthlibOAuthClient().exchange(
                    client_id="oaiapp-synthetic", code="PRIVATE_AUTH_CODE",
                    verifier="synthetic-verifier",
                    redirect_uri="http://127.0.0.1:1455/auth/callback",
                )

        error = raised.exception
        self.assertEqual(error._diagnostics, {
            "failure_category": "unknown_exchange_error",
            "exception_family": "authlib",
            "exception_type": "AuthlibOAuthErrorShape",
            "provider_error_code": "invalid_grant",
        })
        rendered = str(error)
        self.assertIn("provider error code: invalid_grant", rendered)
        for sensitive in ("PRIVATE_AUTH_CODE", "access-token", "PRIVATE_TOKEN",
                          "description", "error?", "https://provider.invalid"):
            self.assertNotIn(sensitive, rendered)

    def test_exchange_failure_has_fixed_safe_category_when_provider_metadata_is_absent(self):
        cases = (
            (TimeoutError("PRIVATE_TIMEOUT_DETAIL"), "exchange_timeout"),
            (ConnectionError("PRIVATE_CONNECTION_DETAIL"), "exchange_connection_error"),
            (RequestsTimeout("PRIVATE_REQUEST_TIMEOUT"), "exchange_timeout"),
            (RequestsConnectionError("PRIVATE_REQUEST_CONNECTION"),
             "exchange_connection_error"),
            (RequestsSSLError("PRIVATE_REQUEST_TLS"), "exchange_tls_error"),
            (RequestsJSONDecodeError("PRIVATE_JSON_DETAIL", "PRIVATE_DOC", 0),
             "exchange_invalid_json_response"),
            (RequestsException("PRIVATE_REQUEST_ERROR"), "unknown_exchange_error",
             "requests", "RequestException"),
            (RuntimeError("PRIVATE_PROVIDER_DETAIL"), "unknown_exchange_error",
             "stdlib", "RuntimeError"),
            (type("Bad Name;PRIVATE_CLASS_CANARY", (Exception,), {
                "__module__": "private.provider.module",
            })("PRIVATE_ERROR_TEXT"), "unknown_exchange_error", "other", None),
        )
        for case in cases:
            failure, expected_category = case[:2]
            expected_family = case[2] if len(case) > 2 else None
            expected_exception_type = case[3] if len(case) == 4 else None
            with self.subTest(category=expected_category):
                class SyntheticSession:
                    def __init__(self, **kwargs):
                        pass

                    def fetch_token(self, *args, **kwargs):
                        raise failure

                authlib = ModuleType("authlib")
                integrations = ModuleType("authlib.integrations")
                requests_client = ModuleType("authlib.integrations.requests_client")
                requests_client.OAuth2Session = SyntheticSession

                with unittest.mock.patch.dict(sys.modules, {
                    "authlib": authlib,
                    "authlib.integrations": integrations,
                    "authlib.integrations.requests_client": requests_client,
                }):
                    with self.assertRaises(AuthError) as raised:
                        _AuthlibOAuthClient().exchange(
                            client_id="oaiapp-synthetic", code="PRIVATE_CODE",
                            verifier="PRIVATE_VERIFIER",
                            redirect_uri="http://127.0.0.1:1455/auth/callback",
                        )

                error = raised.exception
                expected_diagnostics = {
                    "failure_category": expected_category,
                }
                if expected_family is not None:
                    expected_diagnostics["exception_family"] = expected_family
                if expected_exception_type is not None:
                    expected_diagnostics["exception_type"] = expected_exception_type
                self.assertEqual(error._diagnostics, expected_diagnostics)
                self.assertIn(f"failure category: {expected_category}", str(error))
                if expected_family is not None:
                    self.assertIn(f"exception family: {expected_family}", str(error))
                if expected_exception_type is not None:
                    self.assertIn(f"exception type: {expected_exception_type}", str(error))
                for sensitive in ("PRIVATE_TIMEOUT_DETAIL", "PRIVATE_CONNECTION_DETAIL",
                                  "PRIVATE_PROVIDER_DETAIL", "PRIVATE_REQUEST_TIMEOUT",
                                  "PRIVATE_REQUEST_CONNECTION", "PRIVATE_REQUEST_TLS",
                                  "PRIVATE_JSON_DETAIL", "PRIVATE_DOC",
                                  "PRIVATE_REQUEST_ERROR", "PRIVATE_CODE",
                                  "PRIVATE_VERIFIER", "JSONDecodeError",
                                  "Bad Name;PRIVATE_CLASS_CANARY",
                                  "private.provider.module",
                                  "PRIVATE_ERROR_TEXT"):
                    self.assertNotIn(sensitive, str(error))

    def test_host_id_is_persisted_as_non_secret_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = __import__("pathlib").Path(directory) / "metadata" / "host-id"
            first = load_or_create_host_id(path)
            second = load_or_create_host_id(path)
            self.assertEqual(first, second)
            self.assertTrue(first.startswith("urn:uuid:"))

    def test_auth_module_compiles_without_literal_escape_regression(self):
        from pathlib import Path
        source = Path(__file__).parents[1] / "src" / "rsna_knee" / "labels" / "siwc_auth.py"
        compile(source.read_text(encoding="utf-8"), str(source), "exec")

    def test_concurrent_expired_profile_refresh_rotates_once(self):
        profile = self.complete(self.auth.begin_sign_in())
        record = self.store.load(profile.profile_id)
        record["expires_at"] = self.now() - 1
        self.store.save(profile.profile_id, record)

        class SlowOAuth(FakeOAuth):
            def __init__(self):
                super().__init__()
                self.calls = 0
                self.started = threading.Event()
                self.release = threading.Event()

            def refresh(self, *, client_id, refresh_token):
                self.calls += 1
                self.refresh_args = (client_id, refresh_token)
                self.started.set()
                if not self.release.wait(2):
                    raise AssertionError("synthetic refresh release timed out")
                return {
                    "access_token": "one-new-access", "refresh_token": "one-new-refresh",
                    "expires_in": 3600,
                }

        slow_oauth = SlowOAuth()
        self.auth._oauth = slow_oauth
        results = []
        errors = []
        second_started = threading.Event()

        def refresh(second=False):
            try:
                if second:
                    second_started.set()
                results.append(self.auth.refresh_if_needed(profile.profile_id))
            except Exception as exc:  # surfaced in the parent test thread
                errors.append(exc)

        first = threading.Thread(target=refresh)
        second = threading.Thread(target=lambda: refresh(second=True))
        first.start()
        self.assertTrue(slow_oauth.started.wait(1))
        second.start()
        self.assertTrue(second_started.wait(1))
        time.sleep(0.05)
        calls_before_release = slow_oauth.calls
        slow_oauth.release.set()
        first.join(2)
        second.join(2)

        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(calls_before_release, 1)
        self.assertEqual(slow_oauth.calls, 1)
        self.assertEqual(len(results), 2)
        self.assertEqual(self.store.load(profile.profile_id)["refresh_token"], "one-new-refresh")

    def test_callback_state_tampering_and_wrong_loopback_target_are_rejected(self):
        request = self.auth.begin_sign_in()
        with self.assertRaises(AuthError):
            self.auth.complete_sign_in("http://127.0.0.1:1455/auth/callback?code=x&state=wrong&client_id=oaiapp-x")
        with self.assertRaises(AuthError):
            self.auth.complete_sign_in("http://localhost:1455/auth/callback?code=x&state=AAAA&client_id=oaiapp-x")

    def test_denial_and_missing_plan_scope_do_not_save_profile(self):
        request = self.auth.begin_sign_in()
        state = parse_qs(urlparse(request.authorization_url).query)["state"][0]
        with self.assertRaises(AuthError):
            self.auth.complete_sign_in(f"http://127.0.0.1:1455/auth/callback?error=access_denied&state={state}")
        request = self.auth.begin_sign_in()
        with self.assertRaises(AuthError):
            self.complete(request, scopes=["openid", "profile", "email"])
        self.assertEqual(self.store.values, {})

    def test_invalid_identity_and_returning_identity_mismatch_are_rejected(self):
        self.auth = SIWCAuthenticator(
            oauth=self.oauth, store=self.store,
            id_token_verifier=lambda token, client, nonce: {"iss": "bad", "sub": "subject-1", "aud": client, "nonce": nonce, "exp": self.now() + 5, "iat": self.now()},
            clock=self.now, host_id="urn:uuid:synthetic-host",
            redirect_uri="http://127.0.0.1:1455/auth/callback", random_value=lambda n: b"B" * n,
        )
        request = self.auth.begin_sign_in()
        with self.assertRaises(AuthError):
            self.complete(request)
        self.assertFalse(self.store.values)

    def test_reauthentication_must_preserve_verified_account_identity(self):
        profile = self.complete(self.auth.begin_sign_in())
        request = self.auth.begin_sign_in(profile.profile_id)
        self.auth._verify_id_token = lambda token, client_id, nonce: {
            "iss": "https://auth.openai.com", "sub": "different-subject",
            "aud": client_id, "nonce": nonce, "exp": self.now() + 3600,
            "iat": self.now(),
        }
        with self.assertRaises(AuthError):
            self.complete(request)
        self.assertEqual(self.store.load(profile.profile_id)["subject"], "subject-1")

    def test_invalid_client_id_and_malformed_callback_are_rejected(self):
        request = self.auth.begin_sign_in()
        query = parse_qs(urlparse(request.authorization_url).query)
        with self.assertRaises(AuthError):
            self.auth.complete_sign_in(f"http://127.0.0.1:1455/auth/callback?code=x&state={query['state'][0]}&client_id=dynamic_agent_client")
        with self.assertRaises(AuthError):
            self.auth.complete_sign_in("http://127.0.0.1:1455.evil.test/auth/callback?code=x&state=x")

    def test_expired_access_token_refresh_rotates_refresh_token(self):
        profile = self.complete(self.auth.begin_sign_in())
        record = self.store.load(profile.profile_id)
        record["expires_at"] = self.now() - 1
        self.store.save(profile.profile_id, record)
        self.oauth.refresh_result = {
            "access_token": "new-access-secret", "refresh_token": "rotated-refresh-secret",
            "id_token": "new-id-secret", "token_type": "Bearer", "expires_in": 3600,
            "scope": record["scopes"],
        }
        refreshed = self.auth.refresh_if_needed(profile.profile_id)
        self.assertTrue(refreshed.plan_usage_enabled)
        self.assertEqual(self.oauth.refresh_args[1], "refresh-secret")
        self.assertEqual(self.store.load(profile.profile_id)["refresh_token"], "rotated-refresh-secret")

    def test_credentials_are_not_exposed_in_returned_profile_or_repr(self):
        profile = self.complete(self.auth.begin_sign_in())
        serialized = repr(profile) + json.dumps(profile.__dict__)
        for secret in ("access-secret", "refresh-secret", "id-secret"):
            self.assertNotIn(secret, serialized)

    def test_keyring_store_rejects_null_and_unknown_backends(self):
        class NullBackend:
            priority = 0
            def set_password(self, *args): pass
            def get_password(self, *args): return None

        with self.assertRaises(AuthError):
            KeyringCredentialStore(keyring_module=Mock(get_keyring=lambda: NullBackend()))

        class UnknownBackend:
            priority = 5
            def set_password(self, *args): pass
            def get_password(self, *args): return None

        with self.assertRaises(AuthError):
            KeyringCredentialStore(keyring_module=Mock(get_keyring=lambda: UnknownBackend()))

    def test_keyring_store_rejects_unavailable_backend_and_never_falls_back(self):
        class WindowsKeyring:
            priority = 5
            __module__ = "keyring.backends.Windows"
            def set_password(self, *args): raise RuntimeError("synthetic backend unavailable")
            def get_password(self, *args): raise RuntimeError("synthetic backend unavailable")

        keyring = Mock(get_keyring=lambda: WindowsKeyring())
        keyring.set_password.side_effect = RuntimeError("synthetic backend unavailable")
        store = KeyringCredentialStore(keyring_module=keyring)
        with self.assertRaises(AuthError):
            store.save("profile", {"refresh_token": "secret"})

    def test_refresh_requires_rotated_refresh_token_and_granted_plan_scope(self):
        profile = self.complete(self.auth.begin_sign_in())
        record = self.store.load(profile.profile_id)
        record["expires_at"] = self.now() - 1
        self.store.save(profile.profile_id, record)
        self.oauth.refresh_result = {"access_token": "new", "expires_in": 3600, "scope": "openid"}
        with self.assertRaises(AuthError):
            self.auth.refresh_if_needed(profile.profile_id)

    def test_refresh_without_id_token_retains_verified_identity_and_prior_scope(self):
        profile = self.complete(self.auth.begin_sign_in())
        record = self.store.load(profile.profile_id)
        record["expires_at"] = self.now() - 1
        self.store.save(profile.profile_id, record)
        self.oauth.refresh_result = {
            "access_token": "new-access-secret", "refresh_token": "rotated-refresh-secret",
            "expires_in": 3600,
        }
        refreshed = self.auth.refresh_if_needed(profile.profile_id)
        self.assertEqual(refreshed.subject, "subject-1")
        self.assertTrue(refreshed.plan_usage_enabled)
        self.assertEqual(self.store.load(profile.profile_id)["id_token"], "id-secret")
        self.assertEqual(self.store.load(profile.profile_id)["refresh_token"], "rotated-refresh-secret")

    def test_expired_or_invalid_id_token_is_rejected_before_credentials_are_saved(self):
        self.auth._verify_id_token = lambda token, client_id, nonce: {
            "iss": "https://auth.openai.com", "sub": "subject-1",
            "aud": client_id, "nonce": nonce, "exp": self.now() - 1,
            "iat": self.now(),
        }
        with self.assertRaises(AuthError):
            self.complete(self.auth.begin_sign_in())
        self.assertFalse(self.store.values)


if __name__ == "__main__":
    unittest.main()
