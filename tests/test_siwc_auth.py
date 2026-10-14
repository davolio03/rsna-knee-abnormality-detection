import builtins
import json
import hashlib
import sys
import threading
import tempfile
import time
import unittest
from datetime import datetime, timezone
from types import ModuleType
from unittest.mock import Mock, patch
import requests
from urllib.parse import parse_qs, urlparse
from requests.exceptions import (
    ConnectionError as RequestsConnectionError,
    JSONDecodeError as RequestsJSONDecodeError,
    RequestException as RequestsException,
    SSLError as RequestsSSLError,
    Timeout as RequestsTimeout,
)

from rsna_knee.labels.siwc_auth import (
    AUTHORIZATION_ENDPOINT,
    PLAN_SCOPE,
    RESOURCE,
    SCOPES,
    AuthError,
    KeyringCredentialStore,
    SIWCAuthenticator,
    _AuthlibOAuthClient,
    _verify_openai_id_token,
    load_or_create_host_id,
)


class FakeOAuth:
    def __init__(self):
        self.authorization = None
        self.exchange_result = None
        self.refresh_result = None
        self.exchange_args = None
        self.refresh_args = None
        self.refresh_calls = 0

    def exchange(self, *, client_id, code, verifier, redirect_uri):
        self.exchange_args = (client_id, code, verifier, redirect_uri)
        return dict(self.exchange_result)

    def refresh(self, *, client_id, refresh_token):
        self.refresh_calls += 1
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

    def _signed_id_token(self, **overrides):
        from authlib.jose import jwk, jwt
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import rsa

        now = int(self.now())
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        private_pem = private_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        public_jwk = jwk.dumps(private_key.public_key(), "RSA")
        public_jwk.update(kid="offline-test-key")
        claims = {
            "iss": "https://auth.openai.com", "sub": "SUBJECT-CANARY",
            "aud": "synthetic-client", "nonce": "NONCE-CANARY",
            "iat": now, "exp": now + 120,
        }
        claims.update(overrides)
        token = jwt.encode(
            {"alg": "RS256", "kid": "offline-test-key"}, claims, private_pem
        )
        return token, {"keys": [public_jwk]}, now, private_key

    @staticmethod
    def _jwks_response(jwks):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps(jwks).encode("utf-8")

        return Response()

    def test_real_authlib_accepts_iat_and_nbf_within_five_second_clock_skew(self):
        token, jwks, now, _private_key = self._signed_id_token(
            iat=int(self.now()) + 5, nbf=int(self.now()) + 5
        )
        with patch("rsna_knee.labels.siwc_auth.time.time", return_value=now):
            with patch("urllib.request.urlopen", return_value=self._jwks_response(jwks)):
                claims = _verify_openai_id_token(
                    token, "synthetic-client", "NONCE-CANARY"
                )
        self.assertEqual(claims["iat"], now + 5)
        self.assertEqual(claims["nbf"], now + 5)

    def test_real_authlib_rejects_iat_and_nbf_beyond_five_second_clock_skew(self):
        for claim in ("iat", "nbf"):
            with self.subTest(claim=claim):
                token, jwks, now, _private_key = self._signed_id_token(
                    **{claim: int(self.now()) + 6}
                )
                with patch("rsna_knee.labels.siwc_auth.time.time", return_value=now):
                    with patch("urllib.request.urlopen", return_value=self._jwks_response(jwks)):
                        with self.assertRaises(AuthError) as raised:
                            _verify_openai_id_token(
                                token, "synthetic-client", "NONCE-CANARY"
                            )
                self.assertEqual(
                    raised.exception._diagnostics,
                    {"failure_category": "identity_claims_invalid"},
                )

    def test_real_authlib_rejects_expired_token_at_verifier_and_profile_boundaries(self):
        token, jwks, now, _private_key = self._signed_id_token(exp=int(self.now()) - 1)
        with patch("rsna_knee.labels.siwc_auth.time.time", return_value=now):
            with patch("urllib.request.urlopen", return_value=self._jwks_response(jwks)):
                with self.assertRaises(AuthError) as verified_error:
                    _verify_openai_id_token(
                        token, "synthetic-client", "NONCE-CANARY"
                    )
        self.assertEqual(
            verified_error.exception._diagnostics,
            {"failure_category": "identity_claims_invalid"},
        )

        self.auth._verify_id_token = lambda value, client_id, nonce: (
            _verify_openai_id_token(value, client_id, nonce)
        )
        with patch("rsna_knee.labels.siwc_auth.time.time", return_value=now):
            with patch("urllib.request.urlopen", return_value=self._jwks_response(jwks)):
                with self.assertRaises(AuthError):
                    self.auth._validated_profile(
                        {"id_token": token}, "synthetic-client", "NONCE-CANARY", None
                    )

    def test_profile_validation_rejects_expiration_independently_of_verifier(self):
        self.auth._verify_id_token = lambda _token, client_id, nonce: {
            "iss": "https://auth.openai.com", "sub": "subject-1",
            "aud": client_id, "nonce": nonce,
            "exp": self.now() - 1, "iat": self.now(),
        }

        with self.assertRaises(AuthError) as raised:
            self.auth._validated_profile(
                {"id_token": "synthetic-already-verified-token"},
                "synthetic-client", "synthetic-nonce", None,
            )

        self.assertEqual(
            raised.exception._diagnostics,
            {"failure_category": "identity_claims_invalid"},
        )
        self.assertFalse(self.store.values)

    def test_real_authlib_still_rejects_bad_signature_issuer_audience_and_nonce(self):
        _token, jwks, now, _private_key = self._signed_id_token()
        bad_signature_token, _bad_jwks, _now, _different_key = self._signed_id_token()
        wrong_issuer, issuer_jwks, _now, _key = self._signed_id_token(
            iss="https://attacker.invalid"
        )
        wrong_audience, audience_jwks, _now, _key = self._signed_id_token(
            aud="another-client"
        )
        wrong_nonce, nonce_jwks, _now, _key = self._signed_id_token(
            nonce="another-nonce"
        )
        invalid_cases = (
            (bad_signature_token, jwks, "synthetic-client", "NONCE-CANARY"),
            (wrong_issuer, issuer_jwks, "synthetic-client", "NONCE-CANARY"),
            (wrong_audience, audience_jwks, "synthetic-client", "NONCE-CANARY"),
            (wrong_nonce, nonce_jwks, "synthetic-client", "NONCE-CANARY"),
        )
        for index, (candidate, candidate_jwks, client_id, nonce) in enumerate(invalid_cases):
            with self.subTest(index=index):
                with patch("rsna_knee.labels.siwc_auth.time.time", return_value=now):
                    with patch("urllib.request.urlopen", return_value=self._jwks_response(candidate_jwks)):
                        with self.assertRaises(AuthError) as raised:
                            _verify_openai_id_token(candidate, client_id, nonce)
                for canary in (
                    candidate.decode("ascii"), "SUBJECT-CANARY", "NONCE-CANARY",
                    "attacker.invalid", "another-client", "another-nonce",
                ):
                    self.assertNotIn(canary, str(raised.exception))
                    self.assertNotIn(canary, repr(raised.exception._diagnostics))

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

    def test_sign_in_persists_refresh_token_and_metadata_only(self):
        profile = self.complete(self.auth.begin_sign_in())

        record = self.store.load(profile.profile_id)
        self.assertEqual(
            set(record),
            {"client_id", "subject", "email", "issuer", "refresh_token",
             "token_type", "scopes", "expires_at"},
        )
        serialized = json.dumps(record)
        self.assertNotIn("access_token", serialized)
        self.assertNotIn("id_token", serialized)

    def test_fresh_authenticator_refreshes_legacy_record_before_callback_use(self):
        profile = self.complete(self.auth.begin_sign_in())
        legacy = self.store.load(profile.profile_id)
        legacy.update({"access_token": "synthetic-legacy-access",
                       "id_token": "synthetic-legacy-id",
                       "expires_at": self.now() + 3600})
        self.store.save(profile.profile_id, legacy)
        fresh_oauth = FakeOAuth()
        fresh_oauth.refresh_result = {
            "access_token": "synthetic-fresh-access",
            "refresh_token": "synthetic-rotated-refresh",
            "expires_in": 3600,
        }
        fresh_auth = SIWCAuthenticator(
            store=self.store, oauth=fresh_oauth, clock=self.now,
            host_id="urn:uuid:synthetic-host",
            redirect_uri="http://127.0.0.1:1455/auth/callback",
            random_value=lambda n: b"B" * n,
        )

        result = fresh_auth.with_plan_access_token(
            profile.profile_id, lambda token: token == "synthetic-fresh-access"
        )

        self.assertTrue(result)
        persisted = self.store.load(profile.profile_id)
        self.assertEqual(
            set(persisted),
            {"client_id", "subject", "email", "issuer", "refresh_token",
             "token_type", "scopes", "expires_at"},
        )
        serialized = json.dumps(persisted)
        self.assertNotIn("access_token", serialized)
        self.assertNotIn("id_token", serialized)

    def test_callback_uses_just_refreshed_short_lived_token_once(self):
        profile = self.complete(self.auth.begin_sign_in())
        record = self.store.load(profile.profile_id)
        record["expires_at"] = self.now() - 1
        self.store.save(profile.profile_id, record)
        self.oauth.refresh_result = {
            "access_token": "synthetic-short-lived-access",
            "refresh_token": "synthetic-short-lived-rotation",
            "expires_in": 20,
        }

        result = self.auth.with_plan_access_token(
            profile.profile_id,
            lambda token: token == "synthetic-short-lived-access",
        )

        self.assertTrue(result)
        self.assertEqual(self.oauth.refresh_calls, 1)

    def test_concurrent_callbacks_share_one_refresh_and_release_lock_before_operation(self):
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
                self.started.set()
                if not self.release.wait(2):
                    raise AssertionError("synthetic refresh release timed out")
                return {
                    "access_token": "synthetic-shared-access",
                    "refresh_token": "synthetic-shared-rotation",
                    "expires_in": 20,
                }

        slow_oauth = SlowOAuth()
        self.auth._oauth = slow_oauth
        operation_barrier = threading.Barrier(2)
        results = []
        errors = []

        def call():
            try:
                results.append(self.auth.with_plan_access_token(
                    profile.profile_id,
                    lambda token: (
                        operation_barrier.wait(timeout=1) is not None
                        and token == "synthetic-shared-access"
                    ),
                ))
            except Exception as exc:
                errors.append(type(exc).__name__)

        first = threading.Thread(target=call)
        second = threading.Thread(target=call)
        first.start()
        self.assertTrue(slow_oauth.started.wait(1))
        second.start()
        time.sleep(0.05)
        slow_oauth.release.set()
        first.join(2)
        second.join(2)

        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(slow_oauth.calls, 1)
        self.assertEqual(results, [True, True])

    def test_reauthentication_store_and_cache_update_are_atomic_for_callbacks(self):
        profile = self.complete(self.auth.begin_sign_in())
        original_save = self.store.save
        save_entered = threading.Event()
        release_save = threading.Event()
        callback_finished = threading.Event()
        results = []
        errors = []

        def blocking_save(profile_id, record):
            original_save(profile_id, record)
            save_entered.set()
            if not release_save.wait(2):
                raise AssertionError("synthetic sign-in save release timed out")

        self.store.save = blocking_save
        request = self.auth.begin_sign_in(profile.profile_id)
        self.oauth.exchange_result = {
            "access_token": "synthetic-reauth-access",
            "refresh_token": "synthetic-reauth-refresh",
            "id_token": "synthetic-reauth-id",
            "token_type": "Bearer",
            "expires_in": 3600,
            "scope": "openid profile email offline_access resource.invoke " + PLAN_SCOPE,
        }
        state = parse_qs(urlparse(request.authorization_url).query)["state"][0]
        callback_url = (
            "http://127.0.0.1:1455/auth/callback?code=synthetic-code"
            f"&state={state}"
        )

        def reauthenticate():
            try:
                self.auth.complete_sign_in(callback_url)
            except Exception as exc:
                errors.append(type(exc).__name__)

        def use_profile():
            try:
                results.append(self.auth.with_plan_access_token(
                    profile.profile_id,
                    lambda token: token == "synthetic-reauth-access",
                ))
            except Exception as exc:
                errors.append(type(exc).__name__)
            finally:
                callback_finished.set()

        sign_in_thread = threading.Thread(target=reauthenticate)
        sign_in_thread.start()
        self.assertTrue(save_entered.wait(1))
        callback_thread = threading.Thread(target=use_profile)
        callback_thread.start()
        time.sleep(0.05)
        callback_was_blocked = not callback_finished.is_set()
        release_save.set()
        sign_in_thread.join(2)
        callback_thread.join(2)
        self.store.save = original_save

        self.assertFalse(sign_in_thread.is_alive())
        self.assertFalse(callback_thread.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(callback_was_blocked)
        self.assertEqual(results, [True])

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

    def test_unknown_exception_type_str_subclass_cannot_customize_diagnostics(self):
        class HostileTypeName(str):
            def __format__(self, format_spec):
                return "PRIVATE_FORMAT_LEAK"

            def __repr__(self):
                return "PRIVATE_REPR_LEAK"

        class HostileNamedError(Exception):
            pass

        HostileNamedError.__name__ = HostileTypeName("SafeLookingError")

        class SyntheticSession:
            def __init__(self, **kwargs):
                pass

            def fetch_token(self, *args, **kwargs):
                raise HostileNamedError("PRIVATE_ERROR_MESSAGE")

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
        rendered = str(error)
        diagnostics_repr = repr(error._diagnostics)
        for sensitive in ("SafeLookingError", "PRIVATE_FORMAT_LEAK",
                          "PRIVATE_REPR_LEAK", "PRIVATE_ERROR_MESSAGE",
                          "PRIVATE_CODE", "PRIVATE_VERIFIER"):
            self.assertNotIn(sensitive, rendered)
            self.assertNotIn(sensitive, diagnostics_repr)
        self.assertNotIn("exception_type", error._diagnostics)

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
        with self.assertRaises(AuthError) as raised:
            self.complete(request)
        self.assertEqual(
            raised.exception._diagnostics,
            {"failure_category": "identity_claims_invalid"},
        )
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

    def test_authlib_refresh_serializes_no_scope_and_preserves_refresh_fields(self):
        requests_seen = []

        def send_without_network(_session, request, **kwargs):
            requests_seen.append(request)
            response = requests.Response()
            response.status_code = 200
            response.headers["Content-Type"] = "application/json"
            response._content = (
                b'{"access_token":"new-access",'
                b'"refresh_token":"rotated-refresh",'
                b'"token_type":"Bearer","expires_in":3600}'
            )
            response.request = request
            return response

        with patch("requests.sessions.Session.send", send_without_network):
            result = _AuthlibOAuthClient().refresh(
                client_id="synthetic-client",
                refresh_token="synthetic-refresh",
            )

        self.assertEqual(result["refresh_token"], "rotated-refresh")
        self.assertEqual(len(requests_seen), 1)
        form = parse_qs(requests_seen[0].body)
        self.assertEqual(form.get("grant_type"), ["refresh_token"])
        self.assertEqual(form.get("client_id"), ["synthetic-client"])
        self.assertEqual(form.get("resource"), [RESOURCE])
        self.assertEqual(form.get("refresh_token"), ["synthetic-refresh"])
        self.assertNotIn("scope", form)
        self.assertNotIn("include_client_id", form)

        authorization_url, _ = _AuthlibOAuthClient._session(
            "synthetic-client"
        ).create_authorization_url(
            AUTHORIZATION_ENDPOINT, scope=SCOPES
        )
        self.assertEqual(
            parse_qs(urlparse(authorization_url).query).get("scope"), [SCOPES]
        )

    def test_authlib_refresh_preserves_only_safe_failure_diagnostics(self):
        class FailedSession:
            def refresh_token(self, *args, **kwargs):
                raise requests.ConnectionError("PRIVATE-REFRESH-CANARY")

        with patch.object(_AuthlibOAuthClient, "_session", return_value=FailedSession()):
            with self.assertRaises(AuthError) as raised:
                _AuthlibOAuthClient().refresh(
                    client_id="synthetic-client", refresh_token="REFRESH-CANARY"
                )

        self.assertEqual(raised.exception._diagnostics, {
            "failure_category": "exchange_connection_error",
        })
        self.assertIn("failure category: exchange_connection_error", str(raised.exception))
        for canary in ("PRIVATE-REFRESH-CANARY", "REFRESH-CANARY", "synthetic-client"):
            self.assertNotIn(canary, str(raised.exception))
            self.assertNotIn(canary, repr(raised.exception._diagnostics))

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

    def test_keyring_payload_filters_legacy_access_and_identity_token_fields(self):
        class WindowsKeyring:
            priority = 5
            __module__ = "keyring.backends.Windows"
            def set_password(self, *args): pass
            def get_password(self, *args): return None

        keyring = Mock(get_keyring=lambda: WindowsKeyring())
        store = KeyringCredentialStore(keyring_module=keyring)
        store.save("profile", {
            "client_id": "synthetic-client", "subject": "synthetic-subject",
            "refresh_token": "synthetic-refresh", "access_token": "synthetic-access",
            "id_token": "synthetic-id", "expires_at": self.now() + 300,
        })

        payload = json.loads(keyring.set_password.call_args.args[2])
        self.assertEqual(
            set(payload), {"client_id", "subject", "refresh_token", "expires_at"}
        )
        self.assertNotIn("access_token", payload)
        self.assertNotIn("id_token", payload)

    def test_keyring_save_failure_reports_bounded_payload_size_without_secrets(self):
        class WindowsKeyring:
            priority = 5
            __module__ = "keyring.backends.Windows"
            def set_password(self, *args):
                raise RuntimeError("PRIVATE_BACKEND_ERROR private-token")
            def get_password(self, *args):
                return None

        secret = "private-token"
        empty_payload = json.dumps(
            {"refresh_token": ""}, separators=(",", ":"), sort_keys=True
        )
        value = secret + "x" * (1280 - len(empty_payload) - len(secret))
        record = {"refresh_token": value}
        serialized = json.dumps(record, separators=(",", ":"), sort_keys=True)
        self.assertEqual(len(serialized.encode("utf-16-le")), 2560)
        keyring = Mock(get_keyring=lambda: WindowsKeyring())
        keyring.set_password.side_effect = RuntimeError(
            "PRIVATE_BACKEND_ERROR private-token"
        )
        store = KeyringCredentialStore(keyring_module=keyring)

        with self.assertRaises(AuthError) as raised:
            store.save("profile", record)

        error = raised.exception
        self.assertEqual(error._diagnostics, {
            "windows_credential_blob_bytes": len(serialized.encode("utf-16-le")),
            "windows_blob_limit_bytes": 2560,
            "exceeds_windows_blob_limit": False,
        })
        self.assertIn(
            f"Windows credential blob: {len(serialized.encode('utf-16-le'))} bytes",
            str(error),
        )
        self.assertIn("exceeds limit: no", str(error))
        for sensitive in (secret, "PRIVATE_BACKEND_ERROR", serialized):
            self.assertNotIn(sensitive, str(error))
            self.assertNotIn(sensitive, repr(error))
            self.assertNotIn(sensitive, repr(error._diagnostics))

    def test_keyring_save_failure_reports_oversize_payload_without_secrets(self):
        class WindowsKeyring:
            priority = 5
            __module__ = "keyring.backends.Windows"
            def set_password(self, *args):
                raise RuntimeError("PRIVATE_BACKEND_ERROR private-token")
            def get_password(self, *args):
                return None

        secret = "private-token"
        empty_payload = json.dumps(
            {"refresh_token": ""}, separators=(",", ":"), sort_keys=True
        )
        value = secret + "x" * (1281 - len(empty_payload) - len(secret))
        record = {"refresh_token": value}
        serialized = json.dumps(record, separators=(",", ":"), sort_keys=True)
        self.assertGreater(len(serialized.encode("utf-16-le")), 2560)
        keyring = Mock(get_keyring=lambda: WindowsKeyring())
        keyring.set_password.side_effect = RuntimeError(
            "PRIVATE_BACKEND_ERROR private-token"
        )
        store = KeyringCredentialStore(keyring_module=keyring)

        with self.assertRaises(AuthError) as raised:
            store.save("profile", record)

        error = raised.exception
        self.assertEqual(error._diagnostics, {
            "windows_credential_blob_bytes": len(serialized.encode("utf-16-le")),
            "windows_blob_limit_bytes": 2560,
            "exceeds_windows_blob_limit": True,
        })
        self.assertIn(
            f"Windows credential blob: {len(serialized.encode('utf-16-le'))} bytes",
            str(error),
        )
        self.assertIn("exceeds limit: yes", str(error))
        for sensitive in (secret, "PRIVATE_BACKEND_ERROR", serialized):
            self.assertNotIn(sensitive, str(error))
            self.assertNotIn(sensitive, repr(error))
            self.assertNotIn(sensitive, repr(error._diagnostics))

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
        persisted = self.store.load(profile.profile_id)
        self.assertNotIn("id_token", persisted)
        self.assertNotIn("access_token", persisted)
        self.assertEqual(
            set(persisted),
            {"client_id", "subject", "email", "issuer", "refresh_token",
             "token_type", "scopes", "expires_at"},
        )

    def test_expired_or_invalid_id_token_is_rejected_before_credentials_are_saved(self):
        self.auth._verify_id_token = lambda token, client_id, nonce: {
            "iss": "https://auth.openai.com", "sub": "subject-1",
            "aud": client_id, "nonce": nonce, "exp": self.now() - 1,
            "iat": self.now(),
        }
        with self.assertRaises(AuthError):
            self.complete(self.auth.begin_sign_in())
        self.assertFalse(self.store.values)

    def test_id_token_verifier_classifies_jwks_fetch_failure_without_leaking_values(self):
        token_canary = "JWT-PRIVATE-CANARY"
        url_canary = "https://private.invalid/jwks?secret=URL-CANARY"
        exception_canary = "PRIVATE-EXCEPTION-CANARY"
        with patch(
            "urllib.request.urlopen",
            side_effect=RuntimeError(f"{exception_canary} {url_canary}"),
        ):
            with self.assertRaises(AuthError) as raised:
                _verify_openai_id_token(token_canary, "client-secret", "nonce-secret")

        self.assertEqual(
            raised.exception._diagnostics,
            {"failure_category": "jwks_fetch_failed"},
        )
        self.assertIn("failure category: jwks_fetch_failed", str(raised.exception))
        for canary in (token_canary, url_canary, exception_canary, "client-secret", "nonce-secret"):
            self.assertNotIn(canary, str(raised.exception))
            self.assertNotIn(canary, repr(raised.exception._diagnostics))

    def test_id_token_verifier_classifies_malformed_jwks_without_leaking_body(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b'{"keys":"JWKS-BODY-CANARY"}'

        with patch("urllib.request.urlopen", return_value=Response()):
            with self.assertRaises(AuthError) as raised:
                _verify_openai_id_token("JWT-CANARY", "client", "nonce")

        self.assertEqual(
            raised.exception._diagnostics,
            {"failure_category": "jwks_document_invalid"},
        )
        self.assertIn("failure category: jwks_document_invalid", str(raised.exception))
        for canary in ("JWKS-BODY-CANARY", "JWT-CANARY", "client", "nonce"):
            self.assertNotIn(canary, str(raised.exception))
            self.assertNotIn(canary, repr(raised.exception._diagnostics))

    def test_id_token_verifier_classifies_token_verification_rejection_safely(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b'{"keys":[{"kty":"RSA","kid":"synthetic"}]}'

        jose = ModuleType("authlib.jose")

        class RejectingJWT:
            def __init__(self, algorithms):
                self.algorithms = algorithms

            def decode(self, token, jwks, *, claims_options):
                raise ValueError("JWT-PRIVATE-CANARY URL-PRIVATE-CANARY")

        jose.JsonWebToken = RejectingJWT
        with patch.dict(sys.modules, {"authlib.jose": jose}):
            with patch("urllib.request.urlopen", return_value=Response()):
                with self.assertRaises(AuthError) as raised:
                    _verify_openai_id_token("JWT-CANARY", "client", "nonce")

        self.assertEqual(
            raised.exception._diagnostics,
            {"failure_category": "identity_token_verification_failed"},
        )
        self.assertIn(
            "failure category: identity_token_verification_failed",
            str(raised.exception),
        )
        for canary in ("JWT-PRIVATE-CANARY", "URL-PRIVATE-CANARY", "JWT-CANARY", "client", "nonce"):
            self.assertNotIn(canary, str(raised.exception))
            self.assertNotIn(canary, repr(raised.exception._diagnostics))

    def test_id_token_verifier_classifies_claim_validation_failure_separately(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b'{"keys":[{"kty":"RSA","kid":"synthetic"}]}'

        class InvalidClaims(dict):
            def validate(self, *, now, leeway):
                self_test.assertEqual(leeway, 5)
                raise ValueError("CLAIM-EXCEPTION-CANARY CLAIM-VALUE-CANARY")

        jose = ModuleType("authlib.jose")
        self_test = self

        class DecodingJWT:
            def __init__(self, algorithms):
                self.algorithms = algorithms

            def decode(self, token, jwks, *, claims_options):
                return InvalidClaims({"iss": "claim-private", "sub": "synthetic"})

        jose.JsonWebToken = DecodingJWT
        with patch.dict(sys.modules, {"authlib.jose": jose}):
            with patch("urllib.request.urlopen", return_value=Response()):
                with self.assertRaises(AuthError) as raised:
                    _verify_openai_id_token("CLAIM-TOKEN-CANARY", "client", "nonce")

        self.assertEqual(
            raised.exception._diagnostics,
            {"failure_category": "identity_claims_invalid"},
        )
        self.assertIn("failure category: identity_claims_invalid", str(raised.exception))
        for canary in (
            "CLAIM-EXCEPTION-CANARY", "CLAIM-VALUE-CANARY",
            "CLAIM-TOKEN-CANARY", "claim-private", "client", "nonce",
        ):
            self.assertNotIn(canary, str(raised.exception))
            self.assertNotIn(canary, repr(raised.exception._diagnostics))

    def test_id_token_verifier_accepts_verified_claims(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b'{"keys":[{"kty":"RSA","kid":"synthetic"}]}'

        class Claims(dict):
            def validate(self, *, now, leeway):
                self_test.assertEqual(leeway, 5)
                self.validated_at = now
                self.leeway = leeway

        expected = {
            "iss": "https://auth.openai.com",
            "sub": "synthetic-subject",
            "aud": "synthetic-client",
            "nonce": "synthetic-nonce",
            "exp": int(time.time()) + 3600,
        }
        jose = ModuleType("authlib.jose")
        self_test = self

        class AcceptingJWT:
            def __init__(self, algorithms):
                self.algorithms = algorithms

            def decode(self, token, jwks, *, claims_options):
                self.token = token
                self.jwks = jwks
                self.claims_options = claims_options
                return Claims(expected)

        jose.JsonWebToken = AcceptingJWT
        with patch.dict(sys.modules, {"authlib.jose": jose}):
            with patch("urllib.request.urlopen", return_value=Response()):
                result = _verify_openai_id_token(
                    "synthetic-token", "synthetic-client", "synthetic-nonce"
                )

        self.assertEqual(result, expected)

    def test_id_token_verifier_unavailable_import_is_classified_without_leaks(self):
        token_canary = "IMPORT-TOKEN-CANARY"
        exception_canary = "IMPORT-EXCEPTION-CANARY"
        original_import = builtins.__import__

        def fail_authlib_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "authlib.jose":
                raise RuntimeError(exception_canary)
            return original_import(name, globals, locals, fromlist, level)

        network = Mock()
        with patch("builtins.__import__", side_effect=fail_authlib_import):
            with patch("urllib.request.urlopen", network):
                with self.assertRaises(AuthError) as raised:
                    _verify_openai_id_token(token_canary, "IMPORT-CLIENT-CANARY", None)

        self.assertEqual(
            raised.exception._diagnostics,
            {"failure_category": "identity_verifier_unavailable"},
        )
        self.assertIn(
            "failure category: identity_verifier_unavailable",
            str(raised.exception),
        )
        for canary in (token_canary, exception_canary, "IMPORT-CLIENT-CANARY"):
            self.assertNotIn(canary, str(raised.exception))
            self.assertNotIn(canary, repr(raised.exception._diagnostics))
        network.assert_not_called()


if __name__ == "__main__":
    unittest.main()
