import os
import sys
import httpx

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import data_loader


class FakeAuth:
    def __init__(self):
        self.calls = []

    def set_session(self, jwt, refresh_token):
        self.calls.append(("set_session", jwt, refresh_token))

    def sign_in_with_password(self, creds):
        self.calls.append(("sign_in_with_password", creds))


class FakeClient:
    def __init__(self):
        self.auth = FakeAuth()


def make_client(*args, **kwargs):
    return FakeClient()


def test_get_client_uses_jwt(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://sb.example.com")
    monkeypatch.setenv("SUPABASE_KEY", "anon")
    monkeypatch.setenv("SUPABASE_JWT", "token")
    called = {}

    def fake_create(url, key):
        called["args"] = (url, key)
        return make_client()

    monkeypatch.setattr(data_loader, "create_client", fake_create)
    client = data_loader._get_client()
    assert called["args"] == ("https://sb.example.com", "anon")
    assert client.auth.calls == [("set_session", "token", "")]


def test_get_client_with_password(monkeypatch):
    monkeypatch.delenv("SUPABASE_JWT", raising=False)
    monkeypatch.setenv("SUPABASE_URL", "https://sb.example.com")
    monkeypatch.setenv("SUPABASE_KEY", "anon")
    monkeypatch.setenv("SUPABASE_USER_EMAIL", "u@example.com")
    monkeypatch.setenv("SUPABASE_PASSWORD", "pw")
    called = {}

    def fake_create(url, key):
        called["args"] = (url, key)
        return make_client()

    monkeypatch.setattr(data_loader, "create_client", fake_create)
    client = data_loader._get_client()
    assert called["args"] == ("https://sb.example.com", "anon")
    assert client.auth.calls == [
        ("sign_in_with_password", {"email": "u@example.com", "password": "pw"})
    ]


@pytest.mark.parametrize("exc_type", [httpx.HTTPError, ValueError])
def test_get_client_auth_error(monkeypatch, exc_type):
    monkeypatch.delenv("SUPABASE_JWT", raising=False)
    monkeypatch.setenv("SUPABASE_URL", "https://sb.example.com")
    monkeypatch.setenv("SUPABASE_KEY", "anon")
    monkeypatch.setenv("SUPABASE_USER_EMAIL", "u@example.com")
    monkeypatch.setenv("SUPABASE_PASSWORD", "pw")

    class BadClient(FakeClient):
        def __init__(self):
            super().__init__()

    def bad_create(url, key):
        c = BadClient()

        def fail(*a, **k):
            raise exc_type("fail")

        c.auth.sign_in_with_password = fail
        return c

    monkeypatch.setattr(data_loader, "create_client", bad_create)
    with pytest.raises(exc_type):
        data_loader._get_client()
