"""Tests for GitHub Enterprise Cloud data-residency (GHE.com) Copilot support.

GHE.com tenants serve the Copilot API from ``copilot-api.<tenant>.ghe.com``,
accept the raw ``gho_`` token directly (no ``copilot_internal/v2/token``
exchange endpoint exists on the tenant), and select the model catalog by
the ``Copilot-Integration-Id`` header. Configuration is driven by
``COPILOT_GH_HOST`` (existing var, also used for ``gh auth token
--hostname``) with optional ``COPILOT_API_BASE_URL`` /
``COPILOT_INTEGRATION_ID`` overrides. With none of them set, behaviour is
identical to stock github.com Copilot.
"""

from unittest.mock import patch

import pytest

from hermes_cli.copilot_auth import (
    copilot_api_base_url,
    copilot_gh_host,
    copilot_integration_id,
    copilot_request_headers,
    exchange_copilot_token,
    is_ghe_tenant,
)
from utils import base_url_host_matches


_TENANT_ENV_VARS = (
    "COPILOT_GH_HOST",
    "COPILOT_API_BASE_URL",
    "COPILOT_INTEGRATION_ID",
)


@pytest.fixture(autouse=True)
def _clean_copilot_env(monkeypatch):
    for var in _TENANT_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


class TestDefaultsUnchanged:
    """With no tenant configuration, stock github.com behaviour holds."""

    def test_defaults(self):
        assert copilot_gh_host() == ""
        assert is_ghe_tenant() is False
        assert copilot_api_base_url() == "https://api.githubcopilot.com"
        assert copilot_integration_id() == "vscode-chat"

    def test_github_com_host_is_not_a_tenant(self, monkeypatch):
        monkeypatch.setenv("COPILOT_GH_HOST", "github.com")
        assert is_ghe_tenant() is False
        assert copilot_api_base_url() == "https://api.githubcopilot.com"
        assert copilot_integration_id() == "vscode-chat"

    def test_headers_default_integration_id(self):
        headers = copilot_request_headers()
        assert headers["Copilot-Integration-Id"] == "vscode-chat"

    def test_host_matches_unconfigured_tenant_is_rejected(self):
        assert not base_url_host_matches(
            "https://copilot-api.acme.ghe.com", "api.githubcopilot.com"
        )
        assert not base_url_host_matches(
            "https://copilot-api.acme.ghe.com", "githubcopilot.com"
        )


class TestTenantConfiguration:
    def test_gh_host_drives_base_url_and_integration_id(self, monkeypatch):
        monkeypatch.setenv("COPILOT_GH_HOST", "acme.ghe.com")
        assert copilot_gh_host() == "acme.ghe.com"
        assert is_ghe_tenant() is True
        assert copilot_api_base_url() == "https://copilot-api.acme.ghe.com"
        assert copilot_integration_id() == "copilot-developer-cli"

    def test_gh_host_scheme_and_slash_normalized(self, monkeypatch):
        monkeypatch.setenv("COPILOT_GH_HOST", "https://acme.ghe.com/")
        assert copilot_gh_host() == "acme.ghe.com"

    def test_base_url_override_wins(self, monkeypatch):
        monkeypatch.setenv("COPILOT_GH_HOST", "acme.ghe.com")
        monkeypatch.setenv("COPILOT_API_BASE_URL", "https://proxy.example.com/copilot/")
        assert copilot_api_base_url() == "https://proxy.example.com/copilot"

    def test_integration_id_override_wins(self, monkeypatch):
        monkeypatch.setenv("COPILOT_GH_HOST", "acme.ghe.com")
        monkeypatch.setenv("COPILOT_INTEGRATION_ID", "vscode-chat")
        assert copilot_integration_id() == "vscode-chat"
        headers = copilot_request_headers()
        assert headers["Copilot-Integration-Id"] == "vscode-chat"

    def test_headers_carry_tenant_integration_id(self, monkeypatch):
        monkeypatch.setenv("COPILOT_GH_HOST", "acme.ghe.com")
        headers = copilot_request_headers()
        assert headers["Copilot-Integration-Id"] == "copilot-developer-cli"


class TestTenantTokenExchange:
    def test_exchange_short_circuits_for_tenant(self, monkeypatch):
        """Tenants have no copilot_internal/v2/token endpoint — the raw
        token must be returned without any network round trip."""
        monkeypatch.setenv("COPILOT_GH_HOST", "acme.ghe.com")
        with patch(
            "urllib.request.urlopen",
            side_effect=AssertionError("network call attempted for tenant"),
        ):
            api_token, expires_at, base_url = exchange_copilot_token("gho_raw")
        assert api_token == "gho_raw"
        assert expires_at > 0
        assert base_url == "https://copilot-api.acme.ghe.com"


class TestHostMatchesTenantAlias:
    """base_url_host_matches treats the configured tenant Copilot host as
    an alias of the canonical githubcopilot.com hosts, so every api_mode /
    header gate applies to tenant deployments unchanged."""

    def test_alias_matches_both_domain_spellings(self, monkeypatch):
        monkeypatch.setenv("COPILOT_GH_HOST", "acme.ghe.com")
        tenant = "https://copilot-api.acme.ghe.com"
        assert base_url_host_matches(tenant, "api.githubcopilot.com")
        assert base_url_host_matches(tenant, "githubcopilot.com")

    def test_alias_from_explicit_base_url_override(self, monkeypatch):
        monkeypatch.setenv("COPILOT_API_BASE_URL", "https://copilot.internal.example.com")
        assert base_url_host_matches(
            "https://copilot.internal.example.com", "api.githubcopilot.com"
        )

    def test_public_hosts_still_match(self, monkeypatch):
        monkeypatch.setenv("COPILOT_GH_HOST", "acme.ghe.com")
        assert base_url_host_matches(
            "https://api.githubcopilot.com", "api.githubcopilot.com"
        )
        assert base_url_host_matches(
            "https://api.enterprise.githubcopilot.com", "githubcopilot.com"
        )

    def test_lookalike_hosts_still_rejected(self, monkeypatch):
        monkeypatch.setenv("COPILOT_GH_HOST", "acme.ghe.com")
        assert not base_url_host_matches(
            "https://copilot-api.other.ghe.com", "api.githubcopilot.com"
        )
        assert not base_url_host_matches(
            "https://api.githubcopilot.com.evil.example", "api.githubcopilot.com"
        )

    def test_alias_scoped_to_copilot_domains_only(self, monkeypatch):
        monkeypatch.setenv("COPILOT_GH_HOST", "acme.ghe.com")
        assert not base_url_host_matches(
            "https://copilot-api.acme.ghe.com", "api.openai.com"
        )
