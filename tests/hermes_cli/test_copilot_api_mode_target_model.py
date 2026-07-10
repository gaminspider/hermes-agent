"""Regression tests for Copilot api_mode derivation from the target model.

``_copilot_runtime_api_mode`` used to compute the API mode from the
persisted ``model.default`` even when the caller was resolving for an
explicit target model (one-shot ``hermes -z -m gpt-5.5 --provider copilot``
or a mid-session switch). With a non-Copilot config default (e.g. a Claude
model on the anthropic provider), GPT-5.x requests were derived as
``chat_completions`` and sent to ``/chat/completions``, which the Copilot
API rejects for GPT-5.x — the run failed with "no final response was
produced".

The fix threads ``target_model`` through the credential-pool and api-key
resolution paths and prefers it over the config default, mirroring the
re-derivation already done for opencode-zen/go and Azure Foundry.
"""

from unittest.mock import patch

from hermes_cli.runtime_provider import _copilot_runtime_api_mode


_STALE_ANTHROPIC_CFG = {
    "default": "claude-opus-4-1-20250805",
    "provider": "anthropic",
}


class TestCopilotApiModeTargetModel:
    def test_target_model_overrides_stale_default(self):
        """A GPT-5.x target must derive codex_responses even when the
        persisted default is a non-Copilot Claude model."""
        with patch(
            "hermes_cli.models.copilot_model_api_mode",
            return_value="codex_responses",
        ) as mock_mode:
            result = _copilot_runtime_api_mode(
                _STALE_ANTHROPIC_CFG, "", target_model="gpt-5.5"
            )
        assert result == "codex_responses"
        mock_mode.assert_called_once_with("gpt-5.5", api_key="")

    def test_without_target_model_uses_config_default(self):
        """No target model → existing behaviour: derive from model.default."""
        with patch(
            "hermes_cli.models.copilot_model_api_mode",
            return_value="chat_completions",
        ) as mock_mode:
            result = _copilot_runtime_api_mode(_STALE_ANTHROPIC_CFG, "")
        assert result == "chat_completions"
        mock_mode.assert_called_once_with(
            "claude-opus-4-1-20250805", api_key=""
        )

    def test_persisted_api_mode_honored_without_target_model(self):
        """A persisted api_mode from the same provider family is still
        honored when no explicit target model is in play."""
        cfg = {
            "default": "claude-sonnet-4.6",
            "provider": "copilot",
            "api_mode": "anthropic_messages",
        }
        assert _copilot_runtime_api_mode(cfg, "") == "anthropic_messages"

    def test_persisted_api_mode_skipped_for_target_model(self):
        """A persisted api_mode belonging to the previous model must not
        leak onto an explicit switch to a different model."""
        cfg = {
            "default": "claude-sonnet-4.6",
            "provider": "copilot",
            "api_mode": "anthropic_messages",
        }
        with patch(
            "hermes_cli.models.copilot_model_api_mode",
            return_value="codex_responses",
        ) as mock_mode:
            result = _copilot_runtime_api_mode(cfg, "", target_model="gpt-5.5")
        assert result == "codex_responses"
        mock_mode.assert_called_once_with("gpt-5.5", api_key="")

    def test_empty_model_falls_back_to_chat_completions(self):
        assert _copilot_runtime_api_mode({}, "") == "chat_completions"
        assert _copilot_runtime_api_mode({}, "", target_model="") == "chat_completions"
