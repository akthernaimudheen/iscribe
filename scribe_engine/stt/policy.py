"""STT policy enforcement at the provider boundary.

The hospital's STT policy is evaluated BEFORE any audio leaves our
infrastructure. If the active provider's declared boundary does not satisfy
the policy, the audio is not sent — the refusal is explicit and auditable
(``STT_POLICY_BLOCKED``). When fail-closed mode is enabled, a provider
failure never falls back to another provider: the job fails safely, the
source audio stays under the normal retention lifecycle, and the failure is
audited.

Evidence rule (docs/stt_provider_privacy_and_security.md): this module only
reasons over what a provider DECLARED through its capability surface
(retention_configuration / data_location). Nothing here claims contractual
deletion, compliance, or residency guarantees — it enforces configuration
consistency, nothing more.
"""

from __future__ import annotations


class STTPolicyBlocked(RuntimeError):
    """The hospital's STT policy forbids sending this audio to the provider.

    Raised BEFORE any audio leaves the infrastructure. The message is written
    for operators: it names the violated requirement and never contains keys,
    audio, transcript or patient data.
    """

    def __init__(self, reason: str, message: str, provider_id: str | None = None,
                 required: dict | None = None):
        self.reason = reason
        self.provider_id = provider_id
        self.required = dict(required or {})
        super().__init__(message)


def check_policy(provider, policy: dict) -> None:
    """Evaluate the hospital's STT policy against the provider's declaration.

    Raises :class:`STTPolicyBlocked` when a requirement is violated. The
    provider is never contacted — only its capability surface is read.

    Policy keys (all optional; absent = not required):
      provider        exact provider id required (e.g. "deepgram")
      region          required processing region (e.g. "in", "eu", "au")
      zero_retention  True => provider must declare MIP opt-out / ZDR
      off_host_allowed True => off-host processing is permitted at all
      automatic_fallback_allowed  True => fallback may engage on failure
    """
    rc = provider.retention_configuration()

    required_provider = policy.get("provider")
    if required_provider and provider.provider_id != required_provider:
        raise STTPolicyBlocked(
            "provider_mismatch",
            f"STT policy requires provider '{required_provider}' but the "
            f"active provider is '{provider.provider_id}'. Audio was not sent.",
            provider_id=provider.provider_id, required=policy,
        )

    if policy.get("off_host_allowed") is False and rc.get("audio_sent_off_host"):
        raise STTPolicyBlocked(
            "off_host_not_allowed",
            "STT policy forbids sending audio off-host, but the active "
            "provider processes audio off-host. Audio was not sent.",
            provider_id=provider.provider_id, required=policy,
        )

    # Zero retention is about off-host processors: a provider that never
    # receives the audio satisfies it trivially (data stays on our host).
    if policy.get("zero_retention") and rc.get("audio_sent_off_host") \
            and not rc.get("mip_opt_out"):
        raise STTPolicyBlocked(
            "zero_retention_required",
            "STT policy requires zero-retention processing (mip_opt_out=true), "
            "but the provider is configured to participate in the Model "
            "Improvement Program. Audio was not sent.",
            provider_id=provider.provider_id, required=policy,
        )

    required_region = policy.get("region")
    if required_region:
        location = provider.data_location() if hasattr(provider, "data_location") \
            else "unknown"
        # Exact, declared matches only: map the region code to the location
        # name the provider must declare. "global" (no residency guarantee)
        # never satisfies a regional requirement.
        _REGION_NAMES = {"in": "india", "eu": "eu", "au": "australia"}
        expected = _REGION_NAMES.get(str(required_region).lower(),
                                     str(required_region).lower())
        if not str(location).lower().startswith(expected):
            raise STTPolicyBlocked(
                "region_mismatch",
                f"STT policy requires processing in region '{required_region}' "
                f"but the provider declares '{location}'. Audio was not sent.",
                provider_id=provider.provider_id, required=policy,
            )


def stt_policy(settings) -> dict:
    """Build the effective policy from the service settings object."""
    return {
        "provider": getattr(settings, "stt_provider", "") or None,
        "region": (getattr(settings, "deepgram_region", "global") or "global"),
        "zero_retention": bool(getattr(settings, "deepgram_mip_opt_out", True)),
        "off_host_allowed": True,
        "automatic_fallback_allowed": not bool(
            getattr(settings, "stt_fail_closed", False)),
    }
