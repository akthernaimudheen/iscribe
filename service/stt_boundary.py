"""Safe, admin-facing view of the STT provider boundary.

Everything returned here is configuration metadata a hospital's privacy team
may see: provider identity, model, region, endpoint class, retention posture,
capability flags. Never keys, never URLs with credentials, never patient data.
"""

from __future__ import annotations


def stt_policy(settings) -> dict:
    """The effective hospital STT policy (same shape the pipeline enforces)."""
    return {
        "provider": getattr(settings, "stt_provider", "") or None,
        "region": (getattr(settings, "deepgram_region", "global") or "global"),
        "zero_retention": bool(getattr(settings, "deepgram_mip_opt_out", True)),
        "off_host_allowed": True,
        "automatic_fallback_allowed": not bool(
            getattr(settings, "stt_fail_closed", False)),
    }


def _endpoint_class(endpoint_host: str) -> str:
    # Accept a bare host or a full URL tail ("api.in.deepgram.com/v1/listen"):
    # classification is by HOST only, and a path-bearing input must never fall
    # through to "custom endpoint" — that would misstate the residency class.
    host = (endpoint_host or "").lower().split("/")[0]
    if host == "api.deepgram.com" or host == "":
        return "global (no residency guarantee)"
    for code, label in (("api.eu.deepgram.com", "eu"),
                        ("api.au.deepgram.com", "australia"),
                        ("api.in.deepgram.com", "india")):
        if host == code:
            return label
    return "custom endpoint"


def stt_boundary_view(settings) -> dict:
    """Configuration-only boundary view for the active provider.

    Constructs the provider the same way the pipeline would (without
    contacting it) and asks only for its declared capability surface.
    """
    from scribe_engine.stt import get_provider, resolve_production_provider

    policy = stt_policy(settings)
    kwargs = {
        "model_size": settings.whisper_model,
        "device": settings.whisper_device,
        "compute_type": settings.whisper_compute_type,
        "cpu_threads": settings.whisper_cpu_threads,
    }
    try:
        provider, selection_reason = resolve_production_provider(
            settings.stt_provider or None, language=settings.language, **kwargs)
    except Exception as exc:
        return {
            "provider": settings.stt_provider or None,
            "status": "unavailable",
            "reason_type": type(exc).__name__,
            "policy": policy,
        }

    rc = provider.retention_configuration()
    endpoint = rc.get("endpoint") or ""
    return {
        "provider": provider.provider_id,
        "provider_name": provider.provider_name(),
        "status": "configured",
        "selection_reason": selection_reason,
        "model": rc.get("model"),
        "endpoint": endpoint,
        "endpoint_class": _endpoint_class(
            endpoint.split("//")[-1].split("/")[0]),
        "region": policy["region"],
        "data_location": (provider.data_location()
                          if hasattr(provider, "data_location") else "unknown"),
        "off_host": bool(rc.get("audio_sent_off_host")),
        "zero_retention": bool(rc.get("mip_opt_out")) if rc.get("audio_sent_off_host") else True,
        "zero_retention_basis": ("mip_opt_out=true — configured for "
                                 "zero-retention processing according to the "
                                 "provider's documented API behaviour"
                                 if rc.get("mip_opt_out")
                                 else "not configured"),
        "fail_closed": bool(settings.stt_fail_closed),
        "policy_enforced": bool(getattr(settings, "stt_policy_enforced", True)),
        "supports_delete_after_processing": provider.supports_delete_after_processing(),
        "retention_posture": rc.get("retention_behaviour"),
        "capabilities": {
            "diarization": bool(rc.get("diarize")),
            "metadata_sent": rc.get("metadata_sent"),
        },
        "policy": policy,
        "notes": rc.get("notes"),
    }
