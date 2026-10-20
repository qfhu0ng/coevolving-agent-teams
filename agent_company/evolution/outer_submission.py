"""Provider-compatible tool choice for the bounded outer-loop submission phase."""

from urllib.parse import urlsplit

from agent_company.runtime.relay import ProviderConfig


def submission_tool_choice(payload: dict, provider: ProviderConfig, submit: str) -> None:
    """Keep named forcing except for documented auto-only provider modes.

    With tools present, omitting tool_choice is DeepSeek's documented auto mode.
    The caller still supplies only the submit tool, validates its fields and
    enforces its existing context/output/round limits; auto adds no retry budget.
    See https://api-docs.deepseek.com/api/create-chat-completion/ .
    """
    model = provider.model.rsplit("/", 1)[-1].lower()
    hostname = urlsplit(provider.base_url).hostname
    # GLM-5.3 function calling supports only auto, including the final turn.
    # Keep this scoped to verified official endpoints/model, not arbitrary proxies.
    # https://docs.z.ai/guides/capabilities/function-calling
    if model == "glm-5.3" and hostname in {"open.bigmodel.cn", "api.z.ai"}:
        payload.pop("tool_choice", None)  # Documented default auto with tools.
        return
    official = hostname == "api.deepseek.com"
    deepseek = (official
                or model in {"deepseek-flash", "deepseek-v4-flash", "deepseek-v4-pro",
                             "deepseek-chat", "deepseek-reasoner"})
    thinking = payload.get("thinking")
    mode = thinking.get("type") if isinstance(thinking, dict) else None
    effort = payload.get("reasoning_effort")
    # Current official DeepSeek models default to thinking. Do not infer a
    # third-party endpoint's default from its model alias alone. An off switch
    # retains named forcing; if switches conflict, auto is compatible with
    # either interpretation without changing or guessing their precedence.
    enabled = mode == "enabled" or effort in {"low", "high", "max", "minimal", "medium", "xhigh", "ultra"}
    disabled = mode == "disabled" or effort == "none"
    if deepseek and (enabled or (official and not disabled)):
        payload.pop("tool_choice", None)
    else:
        payload["tool_choice"] = {"type": "function", "function": {"name": submit}}
