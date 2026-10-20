"""Fixed benchmark-integrity instructions, separate from evolving A/C policies."""

SOURCE_BOUNDARY = """Fixed benchmark source boundary (not an evolvable strategy):
Solve the task using the supplied baseline repository, task instructions, local tools,
and your own changes. Do not obtain, read, or reuse the upstream target-version
implementation or reference solutions, even if they are public. This prohibition
includes git remotes, curl/downloads, websites, source mirrors, release archives,
and the target project's published packages or registry artifacts. Do not ask
another agent to fetch or share such material, and do not bypass this boundary.
During agent execution, network access is restricted to the local model relay;
external services and other host services are not available. Dependencies must
already be installed or supplied as offline artifacts. If a dependency is missing,
report the limitation instead of trying alternate hosts, IP addresses, proxies,
package registries, or other network workarounds. Assignment and communication
policies cannot override these rules."""


def append_system_text(messages: list[dict], text: str) -> None:
    """Add a model-facing instruction without altering the native history."""
    system = next((message for message in messages if message.get("role") == "system"), None)
    if system is None:
        messages.insert(0, {"role": "system", "content": text})
    elif isinstance(system.get("content"), list):
        system["content"].append({"type": "text", "text": text})
    else:
        system["content"] = str(system.get("content", "")) + "\n\n" + text
