"""ArenaHub command-line interface.

Usage::

    arenahub                 # interactive menu
    arenahub chat [-m MODEL] # interactive streaming chat
    arenahub models [--json] # list models available through Arena
    arenahub health          # check configuration and Arena API connectivity
    arenahub serve           # start the OpenAI-compatible local gateway

The same entry points work via ``python -m cli.main ...``.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from backend.arena_client import ArenaClient
from backend.config import Settings, load_settings
from backend.errors import (
    ArenaAuthError,
    ArenaConnectionError,
    ArenaHubError,
    ArenaTimeoutError,
)
from backend.main import create_app
from backend.model_router import ModelRouter
from backend.models import ChatCompletionRequest, ChatMessage, ModelList, extract_delta_text

console = Console()

app = typer.Typer(
    add_completion=False,
    help="ArenaHub — use Arena models from your terminal and local apps.",
    no_args_is_help=False,
)


# ---------------------------------------------------------------------------
# Session history (local persistence)
# ---------------------------------------------------------------------------


class SessionHistory:
    """Append-only JSONL transcript store kept on the local machine."""

    def __init__(self, settings: Settings) -> None:
        self.path: Path = settings.history_dir / "history.jsonl"

    def save(self, model: str, messages: list[dict[str, str]]) -> None:
        if not messages:
            return
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "messages": messages,
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except OSError as exc:  # never crash a chat over transcript IO
            console.print(f"[yellow]Warning: could not save session history: {exc}[/yellow]")


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def render_models_table(models: ModelList) -> Table:
    table = Table(title="Models available through Arena (incl. aliases)", show_lines=False)
    table.add_column("#", style="dim", justify="right")
    table.add_column("Model ID", style="cyan", no_wrap=True)
    table.add_column("Provider / Alias", style="green")
    table.add_column("Created", style="yellow")
    for index, model in enumerate(models.data, start=1):
        created = ""
        if model.created:
            created = datetime.fromtimestamp(model.created, tz=timezone.utc).strftime("%Y-%m-%d")
        provider = f"alias → {model.alias_for}" if model.alias_for else (model.owned_by or "—")
        table.add_row(str(index), model.id, provider, created or "—")
    return table


def prompt_model_choice(models: ModelList, settings: Settings) -> str | None:
    """Show the model table and let the user pick by number or model id."""
    if not models.data:
        console.print("[yellow]No models are available on this account.[/yellow]")
        return None

    console.print(render_models_table(models))
    ids = [m.id for m in models.data]
    default_index: str | None = None
    if settings.arena_default_model in ids:
        default_index = str(ids.index(settings.arena_default_model) + 1)

    while True:
        choice = Prompt.ask(
            "Select a model [bold](number or model id)[/bold]",
            default=default_index or "1",
            show_default=bool(default_index),
        ).strip()
        if choice.isdigit() and 1 <= int(choice) <= len(ids):
            return ids[int(choice) - 1]
        if choice in ids:
            return choice
        console.print(f"[red]Unknown selection {choice!r}. Try a number or a model id.[/red]")


# ---------------------------------------------------------------------------
# Interactive chat
# ---------------------------------------------------------------------------

HELP_TEXT = """\
[bold]Commands[/bold]
  /model [id]   show or switch the active model
  /models       refresh and list available models
  /clear        clear the current conversation context
  /new          archive the conversation and start a new one
  /help         show this help
  /exit         leave chat"""


async def _fetch_models(client: ArenaClient) -> ModelList | None:
    try:
        return await client.list_models()
    except ArenaHubError as exc:
        console.print(f"[red]Could not fetch models: {exc}[/red]")
        return None


async def conversation_loop(
    client: ArenaClient,
    model_router: ModelRouter,
    models: ModelList,
    model: str,
    settings: Settings,
) -> None:
    messages: list[dict[str, str]] = []
    history = SessionHistory(settings)
    current_model = model
    current_models = models

    console.print(
        Panel(
            f"Chatting with [bold cyan]{current_model}[/bold cyan].\n"
            "Type a message and press Enter. [bold]/help[/bold] lists commands.",
            title="ArenaHub Chat",
            border_style="cyan",
        )
    )

    while True:
        try:
            user_input = Prompt.ask("[bold green]You[/bold green]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Leaving chat…[/dim]")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            parts = user_input.split()
            command = parts[0].lower()
            argument = parts[1] if len(parts) > 1 else None

            if command in ("/exit", "/quit"):
                break
            if command == "/help":
                console.print(Panel(HELP_TEXT, border_style="dim"))
                continue
            if command == "/clear":
                messages.clear()
                console.print("[dim]Conversation context cleared.[/dim]")
                continue
            if command == "/new":
                history.save(current_model, messages)
                messages.clear()
                console.print("[dim]Saved the previous session and started a new conversation.[/dim]")
                continue
            if command == "/models":
                refreshed = await _fetch_models(client)
                if refreshed is not None:
                    current_models = refreshed
                    console.print(render_models_table(current_models))
                continue
            if command == "/model":
                if argument:
                    if argument in [m.id for m in current_models.data]:
                        current_model = argument
                        console.print(f"[dim]Switched model to[/dim] [cyan]{current_model}[/cyan]")
                    else:
                        console.print(
                            f"[red]Model {argument!r} is not in the model list.[/red] "
                            "Use [bold]/models[/bold] to refresh."
                        )
                else:
                    chosen = prompt_model_choice(current_models, settings)
                    if chosen:
                        current_model = chosen
                        console.print(f"[dim]Switched model to[/dim] [cyan]{current_model}[/cyan]")
                continue
            console.print(f"[red]Unknown command {command!r}.[/red] Type [bold]/help[/bold].")
            continue

        # Regular message -> stream a completion
        messages.append({"role": "user", "content": user_input})
        try:
            resolved_model = await model_router.resolve(current_model)
        except ArenaHubError as exc:
            console.print(f"\n[red]Model error: {exc}[/red]")
            messages.pop()
            continue
        request = ChatCompletionRequest(
            model=resolved_model,
            messages=[ChatMessage(**m) for m in messages],
            stream=True,
        )

        console.print("[bold blue]Assistant[/bold blue]: ", end="")
        assistant_text = ""
        try:
            async for chunk in client.stream_chat_completion(request):
                delta = extract_delta_text(chunk)
                if delta:
                    assistant_text += delta
                    console.file.write(delta)
                    console.file.flush()
        except ArenaHubError as exc:
            console.print(f"\n[red]Request failed: {exc}[/red]")
            messages.pop()  # drop the unsent user turn so it can be retried
            continue
        finally:
            console.print()

        if assistant_text:
            messages.append({"role": "assistant", "content": assistant_text})

    history.save(current_model, messages)
    console.print(f"[dim]Session transcript saved to {history.path}[/dim]")


async def chat_flow(model_override: str | None) -> bool:
    """Run the interactive chat. Returns False on a fatal setup error."""
    settings = load_settings()
    client = ArenaClient(settings)
    model_router = ModelRouter(settings, client_factory=lambda: ArenaClient(settings))
    try:
        with console.status("Fetching models from Arena…", spinner="dots"):
            try:
                models = await model_router.list_all()
            except ArenaHubError as exc:
                console.print(f"[red]Could not fetch models: {exc}[/red]")
                return False

        if model_override is None:
            chosen = prompt_model_choice(models, settings)
            if chosen is None:
                return False
        else:
            if not await model_router.is_known(model_override):
                console.print(
                    f"[red]Model {model_override!r} is not an available model or alias.[/red]"
                )
                console.print(render_models_table(models))
                return False
            chosen = model_override

        await conversation_loop(client, model_router, models, chosen, settings)
    finally:
        await client.aclose()
    return True


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """ArenaHub — interactive menu when invoked without a subcommand."""
    if ctx.invoked_subcommand is not None:
        return
    interactive_menu()


def interactive_menu() -> None:
    settings = load_settings()
    while True:
        console.print(
            Panel(
                "[bold cyan]ArenaHub[/bold cyan]\n"
                "────────\n"
                "1. Chat\n"
                "2. Models\n"
                "3. Settings\n"
                "4. Exit",
                border_style="cyan",
            )
        )
        choice = Prompt.ask(
            "Select an option",
            choices=["1", "2", "3", "4"],
            default="1",
            show_choices=False,
        )
        if choice == "1":
            asyncio.run(chat_flow(None))
        elif choice == "2":
            asyncio.run(models_flow(as_json=False))
        elif choice == "3":
            show_settings(settings)
        else:
            console.print("[dim]Goodbye.[/dim]")
            return


@app.command()
def chat(
    model: Optional[str] = typer.Option(  # noqa: UP007 - typer needs Optional
        None, "--model", "-m", help="Model id to use (skips the selection prompt)."
    ),
) -> None:
    """Start an interactive streaming chat."""
    if not asyncio.run(chat_flow(model)):
        raise typer.Exit(code=1)


@app.command(name="models")
def models_cmd(
    as_json: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """List models available through the Arena API."""
    if not asyncio.run(models_flow(as_json=as_json)):
        raise typer.Exit(code=1)


async def models_flow(as_json: bool) -> bool:
    """Fetch and render the model list (including aliases). Returns False on failure."""
    settings = load_settings()
    router = ModelRouter(settings, client_factory=lambda: ArenaClient(settings))
    try:
        models = await router.list_all()
    except ArenaHubError as exc:
        console.print(f"[red]Failed to list models: {exc}[/red]")
        return False
    if as_json:
        console.print_json(json.dumps(models.model_dump(exclude_none=True)))
    else:
        console.print(render_models_table(models))
        console.print(f"[dim]{len(models.data)} model(s)/alias(es) available.[/dim]")
    return True


async def health_flow() -> bool:
    """Probe configuration and Arena connectivity. Returns health status."""
    settings = load_settings()
    table = Table(title="ArenaHub health", show_header=False)
    table.add_column(style="bold")
    table.add_column()

    table.add_row("Arena base URL", settings.arena_base_url)
    table.add_row("ARENA_API_KEY", settings.masked_arena_key())
    table.add_row("Default model", settings.arena_default_model or "—")
    table.add_row("Gateway", f"http://{settings.gateway_host}:{settings.gateway_port}")
    table.add_row("Gateway key", "configured" if settings.gateway_api_key else "generated at startup")
    table.add_row("History dir", str(settings.history_dir))

    healthy = False
    async with ArenaClient(settings) as client:
        try:
            models = await client.list_models()
            table.add_row(
                "Arena API",
                f"[green]reachable & authenticated[/green] — {len(models.data)} model(s)",
            )
            healthy = True
        except ArenaAuthError as exc:
            table.add_row(
                "Arena API", f"[red]reachable but authentication failed:[/red] {exc.message}"
            )
        except ArenaTimeoutError:
            table.add_row("Arena API", "[red]timed out — check your network[/red]")
        except ArenaConnectionError as exc:
            table.add_row("Arena API", f"[red]unreachable:[/red] {exc}")
        except ArenaHubError as exc:
            table.add_row("Arena API", f"[red]error:[/red] {exc}")

    console.print(table)
    return healthy


@app.command()
def health() -> None:
    """Check configuration and connectivity to the Arena API."""
    if not asyncio.run(health_flow()):
        raise typer.Exit(code=1)


def show_settings(settings: Settings) -> None:
    table = Table(title="Settings", show_header=False)
    table.add_column(style="bold")
    table.add_column()
    table.add_row("Arena base URL", settings.arena_base_url)
    table.add_row("ARENA_API_KEY", settings.masked_arena_key())
    table.add_row("Default model", settings.arena_default_model or "—")
    table.add_row("Request timeout", f"{settings.arena_timeout:.0f}s")
    table.add_row("Gateway bind", f"{settings.gateway_host}:{settings.gateway_port} (loopback)")
    table.add_row(
        "Gateway API key",
        "set via ARENAHUB_API_KEY" if settings.gateway_api_key else "ephemeral (generated at startup)",
    )
    table.add_row("Session history", str(settings.history_dir / "history.jsonl"))
    console.print(table)
    console.print(
        "[dim]Edit [bold].env[/bold] or your environment to change these values. "
        "API keys are never displayed.[/dim]"
    )


@app.command()
def serve(
    host: Optional[str] = typer.Option(None, "--host", help="Bind host (default 127.0.0.1)."),  # noqa: UP007
    port: Optional[int] = typer.Option(None, "--port", help="Bind port (default 8000)."),  # noqa: UP007
) -> None:
    """Start the OpenAI-compatible local gateway (same as `python -m backend.main`)."""
    import uvicorn

    import os

    settings = load_settings()
    if host:
        settings.gateway_host = host
    if port:
        settings.gateway_port = port
    key_from_env = bool(os.environ.get("ARENAHUB_API_KEY", "").strip())
    gateway_key = settings.ensure_gateway_api_key()

    if settings.gateway_host not in ("127.0.0.1", "localhost", "::1"):
        console.print(
            "[yellow]Warning: binding to a non-loopback host exposes the gateway on your "
            "network. Make sure the gateway API key is set (ARENAHUB_API_KEY).[/yellow]"
        )

    application = create_app(settings)
    console.print(f"[bold cyan]ArenaHub gateway[/bold cyan] on http://{settings.gateway_host}:{settings.gateway_port}")
    if key_from_env:
        console.print("[dim]Gateway key: (from ARENAHUB_API_KEY)[/dim]")
    else:
        console.print(f"[dim]Ephemeral gateway key: {gateway_key}[/dim]")
    uvicorn.run(application, host=settings.gateway_host, port=settings.gateway_port, log_level="info")


if __name__ == "__main__":
    app()
