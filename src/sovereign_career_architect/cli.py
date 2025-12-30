"""Command-line interface for Sovereign Career Architect."""

import typer
from rich.console import Console
from rich.table import Table
from typing import Optional

from sovereign_career_architect.config import settings

app = typer.Typer(
    name="sca",
    help="Sovereign Career Architect - Autonomous AI agent for career navigation",
    add_completion=False,
)
console = Console()


@app.command()
def serve(
    host: str = typer.Option(settings.host, help="Host to bind to"),
    port: int = typer.Option(settings.port, help="Port to bind to"),
    reload: bool = typer.Option(settings.reload, help="Enable auto-reload"),
) -> None:
    """Start the API server."""
    import uvicorn
    
    console.print(f"🚀 Starting Sovereign Career Architect on {host}:{port}")
    uvicorn.run(
        "sovereign_career_architect.api.main:app",
        host=host,
        port=port,
        reload=reload,
    )


@app.command()
def config() -> None:
    """Show current configuration."""
    table = Table(title="Sovereign Career Architect Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    # Show non-sensitive settings
    table.add_row("Debug Mode", str(settings.debug))
    table.add_row("Log Level", settings.log_level)
    table.add_row("Reasoning Model", settings.reasoning_model)
    table.add_row("Vision Model", settings.vision_model)
    table.add_row("Voice Model", settings.voice_model)
    table.add_row("Vector Store", settings.vector_store_type)
    table.add_row("Browser Headless", str(settings.browser_headless))
    table.add_row("Browser Stealth", str(settings.browser_stealth))
    table.add_row("Max Retries", str(settings.max_retries))
    
    console.print(table)


@app.command()
def test_setup() -> None:
    """Test the setup and configuration."""
    console.print("🔍 Testing Sovereign Career Architect setup...")
    
    # Test API keys
    if settings.openai_api_key:
        console.print("✅ OpenAI API key configured")
    else:
        console.print("❌ OpenAI API key missing")
    
    if settings.groq_api_key:
        console.print("✅ Groq API key configured")
    else:
        console.print("❌ Groq API key missing")
    
    # Test optional keys
    optional_keys = [
        ("Mem0", settings.mem0_api_key),
        ("Vapi.ai", settings.vapi_api_key),
        ("Sarvam-1", settings.sarvam_api_key),
    ]
    
    for name, key in optional_keys:
        if key:
            console.print(f"✅ {name} API key configured")
        else:
            console.print(f"⚠️  {name} API key not configured (optional)")
    
    console.print("\n🎯 Setup test complete!")


@app.command()
def version() -> None:
    """Show version information."""
    from sovereign_career_architect import __version__
    console.print(f"Sovereign Career Architect v{__version__}")


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()