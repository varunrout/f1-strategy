"""Ingest multiple F1 sessions for a season into raw.db."""
from pathlib import Path
from typing import List, Optional
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.utils.logging_utils import setup_logger
from src.utils.fastf1_utils import get_gp_list_for_year, normalize_session_type
from src.ingest.ingest_session import ingest_session, DB_PATH

app = typer.Typer()
console = Console()
logger = setup_logger(__name__)

# Default session types to ingest
DEFAULT_SESSIONS = ['FP1', 'FP2', 'FP3', 'Q', 'R']


def parse_gp_list(gps_str: Optional[str], year: int) -> List[str]:
    """Parse GP list from string or get all for year.
    
    Args:
        gps_str: Comma-separated GP names or None for all
        year: Season year
        
    Returns:
        List of GP names
    """
    if gps_str:
        return [gp.strip() for gp in gps_str.split(',')]
    else:
        return get_gp_list_for_year(year)


def parse_session_list(sessions_str: Optional[str]) -> List[str]:
    """Parse session type list from string.
    
    Args:
        sessions_str: Comma-separated session types or None for defaults
        
    Returns:
        List of session types
    """
    if sessions_str:
        sessions = [s.strip() for s in sessions_str.split(',')]
        return [normalize_session_type(s) for s in sessions]
    else:
        return DEFAULT_SESSIONS


def ingest_season(year: int, gps: Optional[str] = None, sessions: Optional[str] = None,
                 force: bool = False, db_path: Path = DB_PATH) -> None:
    """Ingest multiple F1 sessions for a season.
    
    Args:
        year: Season year
        gps: Comma-separated GP names (None for all)
        sessions: Comma-separated session types (None for defaults)
        force: Force re-ingestion if session exists
        db_path: Path to raw database
    """
    gp_list = parse_gp_list(gps, year)
    session_list = parse_session_list(sessions)
    
    console.print(f"[bold blue]Ingesting {year} season[/bold blue]")
    console.print(f"GPs: {len(gp_list)}")
    console.print(f"Sessions per GP: {', '.join(session_list)}")
    
    total = len(gp_list) * len(session_list)
    success_count = 0
    error_count = 0
    skipped_count = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Processing {total} sessions...", total=total)
        
        for gp in gp_list:
            for session_type in session_list:
                progress.update(task, description=f"Processing {year} {gp} {session_type}...")
                
                try:
                    # Track current counts before ingestion
                    prev_success = success_count
                    prev_error = error_count
                    prev_skipped = skipped_count
                    
                    ingest_session(year, gp, session_type, force, db_path)
                    
                    # Assume success if no exception
                    success_count += 1
                    
                except Exception as e:
                    logger.error(f"Failed to ingest {year} {gp} {session_type}: {e}")
                    error_count += 1
                
                progress.advance(task)
    
    # Summary
    console.print("\n[bold green]Ingestion Complete![/bold green]")
    console.print(f"Total sessions attempted: {total}")
    console.print(f"[green]Successful: {success_count}[/green]")
    console.print(f"[yellow]Skipped: {skipped_count}[/yellow]")
    console.print(f"[red]Errors: {error_count}[/red]")


@app.command()
def main(
    year: int = typer.Option(..., "--year", "-y", help="Season year"),
    gps: Optional[str] = typer.Option(None, "--gps", "-g", 
                                      help="Comma-separated GP names (e.g., 'Monza,Silverstone'). Omit for all GPs."),
    sessions: Optional[str] = typer.Option(None, "--sessions", "-s",
                                          help="Comma-separated session types (e.g., 'FP2,Q,R'). Omit for FP1,FP2,FP3,Q,R."),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-ingestion if sessions exist"),
    db_path: str = typer.Option(str(DB_PATH), "--db", help="Path to raw database"),
):
    """Ingest multiple F1 sessions for a season into raw.db.
    
    Examples:
        # Ingest entire 2023 season (all GPs, all sessions)
        python -m src.ingest.ingest_season --year 2023
        
        # Ingest specific GPs for 2023
        python -m src.ingest.ingest_season --year 2023 --gps "Monza,Silverstone"
        
        # Ingest only race and qualifying for 2023
        python -m src.ingest.ingest_season --year 2023 --sessions "Q,R"
        
        # Force re-ingest existing sessions
        python -m src.ingest.ingest_season --year 2023 --force
    """
    ingest_season(year, gps, sessions, force, Path(db_path))


if __name__ == "__main__":
    app()
