"""Background job runner for long-running F1 data ingestion.

Supports running ingestion jobs in the background with:
- Progress tracking via log files
- Job status monitoring
- Parallel session ingestion (optional)
"""
import json
import subprocess
import sys
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

from src.utils.logging_utils import setup_logger
from src.utils.fastf1_utils import get_gp_list_for_year, normalize_session_type

app = typer.Typer()
console = Console()
logger = setup_logger(__name__)

# Job tracking directory
JOBS_DIR = Path(__file__).parent.parent.parent / "data" / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

# Default session types (including Sprint weekends)
# SQ = Sprint Qualifying (Sprint Shootout in 2024)
# S = Sprint Race
DEFAULT_SESSIONS = ['FP1', 'FP2', 'FP3', 'Q', 'R', 'SQ', 'S']


class JobTracker:
    """Track background job progress and status."""
    
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.job_file = JOBS_DIR / f"{job_id}.json"
        self.log_file = JOBS_DIR / f"{job_id}.log"
    
    def create(self, total_sessions: int, params: Dict[str, Any]) -> None:
        """Create a new job record."""
        job_data = {
            'job_id': self.job_id,
            'status': 'running',
            'created_at': datetime.now().isoformat(),
            'total_sessions': total_sessions,
            'completed': 0,
            'failed': 0,
            'params': params,
            'results': []
        }
        self._write(job_data)
    
    def update(self, completed: Optional[int] = None, failed: Optional[int] = None, 
               result: Optional[Dict] = None, status: Optional[str] = None) -> None:
        """Update job progress."""
        data = self._read()
        if completed is not None:
            data['completed'] = completed
        if failed is not None:
            data['failed'] = failed
        if result:
            data['results'].append(result)
        if status:
            data['status'] = status
        data['updated_at'] = datetime.now().isoformat()
        self._write(data)
    
    def finish(self, status: str = 'completed') -> None:
        """Mark job as finished."""
        data = self._read()
        data['status'] = status
        data['finished_at'] = datetime.now().isoformat()
        self._write(data)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current job status."""
        return self._read()
    
    def _read(self) -> Dict[str, Any]:
        if self.job_file.exists():
            return json.loads(self.job_file.read_text())
        return {}
    
    def _write(self, data: Dict[str, Any]) -> None:
        self.job_file.write_text(json.dumps(data, indent=2))


def ingest_single_session(args: tuple) -> Dict[str, Any]:
    """Worker function to ingest a single session.
    
    Args:
        args: Tuple of (year, gp_name, session_type, force, lake_path)
        
    Returns:
        Result dict with status
    """
    year, gp_name, session_type, force, lake_path = args
    
    try:
        from src.ingest.ingest_parquet import ingest_session_parquet
        success, message = ingest_session_parquet(
            year, gp_name, session_type, force, Path(lake_path)
        )
        return {
            'year': year,
            'gp': gp_name,
            'session': session_type,
            'success': success,
            'message': message
        }
    except Exception as e:
        return {
            'year': year,
            'gp': gp_name,
            'session': session_type,
            'success': False,
            'message': str(e)
        }


def run_ingestion_batch(
    year: int,
    gp_list: List[str],
    session_types: List[str],
    force: bool = False,
    lake_path: Optional[Path] = None,
    max_workers: int = 4,
    job_tracker: Optional[JobTracker] = None
) -> Dict[str, int]:
    """Run batch ingestion with optional parallelism.
    
    Args:
        year: Season year
        gp_list: List of GP names
        session_types: List of session types
        force: Force re-ingestion
        lake_path: Data lake path
        max_workers: Number of parallel workers
        job_tracker: Optional job tracker for progress
        
    Returns:
        Dict with success/failure counts
    """
    from src.utils.parquet_writer import DATA_LAKE_PATH
    lake_path = lake_path or DATA_LAKE_PATH
    
    # Build list of all sessions to ingest
    tasks = []
    for gp in gp_list:
        for session_type in session_types:
            tasks.append((year, gp, session_type, force, str(lake_path)))
    
    total = len(tasks)
    completed = 0
    failed = 0
    
    console.print(f"[bold]Starting ingestion of {total} sessions with {max_workers} workers[/bold]")
    
    if max_workers == 1:
        # Sequential processing
        for task in tasks:
            result = ingest_single_session(task)
            if result['success']:
                completed += 1
            else:
                failed += 1
            
            if job_tracker:
                job_tracker.update(completed=completed, failed=failed, result=result)
            
            console.print(f"[{'green' if result['success'] else 'red'}]{result['year']} {result['gp']} {result['session']}: {result['message'][:50]}[/]")
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(ingest_single_session, task): task for task in tasks}
            
            for future in as_completed(futures):
                result = future.result()
                if result['success']:
                    completed += 1
                else:
                    failed += 1
                
                if job_tracker:
                    job_tracker.update(completed=completed, failed=failed, result=result)
                
                console.print(f"[{'green' if result['success'] else 'red'}][{completed+failed}/{total}] {result['year']} {result['gp']} {result['session']}[/]")
    
    return {'completed': completed, 'failed': failed, 'total': total}


def start_background_job(
    year: int,
    gps: Optional[str],
    sessions: Optional[str],
    force: bool,
    lake_path: str,
    max_workers: int
) -> str:
    """Start ingestion as a background process.
    
    Returns:
        Job ID for tracking
    """
    job_id = f"ingest_{year}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Build command to run
    cmd = [
        sys.executable, "-m", "src.ingest.background_runner",
        "run-job",
        "--year", str(year),
        "--job-id", job_id,
        "--workers", str(max_workers),
        "--lake", lake_path,
    ]
    
    if gps:
        cmd.extend(["--gps", gps])
    if sessions:
        cmd.extend(["--sessions", sessions])
    if force:
        cmd.append("--force")
    
    # Start background process
    log_file = JOBS_DIR / f"{job_id}.log"
    
    if sys.platform == 'win32':
        # Windows: use CREATE_NO_WINDOW flag
        CREATE_NO_WINDOW = 0x08000000
        with open(log_file, 'w') as log:
            process = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=CREATE_NO_WINDOW,
                cwd=Path(__file__).parent.parent.parent
            )
    else:
        # Unix: use nohup-like behavior
        with open(log_file, 'w') as log:
            process = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                cwd=Path(__file__).parent.parent.parent
            )
    
    console.print(f"[bold green]Started background job: {job_id}[/bold green]")
    console.print(f"PID: {process.pid}")
    console.print(f"Log file: {log_file}")
    console.print(f"\nCheck status with: python -m src.ingest.background_runner status --job-id {job_id}")
    
    return job_id


@app.command("start")
def start_cmd(
    year: int = typer.Option(..., "--year", "-y", help="Season year"),
    gps: Optional[str] = typer.Option(None, "--gps", "-g", help="Comma-separated GP names"),
    sessions: Optional[str] = typer.Option(None, "--sessions", "-s", help="Comma-separated sessions"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-ingestion"),
    lake_path: str = typer.Option(None, "--lake", help="Data lake path"),
    workers: int = typer.Option(4, "--workers", "-w", help="Number of parallel workers"),
    background: bool = typer.Option(True, "--background/--foreground", help="Run in background"),
):
    """Start a batch ingestion job (background by default)."""
    from src.utils.parquet_writer import DATA_LAKE_PATH
    lake_path = lake_path or str(DATA_LAKE_PATH)
    
    if background:
        start_background_job(year, gps, sessions, force, lake_path, workers)
    else:
        # Run in foreground
        gp_list = [g.strip() for g in gps.split(',')] if gps else get_gp_list_for_year(year)
        session_list = [normalize_session_type(s.strip()) for s in sessions.split(',')] if sessions else DEFAULT_SESSIONS
        
        result = run_ingestion_batch(year, gp_list, session_list, force, Path(lake_path), workers)
        console.print(f"\n[bold]Completed: {result['completed']}, Failed: {result['failed']}, Total: {result['total']}[/bold]")


@app.command("run-job")
def run_job_cmd(
    year: int = typer.Option(..., "--year"),
    job_id: str = typer.Option(..., "--job-id"),
    gps: Optional[str] = typer.Option(None, "--gps"),
    sessions: Optional[str] = typer.Option(None, "--sessions"),
    force: bool = typer.Option(False, "--force"),
    lake_path: str = typer.Option(None, "--lake"),
    workers: int = typer.Option(4, "--workers"),
):
    """Internal command to run a job (called by background process)."""
    from src.utils.parquet_writer import DATA_LAKE_PATH
    lake_path = lake_path or str(DATA_LAKE_PATH)
    
    gp_list = [g.strip() for g in gps.split(',')] if gps else get_gp_list_for_year(year)
    session_list = [normalize_session_type(s.strip()) for s in sessions.split(',')] if sessions else DEFAULT_SESSIONS
    
    total_sessions = len(gp_list) * len(session_list)
    
    # Create job tracker
    tracker = JobTracker(job_id)
    tracker.create(total_sessions, {
        'year': year,
        'gps': gp_list,
        'sessions': session_list,
        'force': force
    })
    
    try:
        result = run_ingestion_batch(
            year, gp_list, session_list, force, Path(lake_path), workers, tracker
        )
        tracker.finish('completed' if result['failed'] == 0 else 'completed_with_errors')
    except Exception as e:
        logger.error(f"Job failed: {e}")
        tracker.finish('failed')
        raise


@app.command("status")
def status_cmd(
    job_id: Optional[str] = typer.Option(None, "--job-id", "-j", help="Specific job ID"),
):
    """Check status of background jobs."""
    if job_id:
        # Show specific job
        tracker = JobTracker(job_id)
        status = tracker.get_status()
        
        if not status:
            console.print(f"[red]Job {job_id} not found[/red]")
            return
        
        console.print(f"\n[bold]Job: {job_id}[/bold]")
        console.print(f"Status: {status.get('status', 'unknown')}")
        console.print(f"Progress: {status.get('completed', 0)}/{status.get('total_sessions', '?')} completed")
        console.print(f"Failed: {status.get('failed', 0)}")
        console.print(f"Created: {status.get('created_at', 'N/A')}")
        
        if status.get('finished_at'):
            console.print(f"Finished: {status['finished_at']}")
        
        # Show recent results
        results = status.get('results', [])[-5:]
        if results:
            console.print("\n[bold]Recent sessions:[/bold]")
            for r in results:
                icon = "✓" if r.get('success') else "✗"
                console.print(f"  {icon} {r.get('year')} {r.get('gp')} {r.get('session')}")
    else:
        # List all jobs
        job_files = list(JOBS_DIR.glob("*.json"))
        
        if not job_files:
            console.print("[yellow]No jobs found[/yellow]")
            return
        
        table = Table(title="Ingestion Jobs")
        table.add_column("Job ID")
        table.add_column("Status")
        table.add_column("Progress")
        table.add_column("Created")
        
        for job_file in sorted(job_files, reverse=True)[:10]:
            data = json.loads(job_file.read_text())
            progress = f"{data.get('completed', 0)}/{data.get('total_sessions', '?')}"
            table.add_row(
                data.get('job_id', 'N/A'),
                data.get('status', 'unknown'),
                progress,
                data.get('created_at', 'N/A')[:19]
            )
        
        console.print(table)


@app.command("logs")
def logs_cmd(
    job_id: str = typer.Option(..., "--job-id", "-j", help="Job ID to view logs"),
    tail: int = typer.Option(50, "--tail", "-n", help="Number of lines to show"),
):
    """View logs for a background job."""
    log_file = JOBS_DIR / f"{job_id}.log"
    
    if not log_file.exists():
        console.print(f"[red]Log file not found for job {job_id}[/red]")
        return
    
    with open(log_file) as f:
        lines = f.readlines()
        for line in lines[-tail:]:
            console.print(line.rstrip())


@app.command("cancel")
def cancel_cmd(
    job_id: str = typer.Option(..., "--job-id", "-j", help="Job ID to cancel"),
):
    """Cancel a running job (marks as cancelled, process may continue)."""
    tracker = JobTracker(job_id)
    status = tracker.get_status()
    
    if not status:
        console.print(f"[red]Job {job_id} not found[/red]")
        return
    
    if status.get('status') == 'running':
        tracker.finish('cancelled')
        console.print(f"[yellow]Job {job_id} marked as cancelled[/yellow]")
        console.print("[dim]Note: The background process may still be running. Use Task Manager to kill it if needed.[/dim]")
    else:
        console.print(f"[yellow]Job {job_id} is not running (status: {status.get('status')})[/yellow]")


if __name__ == "__main__":
    app()
