# cli.py
import os, json, requests, typer

APP   = os.environ.get("APP", "https://rob-brain-api-1.onrender.com")
TOKEN = os.environ.get("TOKEN", "")
if not TOKEN:
    raise SystemExit("Set TOKEN env var to your ROB_BRAIN_TOKEN")

HEADERS = {"Authorization": f"Bearer {TOKEN}"}
app = typer.Typer(help="Rob Brain API CLI")

@app.command()
def health():
    r = requests.get(f"{APP}/healthz", headers=HEADERS, timeout=60)
    typer.echo(json.dumps(r.json(), indent=2))

@app.command()
def short_pnl(start: str, end: str, file: str, top_k: int = 5000):
    params = {"start_date": start, "end_date": end, "file": file, "top_k": top_k}
    r = requests.get(f"{APP}/short_pnl", headers=HEADERS, params=params, timeout=120)
    typer.echo(json.dumps(r.json(), indent=2))

@app.command()
def breakdown(
    start: str,
    end: str,
    file: str,
    symbol: str = "",
    allow_carry: bool = False,
    top_k: int = 8000,
    limit: int = 100000,
):
    params = {
        "start_date": start, "end_date": end, "file": file,
        "symbol": (symbol or None),
        "allow_carry": allow_carry, "top_k": top_k, "limit": limit,
    }
    r = requests.get(f"{APP}/short_pnl_breakdown", headers=HEADERS, params=params, timeout=120)
    typer.echo(json.dumps(r.json(), indent=2))

@app.command()
def fees(start: str, end: str, file: str, top_k: int = 8000):
    body = {"file": file, "start": start, "end": end, "top_k": top_k}
    r = requests.post(f"{APP}/fees", headers=HEADERS, json=body, timeout=120)
    typer.echo(json.dumps(r.json(), indent=2))

if __name__ == "__main__":
    app()
