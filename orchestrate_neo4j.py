#!/usr/bin/env python3
import argparse
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Callable

from neo4j import GraphDatabase, basic_auth
from neo4j.exceptions import ServiceUnavailable


# NEW imports
import re
import psutil

@dataclass
class DbSpec:
    name: str
    home: str                 # Path to the DBMS home that contains bin/neo4j
    bolt_uri: str             # e.g., bolt://localhost:7687
    user: str = "neo4j"
    password: str = "neo4j"
    start_timeout: int = 120  # seconds to wait for startup
    stop_timeout: int = 60    # seconds to wait for shutdown

def _neo4j_executable(home: str) -> List[str]:
    is_windows = platform.system().lower().startswith("win")
    exe = os.path.join(home, "bin", "neo4j.bat" if is_windows else "neo4j")
    if not os.path.exists(exe):
        raise FileNotFoundError(f"Neo4j launcher not found: {exe}")
    return [exe]

def _run(cmd: List[str], env: Optional[dict] = None, check: bool = True) -> subprocess.CompletedProcess:
    # Use a shell-less invocation; capture output for diagnostics
    try:
        cp = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            check=check
        )
        return cp
    except subprocess.CalledProcessError as e:
        print(e.stdout, file=sys.stderr)
        raise

def start_dbms(db: DbSpec) -> None:
    print(f"\n=== Starting DBMS '{db.name}' from {db.home} ===")
    exe = _neo4j_executable(db.home)
    # 'neo4j start' returns immediately; server continues in background.
    out = _run(exe + ["start"]).stdout
    print(out.strip())

def stop_dbms(db: DbSpec) -> None:
    print(f"\n=== Stopping DBMS '{db.name}' ===")
    exe = _neo4j_executable(db.home)
    try:
        out = _run(exe + ["stop"]).stdout
        print(out.strip())
    except subprocess.CalledProcessError as e:
        # If it's already stopped, ignore; else re-raise
        msg = (e.stdout or "").lower()
        if "not running" in msg or "no running" in msg:
            print("DBMS was not running.")
        else:
            raise

def wait_for_bolt(bolt_uri: str, user: str, password: str, timeout_s: int) -> None:
    print(f"Waiting for Bolt at {bolt_uri} (timeout {timeout_s}s)...")
    deadline = time.time() + timeout_s
    last_err = None
    while time.time() < deadline:
        try:
            driver = GraphDatabase.driver(bolt_uri, auth=basic_auth(user, password))
            with driver.session(database=None) as s:
                s.run("RETURN 1").consume()
            driver.close()
            print("Bolt is up.")
            return
        except Exception as e:
            last_err = e
            time.sleep(1.0)
    raise TimeoutError(f"Bolt did not become available within {timeout_s}s. Last error: {last_err}")

def wait_for_bolt_down(bolt_uri: str, timeout_s: int) -> None:
    print(f"Waiting for Bolt to go down at {bolt_uri} (timeout {timeout_s}s)...")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            # If connection succeeds, it's still up
            driver = GraphDatabase.driver(bolt_uri, auth=basic_auth("neo4j","neo4j"))
            with driver.session(database=None) as s:
                s.run("RETURN 1").consume()
            driver.close()
            time.sleep(1.0)
        except Exception:
            print("Bolt is down.")
            return
    raise TimeoutError("Server did not shut down in time.")

def run_tests(db: DbSpec, test_fn: Callable[[GraphDatabase], None]) -> None:
    print(f"\n--- Running tests on '{db.name}' ---")
    driver = GraphDatabase.driver(db.bolt_uri, auth=basic_auth(db.user, db.password))
    try:
        test_fn(driver)
    finally:
        driver.close()

def cycle_databases(dbs: List[DbSpec], test_fn: Callable[[GraphDatabase], None]) -> None:
    for i, db in enumerate(dbs, 1):
        print(f"\n########## [{i}/{len(dbs)}] {db.name} ##########")
        start_dbms(db)
        wait_for_bolt(db.bolt_uri, db.user, db.password, db.start_timeout)
        try:
            run_tests(db, test_fn)
        finally:
            stop_dbms(db)
            # Give it a moment to shut down; then verify Bolt is down
            try:
                wait_for_bolt_down(db.bolt_uri, db.stop_timeout)
            except TimeoutError as e:
                print(f"Warning: {e}", file=sys.stderr)

# --- Example test function you can customize ---
def example_tests(driver: GraphDatabase) -> None:
    # 1) Simple smoke check
    with driver.session(database=None) as s:
        v = s.run("RETURN 1 AS ok").single()["ok"]
        print(f"Smoke test: ok={v}")
    # 2) Add your real test logic here, e.g. run migrations, load fixtures, run assertions, etc.
    # with driver.session(database='neo4j') as s:
    #     s.run("MATCH (n) RETURN count(n) AS c").single()

def parse_db_arg(db_arg: str) -> DbSpec:
    """
    --db name=alpha,home=/path/to/dbms,bolt=bolt://localhost:7687,user=neo4j,pass=secret
    Only name,home,bolt are required.
    """
    kv = {}
    for part in db_arg.split(","):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        kv[k.strip()] = v.strip()
    required = ["name", "home", "bolt"]
    for r in required:
        if r not in kv:
            raise ValueError(f"Missing '{r}' in --db spec: {db_arg}")
    return DbSpec(
        name=kv["name"],
        home=kv["home"],
        bolt_uri=kv["bolt"],
        user=kv.get("user", "neo4j"),
        password=kv.get("pass", "neo4j"),
        start_timeout=int(kv.get("start_timeout", "120")),
        stop_timeout=int(kv.get("stop_timeout", "60")),
    )




# --- NEW: discover the running DBMS home from processes ---
def detect_running_dbms_home() -> Optional[str]:
    """
    Try to locate the home directory of the currently running Neo4j DBMS
    (as launched by Neo4j Desktop) by inspecting process command lines
    for -Dneo4j.home=... . Returns the home path or None if not found.
    """
    for proc in psutil.process_iter(attrs=["name", "cmdline"]):
        try:
            cmdline = proc.info.get("cmdline") or []
            if not cmdline:
                continue
            joined = " ".join(cmdline)
            m = re.search(r"-Dneo4j\.home=(?:\"([^\"]+)\"|(\S+))", joined)
            if m:
                home = m.group(1) or m.group(2)
                exe_unix = os.path.join(home, "bin", "neo4j")
                exe_win = os.path.join(home, "bin", "neo4j.bat")
                if os.path.exists(exe_unix) or os.path.exists(exe_win):
                    return home
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None

# --- NEW: wait for a DBMS (by home) to fully stop using 'neo4j status' ---
def wait_until_stopped_by_home(home: str, timeout_s: int = 60) -> None:
    exe = _neo4j_executable(home)
    print(f"Waiting for DBMS at {home} to stop (timeout {timeout_s}s)...")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        cp = subprocess.run(exe + ["status"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
        out = (cp.stdout or "").lower()
        if any(w in out for w in ["not running", "inactive", "stopped"]):
            print("Server stopped.")
            return
        time.sleep(1)
    raise TimeoutError("Server did not shut down in time.")

# --- NEW: stop *whatever* DBMS is currently running (no DbSpec needed) ---
def stop_current_dbms(stop_timeout: int = 60) -> bool:
    """
    Detect the running Neo4j Desktop DBMS and stop it.
    Returns True if a running DBMS was found and a stop was issued, else False.
    """
    home = detect_running_dbms_home()
    if not home:
        print("No running Neo4j DBMS detected.")
        return False

    print(f"Detected running DBMS at: {home}")
    exe = _neo4j_executable(home)
    try:
        out = _run(exe + ["stop"]).stdout
        print(out.strip())
    except subprocess.CalledProcessError as e:
        msg = (e.stdout or "").lower()
        # If already stopped between detect and stop, treat as success
        if "not running" in msg:
            print("DBMS was not running by the time we issued stop.")
            return False
        raise

    # Verify shutdown
    try:
        wait_until_stopped_by_home(home, stop_timeout)
    except TimeoutError as e:
        print(f"Warning: {e}", file=sys.stderr)
    return True


def main():
    ap = argparse.ArgumentParser(description="Start/stop Neo4j Desktop DBMSs in sequence and run tests.")
    ap.add_argument(
        "--db",
        action="append",
        required=True,
        help="DB spec: name=...,home=...,bolt=...,user=...,pass=...,start_timeout=...,stop_timeout=..."
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and print the plan without executing."
    )
    args = ap.parse_args()

    dbs = [parse_db_arg(x) for x in args.db]

    print("Plan:")
    for d in dbs:
        print(f" - {d.name}: home={d.home} bolt={d.bolt_uri}")

    if args.dry_run:
        return

    cycle_databases(dbs, example_tests)

if __name__ == "__main__":
    main()
