import runpy
import traceback

ns = runpy.run_path(
    r".\scripts\qualification\live_performance_qualification.py",
    run_name="diagnostic_only",
)

print("=== CALLING MAIN ===", flush=True)

try:
    ns["main"]()
except BaseException as exc:
    print("=== CAUGHT EXCEPTION ===", flush=True)
    print("TYPE:", type(exc).__name__, flush=True)
    print("ERROR:", repr(exc), flush=True)
    print("=== TRACEBACK ===", flush=True)
    traceback.print_exc()
    raise
