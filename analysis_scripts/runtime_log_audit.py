"""Audit of the runtime records behind docs/GPU4_INFERENCE_LEDGER.md. Two parts, both read-only; run on the cluster under msr_venv.

  python analysis_scripts/runtime_log_audit.py ledger   job-name patterns vs every Slurm accounting record since 2026-10-01 (unmatched jobs,
                                                        Elapsed vs End-Start)
  python analysis_scripts/runtime_log_audit.py timing   every stability-baselines results/*/predictions*/timing.csv: duplicate datasets (reruns),
                                                        rounding artefacts, near-zero wall times (cache hits), loop time vs job Elapsed,
                                                        host vs the node Slurm reports"""
import runpy, sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = {'ledger': os.path.join(HERE, 'runtime_log_audit_ledger.py'), 'timing': os.path.join(HERE, 'runtime_log_audit_timing.py')}
if __name__ == '__main__':
    runpy.run_path(SCRIPTS[sys.argv[1]], run_name='__main__')
