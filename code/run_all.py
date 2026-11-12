"""
run_all.py
──────────
Runs every experiment in the paper and then exports all results to CSV.

The steps run one after another; each script skips the runs whose JSON file
already exists, so the whole thing can be stopped and relaunched at any time.
The CRF and the token classifiers only need a CPU; the BiLSTM-CRF and
XLM-RoBERTa are much faster on a GPU.

Usage:
  python run_all.py              all experiments, then the CSV export
  python run_all.py --quick      quick check of the pipeline (~15 min, numbers are meaningless)
  python run_all.py export       only export the existing results to CSV

Environment variables: DATOS_GENERADOS (token files, default ../datos_generados),
RESULTADOS (raw results, default ../resultados), SALIDA_CSV (CSV export).
"""
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable
ALL_SEEDS = "0,1,2,3,4,5,6,7,8,9"

STEPS = [
    # feature-based CRF: reservation, separate entity labels, ablation, random CV
    ["train_crf.py", "todo", "--ablacion-completa", "--todos-niveles"],
    ["train_crf.py", "ablacv"],
    # token classifiers with the same features
    ["train_token_classifiers.py", "--niveles", "10,20", "--semillas", ALL_SEEDS],
    # BiLSTM-CRF (and its character-level variant)
    ["train_bilstm_crf.py", "reserva", "--niveles", "10,20,30,50", "--semillas", ALL_SEEDS],
    ["train_bilstm_crf.py", "cv"],
    ["train_bilstm_crf_char.py", "reserva", "--niveles", "20", "--semillas", "0,1,2,3,4"],
    # XLM-RoBERTa
    ["train_xlmr.py", "reserva", "--niveles", "10,20,30,50", "--semillas", ALL_SEEDS],
    ["train_xlmr.py", "cv"],
    # ACTER: type reservation and leave-one-domain-out
    ["acter_experiments.py", "preparar"],
    ["train_crf.py", "reserva", "--lang", "acter"],
    ["train_bilstm_crf.py", "reserva", "--lang", "acter", "--niveles", "20", "--semillas", ALL_SEEDS],
    ["train_xlmr.py", "reserva", "--lang", "acter", "--niveles", "20", "--semillas", ALL_SEEDS],
    ["acter_experiments.py", "lodo", "--modelos", "crf,bilstm,transformer"],
]
QUICK = [
    ["train_xlmr.py", "reserva", "--lang", "es", "--rapido"],
    ["train_bilstm_crf.py", "reserva", "--lang", "es", "--rapido", "--epocas", "1"],
    ["train_crf.py", "reserva", "--lang", "es", "--rapido"],
]


def run(cmd, env):
    print(f"\n=== {' '.join(cmd)}  ({time.strftime('%d/%m %H:%M')})", flush=True)
    t0 = time.time()
    code = subprocess.run([PY, str(HERE / cmd[0])] + cmd[1:], cwd=HERE, env=env).returncode
    status = "ok" if code == 0 else f"FAILED (exit code {code})"
    print(f"=== {cmd[0]}: {(time.time() - t0) / 60:.0f} min, {status}", flush=True)
    return code


def main():
    args = sys.argv[1:]
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    env.setdefault("MODELO", "xlm-roberta-base")
    if "export" in args:
        run(["export_results.py"], env)
        return
    if "--quick" in args:
        quick = HERE.parent / "results_quick_check"
        env.update(RESULTADOS=str(quick), SALIDA_CSV=str(quick / "csv"))
        failed = sum(run(cmd, env) != 0 for cmd in QUICK)
        print("\nQuick check failed: see the error above." if failed else
              f"\nQuick check passed. Delete {quick.name}/ and run: python run_all.py")
        return
    failed = [cmd[0] for cmd in STEPS if run(cmd, env) != 0]
    run(["export_results.py"], env)
    if failed:
        print("\nSome steps failed and can be relaunched (finished runs are skipped):", ", ".join(failed))


if __name__ == "__main__":
    main()
