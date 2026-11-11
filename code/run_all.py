"""
ejecutar_articulo.py
────────────────────
Todos los experimentos del artículo, en dos líneas que corren a la vez:

  Línea CPU (procesador)                 Línea GPU (tarjeta gráfica)
  C1  ACTER: preparar + CRF reserva/LODO   G1  XLM-R, reserva del 20 %, EN y ES
  C2  Clasificadores por token (10 y 20 %, G2  BiLSTM-CRF, reserva del 20 % (semillas 3-9)
      10 semillas, y validación cruzada)   G3  XLM-R, reserva del 10/30/50 %
                                           G4  BiLSTM-CRF, reserva del 10/30/50 %
                                           G5  XLM-R, validación cruzada
                                           G6  ACTER: BiLSTM-CRF y XLM-R (reserva y LODO)

El CRF en EN/ES no se vuelve a entrenar: sus 90 ejecuciones (las de la tesis, con
predicciones y tipos reservados) vienen ya en resultados/crf/.

Cuando terminan las dos, exporta TODO a resultados_csv/ (paso14_exportar.py).
Reanudable: si se corta, se vuelve a lanzar y se salta lo ya hecho.

Uso:
  EJECUTAR.bat (doble clic, Windows)       instala, prueba y lo ejecuta todo
  python ejecutar_articulo.py              todo (las dos líneas en paralelo) y exportación
  python ejecutar_articulo.py --prueba     comprobación rápida (~10-15 min)
  python ejecutar_articulo.py exportar     solo exportar lo que haya a resultados_csv/
  python ejecutar_articulo.py cpu | gpu    solo una de las dos líneas
"""
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
S = RAIZ
PY = sys.executable
TODAS = "0,1,2,3,4,5,6,7,8,9"
NUEVAS = "3,4,5,6,7,8,9"

CPU = [
    ("C1", ["paso13_acter.py", "preparar"]),
    ("C1", ["paso2_crf.py", "reserva", "--lang", "acter"]),
    ("C1", ["paso13_acter.py", "lodo", "--modelos", "crf"]),
    ("C2", ["paso6_clasificadores.py", "--niveles", "20,10", "--semillas", TODAS]),
]
GPU = [
    ("G1", ["paso12_transformer.py", "reserva", "--niveles", "20", "--semillas", TODAS]),
    ("G2", ["paso3_bilstm.py", "reserva", "--niveles", "20", "--semillas", NUEVAS]),
    ("G3", ["paso12_transformer.py", "reserva", "--niveles", "10,30,50", "--semillas", TODAS]),
    ("G4", ["paso3_bilstm.py", "reserva", "--niveles", "10,30,50", "--semillas", NUEVAS]),
    ("G5", ["paso12_transformer.py", "cv"]),
    ("G6", ["paso3_bilstm.py", "reserva", "--lang", "acter", "--semillas", TODAS]),
    ("G6", ["paso12_transformer.py", "reserva", "--lang", "acter", "--semillas", TODAS]),
    ("G6", ["paso13_acter.py", "lodo", "--modelos", "bilstm,transformer"]),
]
PRUEBA = [("P", ["paso12_transformer.py", "reserva", "--lang", "es", "--rapido"]),
          ("P", ["paso3_bilstm.py", "reserva", "--lang", "es", "--rapido", "--epocas", "1"]),
          ("P", ["paso13_acter.py", "lodo", "--modelos", "crf"]),
          ("P", ["paso14_exportar.py"])]


def entorno(extra=None):
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")  # en Windows, leer y escribir siempre en UTF-8
    env.setdefault("MODELO", "xlm-roberta-base")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.update(extra or {})
    return env


def lanzar(paso, cmd, env, log=None):
    linea = f"\n=== [{paso}] {' '.join(cmd)}  ({time.strftime('%d/%m %H:%M')})"
    print(linea, flush=True)
    t0 = time.time()
    if log is None:
        # Pantalla y articulo.log a la vez
        with open(RAIZ / "articulo.log", "a", encoding="utf-8") as f:
            f.write(linea + "\n")
            proc = subprocess.Popen([PY, str(S / cmd[0])] + cmd[1:], cwd=S, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
            for l in proc.stdout:
                print(l, end="", flush=True)
                f.write(l)
                f.flush()
            proc.wait()
        r = proc
    else:
        with open(log, "a", encoding="utf-8") as f:
            f.write(linea + "\n")
            f.flush()
            r = subprocess.run([PY, str(S / cmd[0])] + cmd[1:], cwd=S, env=env, stdout=f, stderr=subprocess.STDOUT)
    fin = f"=== [{paso}] fin: {(time.time()-t0)/60:.0f} min, {'correcto' if r.returncode == 0 else 'ERROR (código %d)' % r.returncode}"
    print(fin, flush=True)
    with open(log or RAIZ / "articulo.log", "a", encoding="utf-8") as f:
        f.write(fin + "\n")
    return r.returncode


def ejecutar_linea(pasos, env, log=None):
    fallos = 0
    for paso, cmd in pasos:
        if lanzar(paso, cmd, env, log):
            fallos += 1
            print(f"    El paso {paso} ha fallado; sigo con el siguiente. Al relanzar se reintenta.", flush=True)
    return fallos


def main():
    args = sys.argv[1:]
    if "--prueba" in args:
        env = entorno({"RESULTADOS": str(RAIZ / "resultados_prueba"), "SALIDA_CSV": str(RAIZ / "resultados_prueba" / "csv"),
                       "EPOCAS_RAPIDO": "1"})
        if ejecutar_linea(PRUEBA, env):
            sys.exit("\nLa prueba ha fallado: revisa el mensaje de error de arriba y send it to the authors.")
        print("\nPrueba correcta (las cifras de la prueba no significan nada). "
              "Borra la carpeta resultados_prueba y lanza: python ejecutar_articulo.py")
        return
    env = entorno()
    if "exportar" in args:
        lanzar("EXP", ["paso14_exportar.py"], env)
        return
    solo = "cpu" if "cpu" in args else "gpu" if "gpu" in args else None
    hilo = None
    if solo == "cpu":
        ejecutar_linea(CPU, env)
    elif solo is None:
        print("La línea CPU corre en segundo plano; su salida va a log_cpu.txt", flush=True)
        hilo = threading.Thread(target=ejecutar_linea, args=(CPU, env, RAIZ / "log_cpu.txt"))
        hilo.start()
    if solo in (None, "gpu"):
        ejecutar_linea(GPU, env)
    if hilo is not None:
        print("\nLínea GPU terminada; esperando a que acabe la línea CPU (ver log_cpu.txt)...", flush=True)
        hilo.join()
    lanzar("EXP", ["paso14_exportar.py"], env)
    print("\nTerminado. Return to the authors: resultados_csv/, resultados/, articulo.log y log_cpu.txt.")


if __name__ == "__main__":
    main()
