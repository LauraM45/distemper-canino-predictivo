import io
import sys
import base64
import json
import traceback
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

nb_path = Path("evaluation/model_comparation.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

global_env = {
    "__name__": "__main__",
    "__file__": str(nb_path.resolve()),
}

execution_counter = 1

for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    
    code = "".join(cell.get("source", []))
    if not code.strip():
        continue
        
    old_stdout = sys.stdout
    redirected_stdout = sys.stdout = io.StringIO()
    plt.close('all')
    
    cell_outputs = []
    
    try:
        exec(code, global_env)
        
        # 1. Capturar stdout
        stdout_val = redirected_stdout.getvalue()
        if stdout_val:
            cell_outputs.append({
                "output_type": "stream",
                "name": "stdout",
                "text": stdout_val.splitlines(keepends=True)
            })
            
        # 2. Capturar figuras de matplotlib
        fig_nums = plt.get_fignums()
        for fnum in fig_nums:
            fig = plt.figure(fnum)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
            buf.seek(0)
            b64_str = base64.b64encode(buf.read()).decode("ascii")
            cell_outputs.append({
                "output_type": "display_data",
                "data": {
                    "image/png": b64_str,
                    "text/plain": ["<Figure size ... with ... Axes>"]
                },
                "metadata": {}
            })
            plt.close(fig)
            
        cell["outputs"] = cell_outputs
        cell["execution_count"] = execution_counter
        print(f"[OK] Celda {idx:2d} ejecutada con éxito (outputs: {len(cell_outputs)})")
        execution_counter += 1
        
    except Exception as e:
        sys.stdout = old_stdout
        print(f"[ERROR] en Celda {idx}: {e}")
        traceback.print_exc()
        raise e
    finally:
        sys.stdout = old_stdout

# Guardar notebook actualizado
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\n[ÉXITO] Cuaderno renovado y guardado completamente con los nuevos datos en: {nb_path}")
