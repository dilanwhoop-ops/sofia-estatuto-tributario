"""
Construye el grafo de referencias entre artículos del Estatuto Tributario.

Cada artículo vigente es un nodo; cada vez que el texto de un artículo menciona
"el artículo N" (u "los artículos N, M y P") se crea una arista dirigida hacia
ese artículo. Salida: data/graph.json  (nodes + edges) para visualizar.

Uso:  python -m sofia.build_graph
"""

from __future__ import annotations

import json
import re

from sofia import settings

# "artículo 240", "artículos 360 y 361", "art. 868-1", "los artículos 580, 588 y 698"
REF_RE = re.compile(
    r"art[íi]culos?\s+((?:\d+(?:-\d+)?[º°]?(?:\s*(?:,|y|e)\s*)?){1,12})",
    re.I,
)
NUM_RE = re.compile(r"\d+(?:-\d+)?")


def build() -> dict:
    articles = json.loads(settings.ESTATUTO_JSON.read_text(encoding="utf-8"))
    by_num = {a["numero"]: a for a in articles if not a.get("derogada")}

    edges: dict[tuple[str, str], int] = {}
    for a in articles:
        if a.get("derogada"):
            continue
        src = a["numero"]
        texto = a["texto"]
        for m in REF_RE.finditer(texto):
            for num in NUM_RE.findall(m.group(1)):
                if num != src and num in by_num:
                    edges[(src, num)] = edges.get((src, num), 0) + 1

    # grado (número de conexiones) para dimensionar nodos
    deg: dict[str, int] = {}
    for (s, t) in edges:
        deg[s] = deg.get(s, 0) + 1
        deg[t] = deg.get(t, 0) + 1

    nodes = [{
        "id": n,
        "epigrafe": (by_num[n]["epigrafe"] or "")[:80],
        "libro": by_num[n]["libro"],
        "grado": deg.get(n, 0),
    } for n in by_num]

    edge_list = [{"source": s, "target": t, "n": c} for (s, t), c in edges.items()]
    return {"nodes": nodes, "edges": edge_list}


def main() -> None:
    g = build()
    (settings.DATA_DIR / "graph.json").write_text(
        json.dumps(g, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Grafo: {len(g['nodes'])} nodos, {len(g['edges'])} aristas -> data/graph.json")
    # top conectados
    top = sorted(g["nodes"], key=lambda n: n["grado"], reverse=True)[:8]
    for n in top:
        print(f"  Art. {n['id']:>6}  grado {n['grado']:>3}  {n['epigrafe'][:45]}")


if __name__ == "__main__":
    main()
