"""
Generación SEMI-AUTOMÁTICA de árboles de decisión.

La IA lee el texto de un artículo del Estatuto y propone un BORRADOR de árbol
(condiciones y umbrales que la propia norma fija). El borrador queda en estado
PENDIENTE: no lo usa SOFIA hasta que un humano lo apruebe en /admin.

Uso:
    python -m sofia.generate_trees --candidatos          # sugiere artículos
    python -m sofia.generate_trees --articulo 437        # genera un borrador
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone

from openai import OpenAI

from sofia import settings

# Señales de que un artículo contiene condiciones/umbrales (buen candidato)
COND_PATTERNS = [
    r"\bUVT\b", r"iguales? o superiores? a", r"no exceda", r"superiores? a",
    r"inferiores? a", r"est[áa]n obligad", r"no est[áa]n obligad", r"siempre que",
    r"cuando\b", r"si\b.{0,40}\bentonces", r"no ser[áa]n", r"ser[áa]n responsables",
]
COND_RE = re.compile("|".join(COND_PATTERNS), re.I)

SCHEMA_HINT = """{
  "titulo": "Pregunta clara que resuelve el árbol",
  "tema": "Renta | IVA | Retención | Sanciones | Procedimiento | ...",
  "descripcion": "Una frase de qué determina",
  "triggers": ["frase que activaría el árbol", "otra frase", "..."],
  "advertencia": "Aviso si aplica (p. ej. los topes van en UVT del año)",
  "root": "n1",
  "nodes": {
    "n1": {"type":"decision","pregunta":"...","opciones":[{"texto":"Sí","goto":"n2"},{"texto":"No","goto":"r_no"}]},
    "n2": {"type":"decision","pregunta":"...","opciones":[{"texto":"Sí","goto":"r_si"},{"texto":"No","goto":"r_no"}]},
    "r_si": {"type":"result","resultado":"...","detalle":"...","articulos":["NUM"]},
    "r_no": {"type":"result","resultado":"...","detalle":"...","articulos":["NUM"]}
  }
}"""

PROMPT = """\
Eres un analista tributario. Convierte el siguiente ARTÍCULO del Estatuto Tributario \
de Colombia en un ÁRBOL DE DECISIÓN en JSON.

REGLAS ESTRICTAS
- Usa SOLO lo que dice el artículo. No inventes cifras, topes ni condiciones.
- Copia los umbrales EXACTOS como aparecen (p. ej. "3.500 UVT", "35%").
- Las preguntas deben ser cerradas (Sí/No o pocas opciones claras) y en segunda persona.
- Cada "goto" debe apuntar a una clave existente en "nodes".
- Debe haber al menos un nodo "result". Los result citan el artículo en "articulos".
- Si el artículo NO contiene condiciones que formen un árbol útil, responde exactamente: {"error":"sin_condiciones"}
- Responde SÓLO JSON válido con esta forma:
""" + SCHEMA_HINT + """

ARTÍCULO {numero} — {epigrafe}
\"\"\"
{texto}
\"\"\"
"""


def _articles() -> dict[str, dict]:
    arts = json.loads(settings.ESTATUTO_JSON.read_text(encoding="utf-8"))
    return {a["numero"]: a for a in arts if not a.get("derogada")}


def find_candidates(top_n: int = 15) -> list[dict]:
    """Artículos con más señales de condiciones/umbrales."""
    out = []
    for a in _articles().values():
        hits = len(COND_RE.findall(a["texto"]))
        uvt = len(re.findall(r"\bUVT\b", a["texto"], re.I))
        if hits >= 3 and 200 < len(a["texto"]) < 9000:
            out.append({"numero": a["numero"], "epigrafe": a["epigrafe"][:60],
                        "señales": hits, "uvt": uvt})
    out.sort(key=lambda x: (x["uvt"], x["señales"]), reverse=True)
    return out[:top_n]


def _validate(tree: dict) -> tuple[bool, str]:
    if "nodes" not in tree or "root" not in tree:
        return False, "faltan 'nodes' o 'root'"
    nodes = tree["nodes"]
    if tree["root"] not in nodes:
        return False, "root inexistente"
    if not any(n.get("type") == "result" for n in nodes.values()):
        return False, "sin nodo result"
    for k, n in nodes.items():
        for o in n.get("opciones", []):
            if o.get("goto") not in nodes:
                return False, f"goto inválido en {k}: {o.get('goto')}"
    return True, "ok"


def generate(numero: str) -> dict:
    """Genera un BORRADOR de árbol para un artículo. Lanza ValueError si no aplica."""
    art = _articles().get(numero)
    if not art:
        raise ValueError(f"El artículo {numero} no existe o está derogado.")
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    # replace() y no format(): el prompt lleva llaves {} del propio JSON de ejemplo
    prompt = (PROMPT.replace("{numero}", numero)
                    .replace("{epigrafe}", art["epigrafe"])
                    .replace("{texto}", art["texto"][:7000]))
    resp = client.chat.completions.create(
        model=settings.CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
        max_tokens=2000,
    )
    data = json.loads(resp.choices[0].message.content)
    if data.get("error") == "sin_condiciones":
        raise ValueError(f"El Art. {numero} no tiene condiciones que formen un árbol útil.")

    data["id"] = f"auto_{numero}"
    data.setdefault("articulos", [numero])
    data["generado_por"] = "IA (borrador)"
    data["generado_el"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    ok, msg = _validate(data)
    if not ok:
        raise ValueError(f"El borrador generado no es válido: {msg}")

    settings.DECISION_TREES_DIR.mkdir(parents=True, exist_ok=True)
    path = settings.DECISION_TREES_DIR / f"auto_{numero}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidatos", action="store_true", help="lista artículos candidatos")
    ap.add_argument("--articulo", help="número de artículo a convertir en árbol")
    args = ap.parse_args()

    if args.candidatos:
        print("Artículos candidatos (con condiciones/umbrales):")
        for c in find_candidates():
            print(f"  Art. {c['numero']:>7}  UVT:{c['uvt']:>2}  señales:{c['señales']:>3}  {c['epigrafe']}")
        return
    if args.articulo:
        t = generate(args.articulo)
        print(f"Borrador creado: {t['id']} — {t['titulo']}")
        print(f"  nodos: {len(t['nodes'])} · queda PENDIENTE de aprobación en /admin")
        return
    ap.print_help()


if __name__ == "__main__":
    main()
