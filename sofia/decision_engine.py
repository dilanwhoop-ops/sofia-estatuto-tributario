"""
Motor de árboles de decisión de SOFIA.

Sólo usa árboles APROBADOS (estado en data/trees_status.json). Las respuestas
son DETERMINISTAS (se leen del árbol, no las genera el LLM) para evitar
alucinaciones: la ley fija condiciones/umbrales y el árbol los aplica.

Flujo conversacional:
  - find_tree(query): ¿la pregunta coincide con un árbol aprobado?
  - start(tree): entra por la raíz → muestra pregunta + opciones numeradas.
  - continue_(state, message): interpreta la respuesta y avanza hasta un resultado.
"""

from __future__ import annotations

import json
import re

from sofia import settings

# Disparadores por defecto (frases que activan cada árbol). El árbol puede
# sobreescribirlos con su propio campo "triggers".
DEFAULT_TRIGGERS: dict[str, list[str]] = {
    "obligado_declarar_renta": [
        "obligado a declarar", "debo declarar", "tengo que declarar",
        "declarar renta", "declaración de renta", "declaracion de renta",
    ],
    "responsable_iva": [
        "responsable de iva", "responsable del iva", "soy responsable de iva",
        "no responsable de iva", "régimen de iva", "regimen de iva",
    ],
    "sancion_no_declarar": [
        "sanción por no declarar", "sancion por no declarar", "multa por no declarar",
        "no declaré", "no declare", "no presenté la declaración",
    ],
    "regimen_simple": [
        "régimen simple", "regimen simple", "simple de tributación",
        "simple de tributacion", "puedo optar al simple", "rst",
    ],
}

_EXIT_RE = re.compile(r"\b(salir|cancelar|terminar|otra pregunta|olvídalo|olvidalo)\b", re.I)


class DecisionEngine:
    def __init__(self) -> None:
        self._trees: dict[str, dict] = {}
        if settings.DECISION_TREES_DIR.exists():
            for f in sorted(settings.DECISION_TREES_DIR.glob("*.json")):
                try:
                    t = json.loads(f.read_text(encoding="utf-8"))
                    self._trees[t["id"]] = t
                except Exception:
                    pass

    # ── aprobación ──
    def _approved_ids(self) -> set[str]:
        if not settings.TREES_STATUS.exists():
            return set()
        try:
            status = json.loads(settings.TREES_STATUS.read_text(encoding="utf-8"))
        except Exception:
            return set()
        return {tid for tid, st in status.items() if st.get("estado") == "aprobado"}

    def approved_trees(self) -> list[dict]:
        ids = self._approved_ids()
        return [t for tid, t in self._trees.items() if tid in ids]

    # ── matching ──
    def find_tree(self, query: str) -> dict | None:
        q = " " + query.lower() + " "
        best, best_score = None, 0
        for t in self.approved_trees():
            triggers = t.get("triggers") or DEFAULT_TRIGGERS.get(t["id"], [])
            score = sum(1 for kw in triggers if kw.lower() in q)
            for a in t.get("articulos", []):
                if re.search(r"\bart[íi]culo\s+" + re.escape(a) + r"\b", q):
                    score += 1
            if score > best_score:
                best, best_score = t, score
        return best if best_score >= 1 else None

    # ── recorrido ──
    def start(self, tree: dict) -> dict:
        return self._enter(tree, tree["root"])

    def continue_(self, state: dict, message: str) -> dict:
        tree = self._trees.get(state.get("tree_id", ""))
        if not tree or tree["id"] not in self._approved_ids():
            return {"text": "Ese árbol ya no está disponible. Te respondo de otra forma.",
                    "state": None, "sources": [], "done": True, "fallback": True}
        if _EXIT_RE.search(message):
            return {"text": "Listo, salimos del modo paso a paso. ¿En qué más te ayudo? 😊",
                    "state": None, "sources": [], "done": True}
        node = tree["nodes"].get(state.get("node", ""))
        if not node:
            return {"text": "Reiniciemos. " + self._render(tree, tree["nodes"][tree["root"]], tree["root"])["text"],
                    "state": {"tree_id": tree["id"], "node": tree["root"]}, "sources": [], "done": False}
        goto = self._match(node, message)
        if not goto:
            r = self._render(tree, node, state["node"])
            return {"text": "No te entendí. " + r["text"], "state": state, "sources": [], "done": False}
        return self._enter(tree, goto)

    def _enter(self, tree: dict, node_id: str) -> dict:
        node = tree["nodes"][node_id]
        r = self._render(tree, node, node_id)
        if node.get("type") == "decision":
            r["state"] = {"tree_id": tree["id"], "node": node_id}
            r["done"] = False
        else:  # result / info
            r["state"] = None
            r["done"] = True
        return r

    # ── render (texto determinista) ──
    def _render(self, tree: dict, node: dict, node_id: str) -> dict:
        if node.get("type") == "decision":
            lines = [f"📋 **{tree['titulo']}**", "", node.get("pregunta", "")]
            for i, o in enumerate(node.get("opciones", []), 1):
                lines.append(f"**{i}.** {o['texto']}")
            lines.append("")
            lines.append("_Responde con el número de la opción (o escribe «salir» para terminar)._")
            return {"text": "\n".join(lines), "sources": []}
        # resultado
        arts = node.get("articulos", [])
        cita = " ".join(f"(Art. {a} E.T.)" for a in arts)
        partes = [f"✅ **{node.get('resultado', node.get('texto', ''))}**"]
        if node.get("detalle"):
            partes.append(f"\n{node['detalle']} {cita}".rstrip())
        elif cita:
            partes.append(f"\n{cita}")
        partes.append("\n_Respuesta basada en un árbol de decisión validado y aprobado. "
                      "Es orientativa y no reemplaza la asesoría de un contador o la DIAN._")
        return {"text": "\n".join(partes), "sources": arts}

    # ── interpretación de la respuesta del usuario ──
    @staticmethod
    def _match(node: dict, message: str) -> str | None:
        opts = node.get("opciones", [])
        if not opts:
            return None
        m = message.strip().lower()
        dig = re.match(r"^\s*(\d+)", m)
        if dig:
            i = int(dig.group(1))
            if 1 <= i <= len(opts):
                return opts[i - 1]["goto"]
        for o in opts:
            t = o["texto"].lower()
            if t and (t in m or (len(m) > 2 and m in t)):
                return o["goto"]
        if m in ("si", "sí", "claro", "correcto", "afirmativo", "verdadero"):
            for o in opts:
                if re.match(r"s[íi]\b", o["texto"].strip().lower()):
                    return o["goto"]
        if m in ("no", "negativo", "falso"):
            for o in opts:
                if o["texto"].strip().lower().startswith("no"):
                    return o["goto"]
        return None


_engine: DecisionEngine | None = None


def get_decision_engine() -> DecisionEngine:
    global _engine
    if _engine is None:
        _engine = DecisionEngine()
    return _engine
