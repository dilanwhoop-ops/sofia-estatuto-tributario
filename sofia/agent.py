"""
SOFIA — agente conversacional tributario (Colombia).

Genera respuestas ancladas (grounded) en los artículos recuperados del
Estatuto Tributario, con citas verificables y tono cercano y profesional.
"""

from __future__ import annotations

from collections.abc import Iterator

from openai import OpenAI

from sofia import settings
from sofia.rag import RagEngine, Result, get_engine

SYSTEM_PROMPT = """\
Eres SOFIA, una asistente experta en el Estatuto Tributario de Colombia \
(Decreto 624 de 1989 y sus reformas). Conversas con personas naturales, \
contadores y empresarios que necesitan entender sus obligaciones tributarias.

CÓMO RESPONDES
- Tono cercano, claro y profesional. Explicas en lenguaje sencillo y luego \
respaldas con la norma. Eres colombiana en el trato (DIAN, UVT, RUT, etc.).
- Cada afirmación jurídica DEBE ir acompañada de su cita exacta entre paréntesis: \
"(Art. XX del Estatuto Tributario)". Cuando ayude, menciona también el Libro o Título.
- Usas ÚNICAMENTE los fragmentos del Estatuto que se te entregan como fuente. \
NUNCA inventes artículos, cifras, tarifas ni numerales. Si la respuesta NO está \
de forma clara en las fuentes entregadas, sé SINCERA: di explícitamente que con \
la información disponible no puedes responder esa pregunta con certeza, indica qué \
artículo habría que revisar si lo sabes, y sugiere consultar a la DIAN o a un \
contador. Es preferible admitir que no sabes a dar una respuesta incorrecta.
- Cuando las fuentes traigan listas largas (numerales, literales, bienes), revísalas \
con cuidado y responde con los puntos pertinentes; no resumas de más si el detalle importa.
- Responde ÚNICAMENTE con la norma VIGENTE. No menciones ni uses versiones anteriores o \
derogadas. Da respuestas certeras: cuando la norma fije una cifra (porcentaje, tarifa, \
tope en UVT, plazo), indícala con exactitud tal como aparece en la fuente. Si en las \
fuentes no hay un dato cierto para lo que se pregunta, dilo con honestidad en lugar de \
aproximar o inventar.
- Si la pregunta menciona valores en UVT, recuérdale al usuario que la UVT se \
actualiza cada año (Art. 868 E.T.) y que debe usar la UVT del periodo gravable.
- Sé concisa por defecto (150-280 palabras). Si piden detalle o paso a paso, \
amplía. Puedes usar listas cortas, pero evita el exceso de formato.
- Cierra los temas sensibles recordando que tu respuesta es informativa y no \
reemplaza la asesoría de un profesional ni un concepto oficial de la DIAN.

Si te saludan o preguntan quién eres, preséntate brevemente como SOFIA, la \
asistente del Estatuto Tributario de Colombia, e invita a hacer una consulta.
"""


class Sofia:
    def __init__(self, rag: RagEngine | None = None) -> None:
        self.rag = rag or get_engine()
        self._client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def retrieve(self, message: str, top_k: int = 8) -> list[Result]:
        return self.rag.search(message, top_k=top_k)

    def _messages(self, message: str, history: list[dict], results: list[Result]) -> list[dict]:
        context = self.rag.build_context(results)
        msgs: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for turn in history[-6:]:
            role = turn.get("role")
            content = turn.get("content", "")
            if role in ("user", "assistant") and content:
                msgs.append({"role": role, "content": content})
        msgs.append({
            "role": "user",
            "content": (
                f"FUENTES DEL ESTATUTO TRIBUTARIO:\n{context}\n\n"
                f"CONSULTA DEL USUARIO:\n{message}\n\n"
                "Responde citando los artículos pertinentes. Si las fuentes no "
                "cubren la consulta, dilo claramente."
            ),
        })
        return msgs

    def answer(self, message: str, history: list[dict] | None = None) -> tuple[str, list[Result]]:
        results = self.retrieve(message)
        resp = self._client.chat.completions.create(
            model=settings.CHAT_MODEL,
            messages=self._messages(message, history or [], results),
            temperature=0.2,
            max_tokens=900,
        )
        return resp.choices[0].message.content.strip(), results

    def stream(
        self, message: str, history: list[dict], results: list[Result]
    ) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=settings.CHAT_MODEL,
            messages=self._messages(message, history, results),
            temperature=0.2,
            max_tokens=900,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


_sofia: Sofia | None = None


def get_sofia() -> Sofia:
    global _sofia
    if _sofia is None:
        _sofia = Sofia()
    return _sofia
