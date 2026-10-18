# Agente RAG — Nutrição & Fitness

Estudo de RAG com LangChain. Todo o código está em `main.py`.

## Setup

```bash
python3.12 -m venv .venv
./.venv/bin/pip install -r requirements.txt

cp .env.example .env
# preencha OPENAI_API_KEY no .env
```

| Chave | Onde pegar |
|---|---|
| `OPENAI_API_KEY` | platform.openai.com → API keys · créditos pré-pagos mín. US$ 5. **ChatGPT Plus não serve** |

Uma única chave cobre embeddings e LLM.

> 🔐 Chave nunca dentro de `.py`. Só no `.env`, que está no `.gitignore`.

Coloque os PDFs em `dados/pdfs/`.

## Uso

```bash
./.venv/bin/python main.py
```

Pede a pergunta no terminal (`Pergunta: `). Na primeira execução indexa os PDFs
em `chroma_db/`; nas seguintes, carrega o índice já existente.

Se trocar o conteúdo dos PDFs, apague `chroma_db/` para forçar reindexação.

## Pipeline

| Função | O que faz |
|---|---|
| `carregar_pdfs` | lê os PDFs de `dados/pdfs/` |
| `dividir_documentos` | parte em chunks (`RecursiveCharacterTextSplitter`) |
| `criar_embeddings` | `text-embedding-3-small` |
| `criar_vectorstore` / `carregar_vectorstore` | indexa ou carrega o Chroma |
| `criar_llm` | `ChatOpenAI` (`MODELO_LLM`) |
| `criar_prompt` | prompt com `{contexto}` + `{pergunta}` |
| `responder` | busca os `k` trechos mais relevantes e roda a cadeia LCEL `prompt \| llm \| StrOutputParser()` |

## Problemas comuns

| Sintoma | Causa | Solução |
|---|---|---|
| "Nenhum PDF encontrado" | pasta vazia | copie PDFs para `dados/pdfs/` |
| Resposta cortada no meio | `max_tokens` baixo | aumente em `criar_llm()` |
| `ModuleNotFoundError: chromadb` | venv errado | use `.venv/` (Python 3.12) |
| Resposta com info antiga | `chroma_db/` desatualizado | apague a pasta e rode de novo |
