# Detecção de desinformação sobre vacinas contra a COVID-19

Um pipeline que coleta vídeos do YouTube sobre vacinas contra a COVID-19,
transcreve o áudio e sinaliza as afirmações contraditas por um corpus de
evidências curado. Um modelo de linguagem servido localmente julga cada
afirmação com base nas evidências recuperadas, e o resultado é avaliado contra
vídeos rotulados manualmente que nunca foram usados para ajustar nada.

## Como funciona

```
collect ──► transcribe ──► rag ──► experiments
  │             │           │          │
  YouTube       Whisper     retrieval  test set separado,
  Data API      no áudio    + LLM      avaliado contra os labels
                            ▲
                 data/evidence (PDFs) ──► data/index (vetores)
```

| etapa | pacote | o que faz |
|---|---|---|
| 1 | `collect/` | Busca as queries configuradas na YouTube Data API, completa cada resultado com os metadados completos, remove duplicatas e descarta vídeos com mais de uma hora. Nunca faz scraping. |
| 2 | `transcribe/` | Baixa o áudio de cada vídeo e o transcreve com faster-whisper. Descarta áudio que não está em inglês após a detecção de idioma e salva um checkpoint a cada vídeo. |
| 3 | `rag/` | Divide a transcrição em janelas de afirmação sobrepostas, recupera passagens do corpus para cada uma e pergunta ao modelo se a evidência trata da afirmação e, só então, se a contradiz. Afirmações que o corpus não cobre podem recorrer a uma busca na web. |
| 4 | `experiments/` | Define o split test/dev, executa o pipeline sobre ele e calcula as métricas. Todos os parâmetros de `rag/config.py` foram escolhidos no dev set. |

O pipeline reúne candidatos e só depois os classifica. A coleta é
deliberadamente neutra, para que o dataset seja montado antes de qualquer
julgamento.

## Estrutura do repositório

```
collect/         etapa 1: config  models  filters  data  api  collector  main
transcribe/      etapa 2: config  audio  whisper  data  transcriber  main
rag/             etapa 3: config  corpus  manifest  router  index  lexical  llm
                          websearch  temporal  pipeline  whole  main
experiments/     harness de avaliação: split, runner, scoring, sweeps, probes
scripts/         run_test_model.sh (o protocolo); workstation/ (drivers da máquina com GPU)
data/            youtube/  evidence/  index/  labels/  results/
docs/            o que foi medido e o que isso decidiu
notebooks/       os notebooks exploratórios originais, substituídos pelos pacotes
LICENSE          MIT, para o código;  LICENSE-DATA.md: termos de cada arquivo de dados
CITATION.cff     metadados de citação do repositório e do TCC
```

`experiments/`, `scripts/` e `data/` têm cada um o seu próprio README.

Os três pacotes do pipeline seguem a mesma divisão em camadas, agrupadas por
dependência externa, de modo que uma falha fica contida e cada camada funciona
sem as que estão acima dela:

| camada | depende de |
|---|---|
| `config`, `models` | nada |
| `filters`, `router`, `temporal` | lógica pura, sem I/O |
| `data`, `corpus`, `manifest` | stdlib e pandas; toda escrita é atômica |
| `api`, `audio`, `whisper`, `index`, `llm`, `websearch` | a fronteira externa: rede, yt-dlp, ffmpeg, Whisper, o servidor do modelo |
| `collector`, `transcriber`, `pipeline`, `whole` | as camadas acima; controla a ordem do trabalho |
| `main` | parsing de argumentos e exit codes |

As dependências apontam em um só sentido. Os nomes públicos são reexportados
de forma lazy (PEP 562), então importar um módulo não carrega as dependências
pesadas dos módulos vizinhos. O único import entre pacotes é `is_usable`, de
`transcribe.config`: ele decide quais transcrições entram em um experimento e,
por isso, tem uma definição única.

## Instalação

Python 3.10 ou superior. O `ffmpeg` só é necessário para transcrever; o
experimento usa as transcrições já versionadas no repositório.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
brew install ffmpeg          # ou apt install ffmpeg
```

As chaves ficam no `.env` na raiz do projeto (no gitignore) ou no ambiente:

| chave | usada para |
|---|---|
| `YOUTUBE_API_KEY` | `collect`. Gratuita: crie um projeto no Google Cloud Console e ative a YouTube Data API v3 |
| `TAVILY_API_KEY` | o fallback `--web tavily`. O free tier cobre uma execução completa do test set |
| `OPENROUTER_API_KEY` | os dois modelos remotos |

## Uso

### 1. Coletar

```bash
python -m collect                                   # as queries de collect/config.py
python -m collect --links links.txt --no-search     # uma lista curada no lugar das buscas
python -m collect --name my_dev --exclude data/youtube/exclude_dev2.json
```

`--exclude` mantém vídeos já vistos fora de uma nova coleta, para que o
material de dev não se sobreponha ao test set. O idioma não é filtrado aqui: o
campo de idioma do YouTube é autodeclarado e muitas vezes está errado, então
quem decide é o `transcribe`, a partir do áudio.

### 2. Transcrever

```bash
python -m transcribe --list
python -m transcribe covid_vaccine_videos
python -m transcribe covid_vaccine_videos --limit 5
python -m transcribe covid_vaccine_videos --max-transcribe 0    # vídeos inteiros
```

Cerca de 6x o tempo real com o modelo `small` em um M2 Pro. Uma execução
interrompida retoma a partir do checkpoint. Os dev sets são limitados a 12
minutos por vídeo; o test set foi transcrito na íntegra.

### 3. Checar os fatos

O modelo é servido localmente por um endpoint compatível com a API da OpenAI.
O llama.cpp é o padrão porque impõe a gramática do veredito durante o
sampling, de modo que um veredito malformado não pode ser gerado.

```bash
# gpt-oss-20b (exige --jinja: sem ele, o preâmbulo de reasoning vaza para a saída)
llama-server -hf ggml-org/gpt-oss-20b-GGUF:MXFP4 -c 8192 --port 8080 --host 127.0.0.1 --jinja

# Gemma 4 12B
llama-server -hf google/gemma-4-12B-it-qat-q4_0-gguf -c 8192 --port 8080 --host 127.0.0.1
```

O `-c 8192` importa: o prompt leva até onze passagens, e um contexto menor as
trunca silenciosamente. Em GPU, acrescente `-ngl 99`.

| variável | padrão | significado |
|---|---|---|
| `LLM_SERVER` | `llamacpp` | `ollama` ou `openrouter` para trocar de backend; cada um define sua URL e sua restrição de saída |
| `LLAMA_URL` | conforme o backend | o servidor, se não estiver no host e na porta padrão |
| `LLAMA_MODEL` | `qwen2.5:14b` | **defina explicitamente**: o llama-server o ignora, mas é ele que dá nome aos arquivos de resultado e fica registrado nos metadados da execução |
| `LLM_CONSTRAINT` | conforme o backend | `grammar`, `json_schema` ou `json_object`, para sobrescrever o padrão do backend |
| `LLM_REASONING` | padrão do modelo | `off`, `low`, `medium` ou `high`; `off` acrescenta `_nothink` ao nome de cada saída |
| `LLM_TIMEOUT_S` | 300 (600 no remoto) | timeout por requisição |
| `RESULTS_VERSION` | `v1` | em qual pasta de `data/results/` a execução grava |
| `WEB_SEARCH_BACKEND` | `duckduckgo` | backend do fallback quando `--web` não é informado |

Em seguida:

```bash
python -m rag --build-manifest             # resolve os DOIs em data/evidence/sources.json
python -m rag --build-index --no-extract   # gera os embeddings das passagens versionadas -> data/index/
python -m rag --check-server
python -m rag --list                       # vídeos com transcrições utilizáveis
python -m rag Y6_jn0DzXHI                  # checa um vídeo
python -m rag Y6_jn0DzXHI --web tavily --json out.json
```

O `--build-index` deve reportar 2488 passagens. Um número diferente significa
que o corpus mudou e que os valores de parâmetros documentados foram medidos
sobre outro corpus. Qual documento cobre qual tópico é declarado à mão em
`rag/manifest.py`; trata-se de uma decisão editorial, não de algo derivado.

Um veredito é `FAKE_NEWS`, `NOT_FAKE_NEWS`, `NO_EVIDENCE` ou `ERROR`. O modelo
precisa dizer se a evidência trata da afirmação antes de poder dar um
veredito; essa ordem corrigiu a falha dominante, em que a ausência de evidência
era lida como contradição. Uma sinalização também exige que o falante esteja
afirmando aquilo, e não citando ou negando. Vereditos baseados na web recebem
o sufixo `_WEB` e são contados à parte, porque essa evidência não passou pela
triagem de licença e autoria do corpus.

### 4. Avaliar

```bash
python experiments/datasets.py                 # tamanho dos splits e checagem de leakage
python experiments/run_experiment.py --check   # preflight: servidor, índice, labels, web, estimativa de tempo
```

O protocolo de um modelo é o `scripts/run_test_model.sh`: preflight, depois,
para cada braço, um smoke test com 3 vídeos e a execução completa, e por fim as
métricas. Os quatro modelos reportados:

```bash
unset LLM_SERVER; export LLAMA_MODEL=gpt-oss-20b LLM_CONSTRAINT=json_schema; scripts/run_test_model.sh
unset LLM_SERVER LLM_CONSTRAINT; export LLAMA_MODEL=gemma-4-12b;              scripts/run_test_model.sh
export LLM_SERVER=openrouter LLAMA_MODEL=qwen/qwen3.6-27b LLM_REASONING=off;  scripts/run_test_model.sh
export LLM_SERVER=openrouter LLAMA_MODEL=openai/gpt-oss-120b;                 scripts/run_test_model.sh
```

Os modelos locais rodam um de cada vez na porta 8080. O script retoma uma
execução interrompida, e uma execução já completa é apenas reavaliada. Com
`WHOLE=1`, roda o protocolo de transcrição inteira (v3) em vez do protocolo
por janelas (v1).

Cada braço é avaliado isoladamente, sem e com o overlap veto, e os dois braços
são avaliados juntos como evidence veto: uma sinalização paramétrica se mantém,
a menos que o braço com evidência tenha liberado a mesma janela. Para
reavaliar manualmente:

```bash
python experiments/run_experiment.py --score --out data/results/v1/test_run_<model>_tavily.json
python experiments/run_experiment.py --score --overlap-veto --out data/results/v1/test_run_<model>_tavily.json
python experiments/score_verifier.py data/results/v1/test_run_<model>_parametric.json \
                                     data/results/v1/test_run_<model>_tavily.json --overlap-veto
```

Um vídeo é predito como fake quando qualquer uma de suas janelas é sinalizada.
Os resultados ficam agrupados por protocolo em `data/results/` (veja o README
de lá); cada execução registra a configuração completa, o commit do git e o
tempo em um bloco `run`, de modo que qualquer número pode ser atribuído às
configurações que o produziram.

O test set sempre roda os prompts de produção de `rag/llm.py`. Variantes de
prompt são comparadas apenas no dev set (`--dataset dev --prompt NAME`).

### Reproduzindo as tabelas

Todos os números reportados são calculados a partir de arquivos de resultado
que estão no repositório, então nenhum modelo precisa ser executado. Os
arquivos v1 reportados são, por modelo,
`data/results/v1/test_run_<model>_parametric.json` e `..._tavily.json`, com
`<model>` sendo `gemma-4-12b`, `gpt-oss-20b`, `openai-gpt-oss-120b` ou
`qwen-qwen3.6-27b` (os arquivos do Qwen levam o sufixo `_nothink`). Os
arquivos v3 são os `_whole_tc` em `data/results/v3/`.

| tabela | script | entrada |
|---|---|---|
| precision, recall, F1 e accuracy por braço, sem e com overlap veto | `experiments/run_experiment.py --score [--overlap-veto] --out FILE` | um arquivo de resultado; grava `*_summary.txt` / `*_ov_summary.txt` ao lado |
| verifier e evidence veto por modelo | `experiments/score_verifier.py PARAM RAG [--overlap-veto]` | os dois braços de um modelo |
| limites de precision por janela e por afirmação | `experiments/window_metrics.py FILE...` | quaisquer arquivos de resultado v1 |
| auditoria das sinalizações off-claim | `experiments/offclaim_sample.py` com `data/labels/offclaim_key.json` | os arquivos de resultado v1 |
| classificadores de referência (all-positive, regras de data e de proveniência, TF-IDF + LR) | `experiments/baselines.py` | os oito arquivos v1 acima, lidos de `data/results/v1/` |
| protocolo de transcrição inteira | `experiments/score_verifier.py` no par `_whole_tc`; leia a linha `video-AND` | `data/results/v3/` |
| comparação de prompts no dev set | `experiments/score_prompt_ab.py data/results/v1/prompt_ab_dev.json` | a execução do A/B de prompts |

Por exemplo, a tabela do verifier com os quatro modelos:

```bash
for m in gemma-4-12b gpt-oss-20b openai-gpt-oss-120b; do
  python experiments/score_verifier.py data/results/v1/test_run_${m}_parametric.json \
                                       data/results/v1/test_run_${m}_tavily.json --overlap-veto
done
python experiments/score_verifier.py data/results/v1/test_run_qwen-qwen3.6-27b_parametric_nothink.json \
                                     data/results/v1/test_run_qwen-qwen3.6-27b_tavily_nothink.json --overlap-veto
```

O `data/results/README.md` indica quais arquivos são reportados e quais são
mantidos apenas como registro. O TCC: *link para o PDF depositado entra aqui*.

## Decisões de projeto

- **YouTube Data API, não scraping.** O scraping quebra quando o markup muda,
  faz IPs serem bloqueados e viola os Termos de Serviço. A API é gratuita
  dentro da cota e pode ser citada.
- **Whisper no áudio, não legendas.** A maioria desses vídeos está com as
  legendas desativadas. Assim, a cobertura depende apenas de o vídeo poder ser
  baixado.
- **Um LLM, não keyword matching.** O dataset traz desinformação e
  desmentidos dos mesmos mitos com vocabulário quase idêntico, e a saída do
  Whisper não tem pontuação, então o escopo de uma negação não pode ser
  recuperado por pattern matching.
- **O difícil é o stance.** Os modelos se prendem ao tópico e sinalizam
  conteúdo de desmentido como desinformação. O prompt pede o stance
  explicitamente, e um stance `denying` ou `neutral` anula a sinalização; o
  `docs/adjudication-experiments.md` registra até onde isso funciona.
- **Nada é ajustado nos vídeos de teste.** Os 119 vídeos v1 rotulados ficam
  separados; todo threshold vem de coleções de dev independentes
  (`docs/dev-sets.md`). Um parâmetro só recebe um valor quando o dev set
  discrimina entre valores; caso contrário, recebe o valor que o torna inerte.
- **Tudo roda na máquina local** com os backends locais. Apenas os dois
  modelos do OpenRouter enviam texto de transcrição para fora, e o preflight
  emite um aviso em toda execução remota.

## Status

| | |
|---|---|
| Test set | 119 vídeos com transcrições utilizáveis, rotulados manualmente: 42 afirmam algo falso, 77 não |
| Dev sets | 44 vídeos transcritos que espelham as queries v1 (ajuste), 19 sobre narrativas que o corpus não cobre (apenas fallback web) |
| Corpus | 21 documentos com licença verificada e DOI resolvido, 2.488 passagens |
| Retrieval | cosseno exato sobre embeddings MiniLM, chunks de 55 palavras, roteamento por tópico com confidence gate; BM25 e híbrido foram testados e descartados |
| Modelos | Gemma 4 12B e gpt-oss-20b via llama.cpp; Qwen3.6-27B e gpt-oss-120b via OpenRouter |
| Protocolos | v1 por janelas (reportado), v2 com time check (testado, pior resultado, descartado), v3 com a transcrição inteira |

Em aberto:

- O fallback web não tem ground truth. Seus vereditos são reportados
  separadamente dos números do corpus.
- Três parâmetros de retrieval estão inertes por falta de evidência, e não por
  se saber que são desnecessários; definí-los exige janelas cuja evidência
  esteja de fato ausente do corpus.
- O chunk de 55 palavras só vence com clareza em afirmações específicas; nas
  amplas, o tamanho do chunk quase não faz diferença.

## Dados e ética

Os vídeos são uploads públicos do YouTube encontrados pela Data API oficial
com queries de tópico neutras; nenhum criador foi alvo específico, e o dataset
contém tanto desinformação quanto desmentidos dos mesmos mitos. As
transcrições são geradas automaticamente a partir do áudio e redistribuídas
aqui para que o experimento possa ser reproduzido com exatidão: os arquivos de
resultado por janelas repetem esse texto em janelas de 140 palavras, e todo
script de avaliação lê as transcrições. O conteúdo falado continua sendo
propriedade dos criadores e não é relicenciado. Cada registro mantém o id do
vídeo e do canal, de modo que um criador pode identificar o seu material e
pedir a remoção; o `experiments/redact_text.py` remove o texto das janelas dos
arquivos de resultado e o restaura a partir de áudio retranscrito, então uma
cópia sem transcrições continua podendo ser avaliada.

As páginas recuperadas pelo fallback web não são redistribuídas. Por
execução, guarda-se apenas um índice das queries, URLs e horários de
recuperação (`data/results/<version>/web_cache_index_*.json`).

## Licença

O código é MIT (`LICENSE`). Os labels são CC BY 4.0. As passagens de
evidência mantêm a licença do documento de onde foram extraídas, e três delas
são não comerciais, então o corpus como um todo não é relicenciado. As
transcrições são apenas para uso em pesquisa. O `LICENSE-DATA.md` lista os
termos de cada arquivo de dados, incluindo a licença de cada documento do
corpus.

## Citação e releases

O `CITATION.cff` contém os metadados de citação que o GitHub exibe em "Cite
this repository"; preencha o DOI do Zenodo e os dados do TCC quando existirem.
O commit correspondente ao TCC entregue recebe a tag `v1.0`, e o TCC cita essa
tag (e o DOI do Zenodo gerado a partir dela). Correções posteriores viram
`v1.1` e assim por diante; a `v1.0` nunca é movida.

```bash
git tag -a v1.0 -m "Versão entregue com o TCC" <commit>
git push origin v1.0
```

## Documentação

| doc | trata de |
|---|---|
| `docs/corpus-inclusion.md` | o que entra no corpus de referência, e por que um artigo foi retirado |
| `docs/chunking-experiments.md` | tamanho e stride dos chunks |
| `docs/retrieval-methods.md` | dense versus BM25 versus híbrido |
| `docs/routing-experiments.md` | roteamento por tópico, o confidence gate e o rank bonus |
| `docs/adjudication-experiments.md` | o relevance gate, o stance e o contrato do veredito |
| `docs/prompt-experiments.md` | o A/B de prompts no dev set |
| `docs/web-search.md` | o fallback web: gatilho, backends, cache |
| `docs/dev-sets.md` | como as coleções de dev foram montadas e mantidas livres do test set |
| `docs/model-backends.md` | llama.cpp, Ollama e OpenRouter, e a comparação entre os quatro modelos |
| `docs/evidence-as-verifier.md`, `docs/evidence-veto.md` | a combinação dos braços paramétrico e com evidência |
| `docs/code-cleanup.md` | a auditoria de código morto antes do experimento final |

## Solução de problemas

| sintoma | causa |
|---|---|
| `ConnectionError: cannot reach` | servidor não iniciado, ou porta errada: o llama.cpp é esperado na 8080, o Ollama na 11434 |
| todo veredito `ERROR` no gpt-oss | a gramática está brigando com o trace de reasoning; inicie o `llama-server` com `--jinja` |
| `unrecognised verdict ''` | gramática desligada com um bloco grande de evidências; você está no Ollama, faça `unset LLM_SERVER` |
| todo veredito `NO_EVIDENCE` | índice vazio ou corpus errado; o `--build-index` deve reportar 2488 |
| `Index is stale` | as passagens mudaram depois que o índice foi construído; reconstrua-o |
| resultados de dois modelos no mesmo arquivo | `LLAMA_MODEL` ficou no padrão; defina-o a cada execução |
| `OPENROUTER_API_KEY not set` / `HTTP 401` | chave ausente, errada, ou conta sem crédito |
| execução muito mais lenta que a estimativa | sem offload para GPU; acrescente `-ngl 99`. Modelos de reasoning também gastam tokens antes de responder |
