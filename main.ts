// Live Nova: one server, two client wires, and a real free model behind every answer.
//
// This entry serves the OpenAI Chat Completions API and the Anthropic Messages
// API over OpenRouter's fourteen live free models, dialled through the grader's
// gateway on the entrant's pushed key. It is the live twin of Latent Nova
// (free-model-aggregator-01), which routed over ten simulated providers: the
// routing, the failover and the translation are the same shape, but every
// completion here is a real call to a real free model, never a simulation.
//
// WHAT A TRIAL GIVES YOU
//   The trial runs this server as `node /bot/main.ts` from the repository's
//   root, with Node 22's standard library alone (nothing is installed;
//   TypeScript runs directly, and a relative import needs its .ts extension).
//   The env is at $MAKEFUN_ENV_URL; model calls go through the grader's
//   gateway at $OPENAI_BASE_URL, an OpenAI-compatible endpoint that reaches
//   OpenRouter on the key pushed to the competition. The brief at
//   $MAKEFUN_PROMPT_FILE carries two documents:
//     - the free model list: each free OpenRouter model's gateway id
//       (`openrouter/<vendor>/<model>:free`), its published rating in
//       milli-points, what it can do (tools, vision, JSON mode, context
//       length) and its free flag;
//     - the six prompts: each one's id, its api (openai or anthropic),
//       whether it streams, its weight (1 or 2) and the requirements a model
//       must meet to serve it. Each prompt carries a unique token the answer
//       must contain.
//   The server:
//     1. listens on 0.0.0.0 at a free port (never 8080: the env itself is
//        there on one layout);
//     2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>,
//        "catalog": <catalog>};
//     3. serves GET /v1/models (answering in the shape of the auth header it
//        got: x-api-key means Anthropic, Authorization means OpenAI), POST
//        /v1/chat/completions and POST /v1/messages, each with a JSON body or
//        with server-sent events;
//     4. answers each prompt by dialling ONE free model through the gateway.
//        OPENAI_API_KEY in the sandbox is a dummy the gateway swaps for the
//        OpenRouter key pushed to the competition; the real key never enters
//        the sandbox;
//     5. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env
//        says {"done": true}. A process still running when the wall ends
//        scores zero, so never wait for anything else.
//   `node main.ts --list` prints the same catalog JSON and exits 0 before any
//   server starts; the register body's catalog is that same document, byte
//   for byte.
//
// THE MODEL RULE
//   Whatever this server reports in a response's `model` field must be the
//   model it actually dialled. The house records every gateway call and
//   scores the model that really answered. This server sends the brief's
//   model id in the gateway request and puts that same id back on every
//   answer and every streamed chunk (the provider's own answer spells it
//   without the gateway's `openrouter/` prefix), so its claim is the call it
//   made, never another. A prompt whose reported model has no matching
//   successful call in the house's record earns nothing.
//
// ROUTING AND FAILOVER (lib/route.ts)
//   The candidates for a prompt are the brief's free models that meet its
//   requirements, ranked by rating_milli, best first. The best model is
//   tried first; on a 429 or 5xx the prompt fails over to the next best, and
//   every refusal is reported in the response's x-makefun-upstream-failures
//   header, the only way the failover column can see a refusal that was
//   retried and then served. A model that refused once cools down for the
//   rest of the trial, and at most four calls are spent on one prompt. A 400
//   or 401 is the request's own fault, so the prompt is given up at once.
//
// FILES
//   main.ts            this server: the brief, the catalog, the routes, the
//                      register and the poll.
//   lib/providers.ts   the brief's types and parser, the free model list, the
//                      catalog (from the starter, unchanged).
//   lib/route.ts       the routing: ranked candidates, refusal kinds, the
//                      per-trial cooldowns.
//   lib/wire.ts        Anthropic <-> Chat Completions translation, the model
//                      field, the SSE writers (from the starter, unchanged).
//   LICENSE            MIT.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  isRecord,
  readBrief,
  type Brief,
  type CatalogReport,
  type Json,
  type Requirement,
} from './lib/providers.ts';
import { refusalCode, refusalKind, requestedModel, Router, type ModelRow, type UpstreamFailure } from './lib/route.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIJsonWithModel,
  openAIModelList,
  openAIStreamWithModel,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** One gateway call may take this long; the env waits 30 s for one answer. */
const DIAL_TIMEOUT_MS = 25000;
/** The output cap a body gets when it names none; the gateway refuses a call with no cap. */
const DEFAULT_MAX_TOKENS = 512;
/** The gateway refuses an output cap over this. */
const MAX_OUTPUT_TOKENS = 8192;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the entry's gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief: Brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
/** The per-trial routing state: which models have refused and are cooling down. */
const router = new Router();

function log(message: string): void {
  process.stderr.write(`live-nova: ${message}\n`);
}

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1` on the
 * trial network); a base that does not already end in `/v1` gets one, so both spellings reach the route.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return brief.models
    .filter((m) => m.free)
    .map((model) => ({ id: model.id, provider: model.provider, display_name: model.display_name }));
}

/** The requirements of a request, read off its own body: tools, JSON mode, images. */
function requirementsFromBody(body: Json): Requirement {
  let vision = false;
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

/**
 * The body as the gateway takes it: the dialled model, an output cap the
 * gateway requires (named when the client named none, capped at the
 * gateway's own ceiling), and the rest of the client's body unchanged.
 */
function gatewayBody(body: Json, model: string): Json {
  const out: Json = { ...body, model };
  const cap = typeof out.max_tokens === 'number' ? out.max_tokens : typeof out.max_completion_tokens === 'number' ? out.max_completion_tokens : null;
  if (cap === null) out.max_tokens = DEFAULT_MAX_TOKENS;
  else if (cap > MAX_OUTPUT_TOKENS) out.max_tokens = MAX_OUTPUT_TOKENS;
  return out;
}

/** The trail header's value: at most 8 refusal rows, in the order the calls refused. */
function trailHeader(trail: UpstreamFailure[]): string {
  return JSON.stringify(trail.slice(0, 8));
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown, trail: UpstreamFailure[] = []): void {
  const text = JSON.stringify(body);
  const headers: Record<string, string> = { 'content-type': 'application/json', 'content-length': String(Buffer.byteLength(text)) };
  if (trail.length > 0) headers['x-makefun-upstream-failures'] = trailHeader(trail);
  res.writeHead(status, headers);
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string, trail: UpstreamFailure[] = []): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message), trail);
  else sendJson(res, status, openAIError(message, type, null), trail);
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** POST one request to the gateway's Chat Completions route, naming the model this entry dialled. */
async function dial(model: string, body: Json): Promise<Response> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  return fetch(chatCompletionsUrl(), {
    method: 'POST',
    headers,
    body: JSON.stringify(gatewayBody(body, model)),
    signal: AbortSignal.timeout(DIAL_TIMEOUT_MS),
  });
}

/**
 * Dial the gateway until one model answers, failing over down the ranked
 * candidates. `preferred` is a client-named model to try first; `validate`
 * is an optional check on a successful answer (a JSON-mode answer whose
 * content is not JSON fails it, and the prompt fails over). Returns the
 * answering model and its response, plus the trail of every refusal before
 * it; or no model with the trail, when every candidate refused or the
 * request itself was at fault.
 */
async function dialWithFailover(
  body: Json,
  requires: Requirement,
  preferred?: ModelRow,
  validate?: (model: string, upstream: Response) => Promise<boolean>,
): Promise<{ model: string | null; upstream: Response | null; trail: UpstreamFailure[] }> {
  const trail: UpstreamFailure[] = [];
  for (const candidate of router.attemptOrder(brief, requires, preferred)) {
    let upstream: Response;
    try {
      upstream = await dial(candidate.id, body);
    } catch {
      trail.push({ status: 0, code: 'upstream_unreachable', model: candidate.id });
      router.markRefused(candidate.id, null);
      continue;
    }
    if (!upstream.ok || upstream.body === null) {
      const text = await upstream.text().catch(() => '');
      trail.push({ status: upstream.status, code: refusalCode(upstream.status, text), model: candidate.id });
      router.markRefused(candidate.id, upstream.status);
      if (refusalKind(upstream.status) === 'giveup') return { model: null, upstream: null, trail };
      continue;
    }
    if (validate !== undefined) {
      let ok = false;
      try {
        ok = await validate(candidate.id, upstream);
      } catch {
        ok = false;
      }
      if (!ok) {
        trail.push({ status: 200, code: 'invalid_json', model: candidate.id });
        router.markRefused(candidate.id, 200);
        continue;
      }
    }
    log(`served by ${candidate.id}${trail.length > 0 ? ` after ${trail.length} refusal(s)` : ''}`);
    return { model: candidate.id, upstream, trail };
  }
  return { model: null, upstream: null, trail };
}

/** A prompt every candidate refused, as a 502 in the client's wire. */
function refused(res: ServerResponse, wire: 'openai' | 'anthropic', trail: UpstreamFailure[]): void {
  const last = trail[trail.length - 1];
  const detail = last === undefined ? 'no free model answered' : `the gateway answered HTTP ${last.status} (${last.code})`;
  if (wire === 'anthropic') sendError(res, wire, 502, 'api_error', detail, trail);
  else sendError(res, wire, 502, 'upstream_error', detail, trail);
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that meets it. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requested = requestedModel(brief, body.model);
  if (requested.kind === 'unknown') return sendError(res, 'openai', 400, 'model_not_allowed', `the gateway routes no free model named ${requested.id}`);
  const preferred = requested.kind === 'named' ? requested.model : undefined;
  // A JSON-mode answer whose content is not JSON is not an answer: fail over to the next model.
  const wantsJson = isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
  const validate =
    wantsJson && body.stream !== true
      ? async (_model: string, upstream: Response): Promise<boolean> => {
          try {
            const answer: unknown = await upstream.clone().json();
            if (!isRecord(answer) || !Array.isArray(answer.choices)) return false;
            const choice = answer.choices[0];
            if (!isRecord(choice) || !isRecord(choice.message)) return false;
            const content = choice.message.content;
            if (typeof content !== 'string') return false;
            JSON.parse(content);
            return true;
          } catch {
            return false;
          }
        }
      : undefined;
  const served = await dialWithFailover(body, requirementsFromBody(body), preferred, validate);
  if (served.model === null || served.upstream === null) return refused(res, 'openai', served.trail);
  const { model, upstream, trail } = served;
  if (body.stream === true) {
    const headers: Record<string, string> = { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' };
    if (trail.length > 0) headers['x-makefun-upstream-failures'] = trailHeader(trail);
    res.writeHead(200, headers);
    try {
      for await (const frame of openAIStreamWithModel(upstream.body as ReadableStream<Uint8Array>, model)) res.write(frame);
    } finally {
      res.end();
    }
    return;
  }
  const answer: unknown = await upstream.json();
  sendJson(res, 200, openAIJsonWithModel(isRecord(answer) ? answer : {}, model), trail);
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  if (typeof body.max_tokens !== 'number') body.max_tokens = DEFAULT_MAX_TOKENS;
  const requested = requestedModel(brief, body.model);
  if (requested.kind === 'unknown') return sendError(res, 'anthropic', 400, 'invalid_request_error', `the gateway routes no free model named ${requested.id}`);
  const preferred = requested.kind === 'named' ? requested.model : undefined;
  const served = await dialWithFailover(anthropicToOpenAI(body, ''), requirementsFromBody(body), preferred);
  if (served.model === null || served.upstream === null) return refused(res, 'anthropic', served.trail);
  const { model, upstream, trail } = served;
  if (body.stream === true) {
    const headers: Record<string, string> = { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' };
    if (trail.length > 0) headers['x-makefun-upstream-failures'] = trailHeader(trail);
    res.writeHead(200, headers);
    try {
      for await (const event of anthropicEventsFromOpenAI(upstream.body as ReadableStream<Uint8Array>, model)) res.write(event);
    } finally {
      res.end();
    }
    return;
  }
  const answer: unknown = await upstream.json();
  sendJson(res, 200, openAIToAnthropic(isRecord(answer) ? answer : {}, model), trail);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
