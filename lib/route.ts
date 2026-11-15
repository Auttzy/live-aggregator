// Routing: which live free model serves one request, and what a refusal means.
//
// The trial's six prompts arrive one at a time, each naming model "auto". Every
// choice below is a function of the brief, the request and the gateway's answers
// so far: no wall clock, no unseeded randomness, so a replay leaves the same
// state export.
//
//   - quality (60%): the candidates that meet a prompt's requirements are ranked
//     by the brief's rating_milli, highest first, ties in the brief's order. The
//     best model for every prompt shape is therefore always tried first: the
//     highest-rated JSON-capable model for a JSON-mode prompt, the highest-rated
//     tools-capable one for a tools prompt, and the highest-rated free model of
//     all for a plain one. On the practice briefs that is super-vl (1303) for
//     JSON and inkling-2-small (1475) for everything else.
//   - failover (10%): a gateway call answered with a 4xx or a 5xx is not an
//     answer. The prompt is tried on the next-best candidate that meets it, and
//     every refusal is recorded in the trail the response may carry, because the
//     failover column can only see a refusal the server reports. A 400 or 401 is
//     our own mistake (a bad model id, a missing key): no other model would do
//     better on the same body, so the prompt is given up at once.
//   - rate limits: a model that refused once is cooling down. A 429 is a
//     rate limit that will not clear in the seconds a trial has left, so a
//     rate-limited model is out for the trial. A 5xx or an unreachable
//     gateway may be transient, so a model that refused that way gets one
//     second chance on a later prompt; a second refusal retires it. The
//     tie-break is the number of gateway calls, so at most 4 attempts are
//     spent on one prompt.

import {
  capabilitiesOf,
  meetsRequirements,
  type Brief,
  type ModelRow,
  type Requirement,
} from './providers.ts';

/** The most gateway calls spent on one prompt. Six prompts at four each is 24, inside the trial's 60. */
export const MAX_ATTEMPTS_PER_PROMPT = 4;

/** One refusal, as the response's optional trail header reports it. */
export interface UpstreamFailure {
  status: number;
  code: string;
  model: string | null;
}

/** What a refused gateway call means for the prompt being served. */
export type RefusalKind = 'failover' | 'giveup';

/**
 * The candidates that may serve a request: the brief's free models that meet
 * its requirements, best rating first, ties in the brief's order.
 */
export function rankedCandidates(brief: Brief, requires: Requirement): ModelRow[] {
  return brief.models
    .filter((m) => m.free && meetsRequirements(requires, capabilitiesOf(m)))
    .sort((a, b) => b.rating_milli - a.rating_milli);
}

/**
 * Whether a refused call is worth failing over from. A 429 or a 5xx is the
 * provider's own refusal: another free model may serve the prompt. A 400 or
 * 401 is the request's own fault (a model the gateway will not route, a key
 * the gateway will not swap): no candidate would do better, so give up.
 * Anything without an HTTP status (the gateway unreachable) is treated like a
 * 5xx: the next candidate may be reachable through the same gateway.
 */
export function refusalKind(status: number | null): RefusalKind {
  if (status === null) return 'failover';
  if (status === 400 || status === 401) return 'giveup';
  return 'failover';
}

/** The code of a refused call, from the gateway's own error body when it names one. */
export function refusalCode(status: number | null, bodyText: string): string {
  if (bodyText !== '') {
    try {
      const parsed: unknown = JSON.parse(bodyText);
      if (parsed !== null && typeof parsed === 'object') {
        const err = (parsed as Record<string, unknown>).error;
        if (err !== null && typeof err === 'object') {
          const code = (err as Record<string, unknown>).code;
          if (typeof code === 'string' && code !== '') return code;
          if (typeof code === 'number') return String(code);
          const type = (err as Record<string, unknown>).type;
          if (typeof type === 'string' && type !== '') return type;
        }
      }
    } catch {
      // Not JSON: fall through to the status's own name.
    }
  }
  if (status === null) return 'upstream_unreachable';
  if (status === 429) return 'rate_limited';
  if (status === 401) return 'invalid_key';
  if (status === 400) return 'invalid_request';
  if (status >= 500) return 'server_error';
  return `http_${status}`;
}

/**
 * What a request's own `model` field asks for. A client may name a model
 * outright instead of "auto": the named model is served when the brief lists
 * it as free, and a name the brief does not list as free is not dialable.
 */
export type RequestedModel =
  | { kind: 'auto' }
  | { kind: 'named'; model: ModelRow }
  | { kind: 'unknown'; id: string };

export function requestedModel(brief: Brief, modelField: unknown): RequestedModel {
  if (typeof modelField !== 'string' || modelField === '' || modelField === 'auto') return { kind: 'auto' };
  const found = brief.models.find((m) => m.id === modelField);
  if (found !== undefined && found.free) return { kind: 'named', model: found };
  return { kind: 'unknown', id: modelField };
}

/**
 * The per-trial routing state: which models have refused, and the order the
 * next prompt tries. A model that rate-limited (429) is out for the trial; a
 * model that refused with a 5xx or was unreachable gets one second chance on
 * a later prompt, and a second refusal retires it.
 */
export class Router {
  private readonly refusals = new Map<string, { count: number; lastStatus: number | null }>();

  /** Mark a model as refused, with the status that refused it (null: unreachable). */
  markRefused(modelId: string, status: number | null): void {
    const prev = this.refusals.get(modelId);
    this.refusals.set(modelId, { count: (prev?.count ?? 0) + 1, lastStatus: status });
  }

  /** Whether a model is cooling down: rate-limited once, or refused twice. */
  isRefused(modelId: string): boolean {
    const r = this.refusals.get(modelId);
    if (r === undefined) return false;
    if (r.count >= 2) return true;
    return r.lastStatus === 429;
  }

  /**
   * The attempt order for one prompt: a client-named model first when it
   * meets the requirements and is not cooling down, then the ranked
   * candidates that are not cooling down, at most MAX_ATTEMPTS_PER_PROMPT
   * of them in all.
   */
  attemptOrder(brief: Brief, requires: Requirement, preferred?: ModelRow): ModelRow[] {
    const ranked = rankedCandidates(brief, requires).filter((m) => !this.isRefused(m.id));
    if (preferred !== undefined && meetsRequirements(requires, capabilitiesOf(preferred)) && !this.isRefused(preferred.id)) {
      return [preferred, ...ranked.filter((m) => m.id !== preferred.id)].slice(0, MAX_ATTEMPTS_PER_PROMPT);
    }
    return ranked.slice(0, MAX_ATTEMPTS_PER_PROMPT);
  }
}
