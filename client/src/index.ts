import { EVENT_TYPES, type Command, type EngineEvent, type EngineEventType } from "./events.js";
import { createWebRtcTransport, type EngineTransport } from "./transport.js";

export type { Command, EngineEvent, EngineEventType, TurnStage } from "./events.js";
export { EVENT_TYPES } from "./events.js";
export type { EngineTransport } from "./transport.js";

export interface VoiceEngineClientOptions {
  /** Base URL of the engine, for example "http://localhost:8080". */
  engineUrl: string;
  /** Single-use token issued by the host server (POST /sessions). Never an API key. */
  clientToken: string;
  /** Replace the WebRTC transport (used by tests). */
  transport?: EngineTransport;
}

type Listener = (e: EngineEvent) => void;

export class VoiceEngineClient {
  private readonly opts: VoiceEngineClientOptions;
  private transport: EngineTransport | undefined;
  private audio: HTMLAudioElement | undefined;
  private readonly byType = new Map<string, Set<Listener>>();
  private readonly any = new Set<Listener>();

  constructor(opts: VoiceEngineClientOptions) {
    this.opts = opts;
    this.transport = opts.transport;
  }

  async connect(): Promise<void> {
    if (!this.transport) {
      this.transport = await createWebRtcTransport({
        offerUrl: `${this.opts.engineUrl.replace(/\/+$/, "")}/webrtc/offer`,
        clientToken: this.opts.clientToken,
      });
    }
    const t = this.transport;
    t.onMessage((m) => this.dispatch(m));
    t.onRemoteAudio((stream) => this.attachAudio(stream));
    await t.connect();
  }

  disconnect(): void {
    const t = this.transport;
    this.transport = this.opts.transport;
    if (this.audio) {
      this.audio.srcObject = null;
      this.audio.remove();
      this.audio = undefined;
    }
    if (t) void Promise.resolve(t.disconnect()).catch(() => {});
  }

  on<T extends EngineEventType>(type: T, cb: (e: Extract<EngineEvent, { type: T }>) => void): () => void {
    let set = this.byType.get(type);
    if (!set) this.byType.set(type, (set = new Set()));
    const listener = cb as Listener;
    set.add(listener);
    return () => {
      set.delete(listener);
    };
  }

  onAny(cb: (e: EngineEvent) => void): () => void {
    this.any.add(cb);
    return () => {
      this.any.delete(cb);
    };
  }

  sendCommand(cmd: Command): void {
    if (!this.transport) throw new Error("not connected");
    this.transport.send(cmd);
  }

  private dispatch(message: unknown): void {
    if (typeof message !== "object" || message === null) return;
    const type = (message as { type?: unknown }).type;
    if (typeof type !== "string" || !(EVENT_TYPES as readonly string[]).includes(type)) return;
    const event = message as EngineEvent;
    for (const cb of [...(this.byType.get(type) ?? []), ...this.any]) {
      try {
        cb(event);
      } catch {
        // A listener's error must not stop the others or the data channel.
      }
    }
  }

  private attachAudio(stream: MediaStream): void {
    if (typeof document === "undefined") return;
    if (!this.audio) {
      this.audio = document.createElement("audio");
      this.audio.autoplay = true;
      document.body.appendChild(this.audio);
    }
    this.audio.srcObject = stream;
  }
}
