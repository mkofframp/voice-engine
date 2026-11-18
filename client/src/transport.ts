/** What VoiceEngineClient needs from a transport. Tests inject a fake one. */
export interface EngineTransport {
  /** Ask for the mic, send the WebRTC offer, resolve when the data channel is open. */
  connect(): Promise<void>;
  disconnect(): void | Promise<void>;
  /** Send one JSON message over the data channel. */
  send(message: unknown): void;
  /** Called with each JSON message the engine sends. */
  onMessage(cb: (message: unknown) => void): void;
  /** Called with the bot's audio stream when it arrives. */
  onRemoteAudio(cb: (stream: MediaStream) => void): void;
}

export interface WebRtcTransportOptions {
  /** Full URL of POST /webrtc/offer. */
  offerUrl: string;
  clientToken: string;
  /** STUN/TURN servers from the session response (`ice_servers`); empty = browser defaults. */
  iceServers?: RTCIceServer[];
}

/**
 * The real transport: Pipecat's SmallWebRTC client transport. It is imported on
 * first use so that code using a fake transport never loads browser-only modules.
 */
export async function createWebRtcTransport(opts: WebRtcTransportOptions): Promise<EngineTransport> {
  const { SmallWebRTCTransport } = await import("@pipecat-ai/small-webrtc-transport");

  let messageCb: (m: unknown) => void = () => {};
  let audioCb: (s: MediaStream) => void = () => {};

  // The engine sends plain `{type, ...payload}` messages. Pipecat's transport
  // only forwards RTVI-labelled messages, so deliver everything else ourselves
  // and leave its own signalling messages to the base class.
  class EngineWebRtcTransport extends SmallWebRTCTransport {
    override handleMessage(message: string): void {
      let parsed: unknown;
      try {
        parsed = JSON.parse(message);
      } catch {
        return;
      }
      if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) return;
      const type = (parsed as { type?: unknown }).type;
      if (typeof type !== "string") return;
      if (type === "signalling") {
        super.handleMessage(message);
        return;
      }
      messageCb(parsed);
    }
  }

  const transport = new EngineWebRtcTransport({
    waitForICEGathering: true,
    ...(opts.iceServers?.length ? { iceServers: opts.iceServers } : {}),
    webrtcRequestParams: {
      endpoint: opts.offerUrl,
      requestData: { client_token: opts.clientToken },
    },
  });

  transport.initialize(
    {
      enableMic: true,
      enableCam: false,
      callbacks: {
        onTrackStarted: (track: MediaStreamTrack, participant?: { local?: boolean }) => {
          if (track.kind === "audio" && !participant?.local) audioCb(new MediaStream([track]));
        },
      },
    } as never,
    () => {},
  );

  return {
    async connect() {
      await transport.initDevices();
      await transport.connect();
    },
    disconnect() {
      return transport.disconnect();
    },
    send(message) {
      transport.sendMessage(message as never);
    },
    onMessage(cb) {
      messageCb = cb;
    },
    onRemoteAudio(cb) {
      audioCb = cb;
    },
  };
}
