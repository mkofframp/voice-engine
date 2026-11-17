import { describe, expect, it } from "vitest";
import { VoiceEngineClient, type EngineEvent, type EngineTransport } from "../src/index.js";

class FakeTransport implements EngineTransport {
  sent: unknown[] = [];
  connected = false;
  offers: unknown[] = [];
  private msg: (m: unknown) => void = () => {};
  constructor(private clientToken: string) {}
  async connect() {
    // The real transport posts the offer with the token; the fake records it.
    this.offers.push({ sdp: "x", type: "offer", client_token: this.clientToken });
    this.connected = true;
  }
  disconnect() {
    this.connected = false;
  }
  send(m: unknown) {
    this.sent.push(m);
  }
  onMessage(cb: (m: unknown) => void) {
    this.msg = cb;
  }
  onRemoteAudio() {}
  emit(m: unknown) {
    this.msg(m);
  }
}

async function setup() {
  const transport = new FakeTransport("tok-1");
  const client = new VoiceEngineClient({ engineUrl: "http://e", clientToken: "tok-1", transport });
  await client.connect();
  return { transport, client };
}

describe("VoiceEngineClient", () => {
  it("test_routes_events_by_type", async () => {
    const { transport, client } = await setup();
    const states: string[] = [];
    const sentences: string[] = [];
    const all: EngineEvent[] = [];
    client.on("state", (e) => states.push(e.state));
    client.on("bot_sentence", (e) => sentences.push(`${e.emotion}:${e.text}`));
    client.onAny((e) => all.push(e));
    transport.emit({ type: "state", state: "speaking" });
    transport.emit({ type: "bot_sentence", emotion: "happy", text: "Hi!" });
    expect(states).toEqual(["speaking"]);
    expect(sentences).toEqual(["happy:Hi!"]);
    expect(all.map((e) => e.type)).toEqual(["state", "bot_sentence"]);
  });

  it("test_unsubscribe", async () => {
    const { transport, client } = await setup();
    const seen: string[] = [];
    const off = client.on("state", (e) => seen.push(e.state));
    transport.emit({ type: "state", state: "listening" });
    off();
    transport.emit({ type: "state", state: "thinking" });
    expect(seen).toEqual(["listening"]);
  });

  it("test_offer_includes_token", async () => {
    const { transport } = await setup();
    expect(transport.connected).toBe(true);
    expect(transport.offers).toEqual([{ sdp: "x", type: "offer", client_token: "tok-1" }]);
  });

  it("test_unknown_event_type_ignored_not_thrown", async () => {
    const { transport, client } = await setup();
    const seen: string[] = [];
    client.onAny((e) => seen.push(e.type));
    expect(() => {
      transport.emit({ type: "from_the_future", x: 1 });
      transport.emit("junk");
      transport.emit(null);
      transport.emit({ no_type: true });
      transport.emit({ type: "state", state: "listening" });
    }).not.toThrow();
    expect(seen).toEqual(["state"]);
  });

  it("sends commands over the data channel unchanged", async () => {
    const { transport, client } = await setup();
    client.sendCommand({ type: "say", text: "Hello" });
    client.sendCommand({ type: "end", graceful: false });
    expect(transport.sent).toEqual([
      { type: "say", text: "Hello" },
      { type: "end", graceful: false },
    ]);
  });

  it("a throwing listener does not stop the others", async () => {
    const { transport, client } = await setup();
    const seen: string[] = [];
    client.on("state", () => {
      throw new Error("boom");
    });
    client.on("state", (e) => seen.push(e.state));
    transport.emit({ type: "state", state: "thinking" });
    expect(seen).toEqual(["thinking"]);
  });
});
