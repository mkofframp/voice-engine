import { beforeEach, describe, expect, it, vi } from "vitest";

const h = vi.hoisted(() => ({
  instances: [] as Array<{ opts: any; superMessages: string[]; initDevices: () => Promise<void> }>,
}));

vi.mock("@pipecat-ai/small-webrtc-transport", () => ({
  SmallWebRTCTransport: class {
    opts: any;
    superMessages: string[] = [];
    constructor(opts: any) {
      this.opts = opts;
      h.instances.push(this as any);
    }
    initialize() {}
    async initDevices() {}
    async connect() {}
    async disconnect() {}
    sendMessage() {}
    handleMessage(message: string) {
      this.superMessages.push(message);
    }
  },
}));

import { VoiceEngineClient, type EngineEvent } from "../src/index.js";
import { createWebRtcTransport } from "../src/transport.js";

beforeEach(() => {
  h.instances.length = 0;
});

describe("real transport wiring", () => {
  it("test_offer_includes_token", async () => {
    const client = new VoiceEngineClient({ engineUrl: "http://engine.test/", clientToken: "tok-1" });
    await client.connect();
    const req = h.instances[0].opts.webrtcRequestParams;
    expect(req.endpoint).toBe("http://engine.test/webrtc/offer");
    expect(req.requestData.client_token).toBe("tok-1");
    expect(JSON.stringify(h.instances[0].opts)).not.toMatch(/api[_-]?key/i);
  });
});

describe("ice servers", () => {
  it("passes ice servers from the session response to the transport", async () => {
    const ice = [{ urls: ["turns:turn.cloudflare.com:443?transport=tcp"], username: "u", credential: "c" }];
    const client = new VoiceEngineClient({ engineUrl: "http://e", clientToken: "t", iceServers: ice });
    await client.connect();
    expect(h.instances[0].opts.iceServers).toEqual(ice);
  });

  it("leaves browser defaults when none are given", async () => {
    await new VoiceEngineClient({ engineUrl: "http://e", clientToken: "t" }).connect();
    expect("iceServers" in h.instances[0].opts).toBe(false);
  });
});

describe("handleMessage override", () => {
  async function setup() {
    const t = await createWebRtcTransport({ offerUrl: "http://e/webrtc/offer", clientToken: "t" });
    const got: unknown[] = [];
    t.onMessage((m) => got.push(m));
    return { inst: h.instances[0] as any, got };
  }

  it("passes signalling messages to the base class", async () => {
    const { inst, got } = await setup();
    const msg = JSON.stringify({ type: "signalling", message: { type: "peer_left" } });
    inst.handleMessage(msg);
    expect(inst.superMessages).toEqual([msg]);
    expect(got).toEqual([]);
  });

  it("delivers engine events to the message callback", async () => {
    const { inst, got } = await setup();
    const ev: EngineEvent = { type: "state", state: "speaking" };
    inst.handleMessage(JSON.stringify(ev));
    expect(got).toEqual([ev]);
    expect(inst.superMessages).toEqual([]);
  });

  it("ignores junk without throwing", async () => {
    const { inst, got } = await setup();
    for (const junk of ["not json{", "42", "null", '"str"', "[1,2]", '{"no_type":1}']) {
      expect(() => inst.handleMessage(junk)).not.toThrow();
    }
    expect(inst.superMessages).toEqual([]);
    expect(got).toEqual([]);
  });
});
