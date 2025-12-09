import { NextResponse } from "next/server";

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? process.env.API_BASE ?? "http://localhost:8000/api";

export async function GET() {
  const upstream = `${API_BASE.replace(/\/$/, "")}/stream/video`;

  try {
    const resp = await fetch(upstream, { cache: "no-store" });

    if (!resp.ok || !resp.body) {
      return NextResponse.json(
        { error: `Upstream video unavailable (${resp.status})` },
        { status: 502 },
      );
    }

    const headers = new Headers();
    headers.set(
      "Content-Type",
      resp.headers.get("content-type") ?? "multipart/x-mixed-replace; boundary=frame",
    );
    headers.set("Cache-Control", "no-store");

    return new NextResponse(resp.body as unknown as BodyInit, {
      status: resp.status,
      headers,
    });
  } catch (error) {
    return NextResponse.json(
      { error: "Failed to fetch upstream video stream", detail: String(error) },
      { status: 502 },
    );
  }
}
