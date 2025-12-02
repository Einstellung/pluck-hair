import { NextResponse } from "next/server";

let running = false;

export async function GET() {
  return NextResponse.json({ running });
}

export async function POST(request: Request) {
  const body = await request.json().catch(() => null);
  if (!body || typeof body.running !== "boolean") {
    return NextResponse.json(
      { error: "Field `running` (boolean) is required" },
      { status: 400 },
    );
  }
  running = body.running;
  return NextResponse.json({ running });
}
