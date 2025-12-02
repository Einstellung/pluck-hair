import { NextResponse } from "next/server";

let totalImpurities = 1245;
let currentTargets = 8;
const startedAt = Date.now();

export async function GET() {
  totalImpurities += Math.floor(Math.random() * 3);
  currentTargets = Math.max(
    0,
    Math.min(12, currentTargets + (Math.random() > 0.6 ? 1 : -1)),
  );

  const fps = 28 + Math.random() * 5;
  const confidence = 0.92 + Math.random() * 0.06;
  const durationSec = Math.floor((Date.now() - startedAt) / 1000);

  return NextResponse.json({
    fps: Number(fps.toFixed(1)),
    totalImpurities,
    currentTargets,
    durationSec,
    confidence: Number(confidence.toFixed(3)),
  });
}
