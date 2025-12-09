"use client";

import Image from "next/image";
import { useState } from "react";
import { cn } from "@/lib/utils";
import type { RobotStats } from "@/store/robot-store";

interface VideoFeedProps {
  stats: RobotStats;
}

export function VideoFeed({ stats }: VideoFeedProps) {
  const [errored, setErrored] = useState(false);

  return (
    <div className="relative aspect-video overflow-hidden rounded-xl border border-border bg-panel shadow-panel">
      <div className="absolute inset-0">
        {!errored ? (
          <Image
            src="/api/stream/video"
            alt="实时视频流"
            fill
            sizes="(min-width: 1280px) 60vw, 100vw"
            unoptimized
            className="object-cover"
            onError={() => setErrored(true)}
          />
        ) : (
          <div className="flex h-full w-full items-center justify-center bg-gradient-to-br from-[#0b1323] to-[#13213b] text-muted">
            视频流加载失败，使用占位图
          </div>
        )}
      </div>

      <div className="absolute inset-0 bg-gradient-to-t from-black/35 via-transparent to-black/10" />

      <div className="absolute left-4 top-3 flex items-center gap-2 rounded-full bg-black/40 px-3 py-1 text-xs text-white backdrop-blur">
        <span className="h-2 w-2 rounded-full bg-accent shadow-[0_0_0_4px_rgba(22,163,74,0.18)]" />
        高清默认视图
      </div>

      <div className="absolute bottom-3 left-4 rounded-md bg-black/50 px-3 py-1.5 text-sm text-white backdrop-blur">
        实时帧率 (FPS): {stats.fps.toFixed(0)}
      </div>

      <div
        className={cn(
          "absolute right-4 top-4 rounded-md px-3 py-1.5 text-xs font-medium",
          stats.confidence >= 0.95
            ? "bg-accent/15 text-accent"
            : "bg-yellow-500/20 text-yellow-200",
        )}
      >
        平均置信度 {(stats.confidence * 100).toFixed(1)}%
      </div>
    </div>
  );
}
