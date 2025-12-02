"use client";

import Link from "next/link";
import { useEffect, useRef } from "react";
import useSWR from "swr";
import { Settings } from "lucide-react";
import { buttonVariants } from "@/components/ui/button";
import { StatusIndicator } from "@/components/StatusIndicator";
import { StatCard } from "@/components/StatCard";
import { ControlButtons } from "@/components/ControlButtons";
import { LogPanel } from "@/components/LogPanel";
import { VideoFeed } from "@/components/VideoFeed";
import { cn } from "@/lib/utils";
import { useRobotStore } from "@/store/robot-store";

type StatusResponse = { running: boolean };

export default function HomePage() {
  const status = useRobotStore((state) => state.status);
  const stats = useRobotStore((state) => state.stats);
  const logs = useRobotStore((state) => state.logs);
  const setStatus = useRobotStore((state) => state.setStatus);
  const setStats = useRobotStore((state) => state.setStats);
  const pushLog = useRobotStore((state) => state.pushLog);
  const mounted = useRef(false);
  const statusErrorMsg = useRef<string | null>(null);
  const statsErrorMsg = useRef<string | null>(null);

  useEffect(() => {
    if (mounted.current) return;
    pushLog({ level: "info", message: "前端面板已加载" });
    mounted.current = true;
  }, [pushLog]);

  const {
    data: statusData,
    error: statusError,
  } = useSWR<StatusResponse>("/api/control/status", {
    refreshInterval: 2000,
  });

  useEffect(() => {
    if (statusData) {
      setStatus(statusData.running ? "running" : "stopped");
    }
  }, [statusData, setStatus]);

  useEffect(() => {
    if (statusError) {
      const msg = statusError.message || "获取算法状态失败";
      if (statusErrorMsg.current !== msg) {
        statusErrorMsg.current = msg;
        pushLog({ level: "error", message: msg });
      }
    } else {
      statusErrorMsg.current = null;
    }
  }, [pushLog, statusError]);

  const {
    data: statsData,
    error: statsError,
  } = useSWR("/api/stream/stats", {
    refreshInterval: 1000,
  });

  useEffect(() => {
    if (statsData) {
      setStats(statsData);
    }
  }, [setStats, statsData]);

  useEffect(() => {
    if (statsError) {
      const msg = statsError.message || "获取统计数据失败";
      if (statsErrorMsg.current !== msg) {
        statsErrorMsg.current = msg;
        pushLog({ level: "error", message: msg });
      }
    } else {
      statsErrorMsg.current = null;
    }
  }, [pushLog, statsError]);

  return (
    <div className="min-h-screen bg-surface text-text">
      <div className="mx-auto flex max-w-screen-2xl flex-col gap-4 p-4 lg:p-6">
        <header className="flex flex-col gap-3 rounded-xl border border-border bg-panel px-5 py-4 shadow-panel md:flex-row md:items-center md:justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-accent/10 text-lg font-semibold text-accent shadow-panel">
              燕
            </div>
            <div>
              <p className="text-lg font-semibold">
                燕窝智能挑毛监控系统{" "}
                <span className="text-sm font-normal text-muted">v1.0</span>
              </p>
              <p className="text-sm text-muted">
                Bird Nest Intelligent Plucking Monitoring
              </p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <div className="rounded-lg border border-border bg-muted/10 px-3 py-2 text-sm text-muted">
              实时监控 · 工业模式
            </div>
            <Link
              href="/settings"
              className={cn(
                buttonVariants({ variant: "secondary", size: "sm" }),
                "gap-2",
              )}
            >
              <Settings className="h-4 w-4" />
              系统设置
            </Link>
          </div>
        </header>

        <div className="grid gap-4 xl:grid-cols-[1.65fr_1fr]">
          <div className="rounded-xl border border-border bg-panel p-3 shadow-panel">
            <VideoFeed stats={stats} />
          </div>

          <div className="grid gap-3">
            <StatusIndicator running={status === "running"} />

            <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-1">
              <StatCard
                title="累计挑出杂质"
                value={stats.totalImpurities}
                unit="个"
              />
              <StatCard
                title="当前视野目标"
                value={stats.currentTargets}
                unit="个"
              />
            </div>

            <ControlButtons />

            <LogPanel logs={logs} />
          </div>
        </div>
      </div>
    </div>
  );
}
