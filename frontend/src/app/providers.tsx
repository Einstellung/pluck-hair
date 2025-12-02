"use client";

import { SWRConfig } from "swr";
import { ReactNode } from "react";
import { useSettingsPersistence } from "@/store/robot-store";

const fetcher = async (url: string) => {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error("请求失败");
  }
  return res.json();
};

export default function Providers({ children }: { children: ReactNode }) {
  useSettingsPersistence();

  return (
    <SWRConfig
      value={{
        fetcher,
        revalidateOnFocus: false,
        refreshWhenHidden: false,
        errorRetryInterval: 4000,
      }}
    >
      {children}
    </SWRConfig>
  );
}
