import type { Metadata } from "next";
import "./globals.css";
import Providers from "./providers";

export const metadata: Metadata = {
  title: "燕窝智能挑毛监控系统 v1.0",
  description: "Pluck Robot real-time monitoring dashboard",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN">
      <body className="bg-surface text-text antialiased">
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
