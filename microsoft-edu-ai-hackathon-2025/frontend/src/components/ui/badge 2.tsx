import React from "react";

export function Badge({ className = "", variant = "secondary", children }:{
  className?: string; variant?: "secondary" | "outline"; children: React.ReactNode;
}) {
  const styles = variant === "outline"
    ? "border border-slate-300 text-slate-700"
    : "bg-slate-100 text-slate-700";
  return <span className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs ${styles} ${className}`}>{children}</span>;
}
