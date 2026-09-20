"use client";

import { useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  Cpu,
  Loader2,
  ScanText,
  ShieldCheck,
} from "lucide-react";

type AnalysisResult = {
  harmful_probability: number;
  unharmful_probability: number;
  risk_level: string;
  decision: string;
  threshold: number;
  feature_850_activation: number;
  device: string;
  error?: string;
};

export default function Analyzer() {
  const [text, setText] = useState("");
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);

  const analyzePrompt = async () => {
    const value = text.trim();

    if (!value || loading) {
      return;
    }

    setLoading(true);
    setResult(null);

    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          text: value,
        }),
        cache: "no-store",
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data?.error || "Analysis request failed."
        );
      }

      setResult(data);

      const existing = localStorage.getItem(
        "neural-feature-history"
      );

      let history: AnalysisResult[] & {
        id?: string;
        text?: string;
        timestamp?: string;
      }[] = [];

      if (existing) {
        try {
          history = JSON.parse(existing);
        } catch {
          history = [];
        }
      }

      const historyItem = {
        id: `${Date.now()}-${Math.random()}`,
        text: value,
        harmful_probability: data.harmful_probability,
        unharmful_probability: data.unharmful_probability,
        risk_level: data.risk_level,
        decision: data.decision,
        threshold: data.threshold,
        feature_850_activation: data.feature_850_activation,
        device: data.device,
        timestamp: new Date().toISOString(),
      };

      localStorage.setItem(
        "neural-feature-history",
        JSON.stringify(
          [historyItem, ...history].slice(0, 50)
        )
      );
    } catch (error) {
      setResult({
        harmful_probability: 0,
        unharmful_probability: 0,
        risk_level: "ERROR",
        decision: "UNAVAILABLE",
        threshold: 0.5,
        feature_850_activation: 0,
        device: "Unknown",
        error:
          error instanceof Error
            ? error.message
            : "Unable to connect to the analysis server.",
      });
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (
    event: React.KeyboardEvent<HTMLTextAreaElement>
  ) => {
    if (event.ctrlKey && event.key === "Enter") {
      event.preventDefault();
      analyzePrompt();
    }
  };

  const harmfulPercentage =
    (result?.harmful_probability ?? 0) * 100;

  const safePercentage =
    (result?.unharmful_probability ?? 0) * 100;

  const blocked =
    result?.decision === "BLOCK";

  return (
    <section className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
      <div className="border-b border-slate-200 px-6 py-5 sm:px-7">
        <div className="flex items-start gap-3">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-cyan-50 ring-1 ring-cyan-100">
            <ScanText className="h-5 w-5 text-cyan-600" />
          </div>

          <div>
            <p className="text-[10px] font-bold uppercase tracking-[0.16em] text-cyan-600">
              Neural Analysis
            </p>

            <h3 className="mt-1 text-base font-bold text-slate-950">
              Prompt Safety Analyzer
            </h3>

            <p className="mt-1 text-xs text-slate-500">
              Evaluate a prompt through the trained neural safety pipeline.
            </p>
          </div>
        </div>
      </div>

      <div className="p-6 sm:p-7">
        <div className="relative">
          <textarea
            value={text}
            onChange={(event) => setText(event.target.value)}
            onKeyDown={handleKeyDown}
            disabled={loading}
            placeholder="Enter a prompt to analyze..."
            className="min-h-[210px] w-full resize-y rounded-xl border border-slate-200 bg-slate-50 px-5 py-4 text-sm leading-6 text-slate-900 outline-none transition placeholder:text-slate-400 focus:border-cyan-400 focus:bg-white focus:ring-4 focus:ring-cyan-50 disabled:cursor-not-allowed disabled:opacity-60"
          />

          <div className="pointer-events-none absolute bottom-4 right-4 rounded-lg bg-white px-2 py-1 text-[9px] font-medium text-slate-400 shadow-sm ring-1 ring-slate-200">
            {text.length} characters
          </div>
        </div>

        <div className="mt-4 flex flex-col justify-between gap-3 sm:flex-row sm:items-center">
          <div className="text-[11px] text-slate-400">
            Press{" "}
            <span className="font-semibold text-slate-600">
              Ctrl + Enter
            </span>{" "}
            to analyze
          </div>

          <button
            type="button"
            onClick={analyzePrompt}
            disabled={!text.trim() || loading}
            className="inline-flex items-center justify-center gap-2 rounded-xl bg-slate-950 px-6 py-3 text-xs font-bold text-white shadow-sm transition hover:bg-cyan-600 disabled:cursor-not-allowed disabled:bg-slate-200 disabled:text-slate-400"
          >
            {loading ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <ScanText className="h-4 w-4" />
                Analyze Prompt
              </>
            )}
          </button>
        </div>

        {result && (
          <div className="mt-7 border-t border-slate-200 pt-7">
            {result.error ? (
              <div className="rounded-xl border border-red-200 bg-red-50 p-5">
                <div className="flex items-center gap-3">
                  <AlertTriangle className="h-5 w-5 text-red-600" />

                  <div>
                    <p className="text-sm font-bold text-red-700">
                      Analysis unavailable
                    </p>

                    <p className="mt-1 text-xs text-red-600">
                      {result.error}
                    </p>
                  </div>
                </div>
              </div>
            ) : (
              <>
                <div className="grid gap-4 lg:grid-cols-[1.35fr_0.65fr]">
                  <div
                    className={`rounded-2xl border p-6 ${
                      blocked
                        ? "border-red-200 bg-red-50"
                        : "border-emerald-200 bg-emerald-50"
                    }`}
                  >
                    <div className="flex items-start justify-between gap-4">
                      <div>
                        <p className="text-[10px] font-bold uppercase tracking-[0.15em] text-slate-500">
                          Safety Decision
                        </p>

                        <h4
                          className={`mt-2 text-3xl font-bold ${
                            blocked
                              ? "text-red-700"
                              : "text-emerald-700"
                          }`}
                        >
                          {result.decision}
                        </h4>

                        <p className="mt-2 text-xs text-slate-500">
                          {result.risk_level}
                        </p>
                      </div>

                      <div
                        className={`flex h-12 w-12 items-center justify-center rounded-xl ${
                          blocked
                            ? "bg-red-100"
                            : "bg-emerald-100"
                        }`}
                      >
                        {blocked ? (
                          <AlertTriangle className="h-6 w-6 text-red-600" />
                        ) : (
                          <CheckCircle2 className="h-6 w-6 text-emerald-600" />
                        )}
                      </div>
                    </div>

                    <div className="mt-6">
                      <div className="mb-2 flex justify-between text-[10px] font-medium text-slate-500">
                        <span>Harmful probability</span>

                        <span>
                          {harmfulPercentage.toFixed(2)}%
                        </span>
                      </div>

                      <div className="h-2.5 overflow-hidden rounded-full bg-white/80">
                        <div
                          className={`h-full rounded-full transition-all duration-500 ${
                            blocked
                              ? "bg-red-500"
                              : "bg-emerald-500"
                          }`}
                          style={{
                            width: `${harmfulPercentage}%`,
                          }}
                        />
                      </div>

                      <div className="mt-2 flex justify-between text-[10px] text-slate-400">
                        <span>0%</span>

                        <span>
                          Threshold{" "}
                          {(result.threshold * 100).toFixed(0)}%
                        </span>

                        <span>100%</span>
                      </div>
                    </div>
                  </div>

                  <div className="rounded-2xl border border-slate-200 bg-slate-50 p-6">
                    <p className="text-[10px] font-bold uppercase tracking-[0.15em] text-slate-400">
                      Model Confidence
                    </p>

                    <div className="mt-5 space-y-5">
                      <div>
                        <div className="flex justify-between text-xs">
                          <span className="text-slate-500">
                            Harmful
                          </span>

                          <span className="font-bold text-slate-800">
                            {harmfulPercentage.toFixed(2)}%
                          </span>
                        </div>

                        <div className="mt-2 h-1.5 rounded-full bg-slate-200">
                          <div
                            className="h-full rounded-full bg-red-400"
                            style={{
                              width: `${harmfulPercentage}%`,
                            }}
                          />
                        </div>
                      </div>

                      <div>
                        <div className="flex justify-between text-xs">
                          <span className="text-slate-500">
                            Unharmful
                          </span>

                          <span className="font-bold text-slate-800">
                            {safePercentage.toFixed(2)}%
                          </span>
                        </div>

                        <div className="mt-2 h-1.5 rounded-full bg-slate-200">
                          <div
                            className="h-full rounded-full bg-emerald-400"
                            style={{
                              width: `${safePercentage}%`,
                            }}
                          />
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="mt-4 grid gap-3 sm:grid-cols-3">
                  <div className="rounded-xl border border-slate-200 bg-white p-4">
                    <p className="text-[9px] font-bold uppercase tracking-wider text-slate-400">
                      Feature 850
                    </p>

                    <p className="mt-2 font-mono text-lg font-bold text-cyan-600">
                      {result.feature_850_activation.toFixed(4)}
                    </p>

                    <p className="mt-1 text-[10px] text-slate-400">
                      Observed activation
                    </p>
                  </div>

                  <div className="rounded-xl border border-slate-200 bg-white p-4">
                    <p className="text-[9px] font-bold uppercase tracking-wider text-slate-400">
                      Threshold
                    </p>

                    <p className="mt-2 text-lg font-bold text-slate-900">
                      {result.threshold.toFixed(2)}
                    </p>

                    <p className="mt-1 text-[10px] text-slate-400">
                      Decision boundary
                    </p>
                  </div>

                  <div className="rounded-xl border border-slate-200 bg-white p-4">
                    <p className="text-[9px] font-bold uppercase tracking-wider text-slate-400">
                      Inference
                    </p>

                    <div className="mt-2 flex items-center gap-2">
                      <Cpu className="h-4 w-4 text-cyan-600" />

                      <p className="text-lg font-bold text-slate-900">
                        {result.device}
                      </p>
                    </div>

                    <p className="mt-1 text-[10px] text-slate-400">
                      Runtime device
                    </p>
                  </div>
                </div>
              </>
            )}
          </div>
        )}
      </div>
    </section>
  );
}