import { useState, useEffect } from 'react';
import { clsx } from 'clsx';

export function KillSwitchTimer({ active = false }: { active?: boolean }) {
    const [timeLeft, setTimeLeft] = useState(30); // 30s countdown

    useEffect(() => {
        if (!active) return;
        const interval = setInterval(() => {
            setTimeLeft(t => t > 0 ? t - 0.1 : 0);
        }, 100);
        return () => clearInterval(interval);
    }, [active]);

    return (
        <div className={clsx(
            "relative overflow-hidden rounded border p-2 flex items-center justify-between transition-colors",
            active ? "bg-red-900/20 border-red-500 animate-pulse" : "bg-black/40 border-gray-800"
        )}>
            <div className="flex flex-col">
                <span className={clsx("text-[10px] font-bold uppercase tracking-widest", active ? "text-red-500" : "text-gray-600")}>
                    {active ? "KILL SWITCH ENGAGED" : "SYSTEM NORMAL"}
                </span>
                <span className="text-xs text-gray-500">Auto-shutdown in:</span>
            </div>
            <div className={clsx("font-mono text-2xl font-black", active ? "text-red-500" : "text-gray-700")}>
                {active ? timeLeft.toFixed(1) : "--.--"}s
            </div>

            {/* Progress Bar Background */}
            {active && (
                <div
                    className="absolute bottom-0 left-0 h-1 bg-red-500 transition-all duration-100 ease-linear"
                    style={{ width: `${(timeLeft / 30) * 100}%` }}
                />
            )}
        </div>
    );
}
