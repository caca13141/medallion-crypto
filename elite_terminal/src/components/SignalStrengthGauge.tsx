import { useMemo } from 'react';
import { TrendingUp, TrendingDown, Activity } from 'lucide-react';
import { clsx } from 'clsx';

interface SignalStrengthProps {
    tti?: number;
    loopScore?: number;
    whaleAlignment?: 'aligned' | 'divergent' | 'neutral';
    predictionAccuracy?: number;
    bullLoops?: number;
    bearLoops?: number;
    bifiltration?: number;
    smartMoneyScore?: number;
}

export function SignalStrengthGauge({
    tti = 0,
    loopScore = 0,
    whaleAlignment = 'neutral',
    predictionAccuracy = 0.5,
    bullLoops = 0,
    bearLoops = 0,
    bifiltration = 0,
    smartMoneyScore = 0
}: SignalStrengthProps) {

    const signalStrength = useMemo(() => {
        // Normalize TTI (typically 0-10, higher = stronger)
        const ttiScore = Math.min(tti / 10, 1);

        // Normalize Loop Score (typically 0-5, higher = more complex pattern)
        const loopNormalized = Math.min(loopScore / 5, 1);

        // Whale alignment bonus/penalty
        const whaleBonus = whaleAlignment === 'aligned' ? 0.25 : whaleAlignment === 'divergent' ? -0.2 : 0;

        // Bifiltration Bonus (Volume confirmation)
        // If bifiltration score is high, it means volume supports the structure
        const bifBonus = Math.min(bifiltration / 100, 0.1);

        // Smart Money Bonus (On-Chain)
        const smartMoneyBonus = Math.min(smartMoneyScore / 100, 0.15);

        // Combine (weighted average + whale adjustment)
        let combined = (ttiScore * 0.3 + loopNormalized * 0.2 + predictionAccuracy * 0.2) + whaleBonus + bifBonus + smartMoneyBonus;

        // Clamp to 0-1
        return Math.max(0, Math.min(1, combined));
    }, [tti, loopScore, whaleAlignment, predictionAccuracy, bifiltration, smartMoneyScore]);

    const getStrengthLabel = () => {
        if (signalStrength < 0.3) return 'WEAK';
        if (signalStrength < 0.6) return 'MODERATE';
        if (signalStrength < 0.8) return 'STRONG';
        return 'VERY STRONG';
    };

    const getColor = () => {
        if (signalStrength < 0.3) return 'from-red-500 to-red-700';
        if (signalStrength < 0.6) return 'from-yellow-500 to-yellow-700';
        if (signalStrength < 0.8) return 'from-cyan-500 to-cyan-700';
        return 'from-green-400 to-green-600';
    };

    const getGlowColor = () => {
        if (signalStrength < 0.3) return 'shadow-[0_0_20px_rgba(239,68,68,0.5)]';
        if (signalStrength < 0.6) return 'shadow-[0_0_20px_rgba(234,179,8,0.5)]';
        if (signalStrength < 0.8) return 'shadow-[0_0_20px_rgba(6,182,212,0.5)]';
        return 'shadow-[0_0_20px_rgba(74,222,128,0.5)]';
    };

    return (
        <div className="w-full h-full flex flex-col justify-center items-center gap-3 p-4">
            {/* Circular Gauge */}
            <div className="relative w-32 h-32">
                {/* Background Circle */}
                <svg className="w-full h-full transform -rotate-90">
                    <circle
                        cx="64"
                        cy="64"
                        r="56"
                        stroke="rgba(255,255,255,0.1)"
                        strokeWidth="8"
                        fill="none"
                    />
                    {/* Progress Circle */}
                    <circle
                        cx="64"
                        cy="64"
                        r="56"
                        stroke="url(#gradient)"
                        strokeWidth="8"
                        fill="none"
                        strokeDasharray={`${2 * Math.PI * 56}`}
                        strokeDashoffset={`${2 * Math.PI * 56 * (1 - signalStrength)}`}
                        className={clsx("transition-all duration-1000", getGlowColor())}
                        strokeLinecap="round"
                    />
                    <defs>
                        <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="100%">
                            <stop offset="0%" className={clsx("transition-colors duration-500")} stopColor={signalStrength < 0.3 ? '#ef4444' : signalStrength < 0.6 ? '#eab308' : signalStrength < 0.8 ? '#06b6d4' : '#4ade80'} />
                            <stop offset="100%" className={clsx("transition-colors duration-500")} stopColor={signalStrength < 0.3 ? '#b91c1c' : signalStrength < 0.6 ? '#a16207' : signalStrength < 0.8 ? '#0e7490' : '#16a34a'} />
                        </linearGradient>
                    </defs>
                </svg>

                {/* Center Text */}
                <div className="absolute inset-0 flex flex-col items-center justify-center">
                    <div className={clsx(
                        "text-3xl font-bold transition-colors duration-500",
                        signalStrength < 0.3 ? "text-red-400" :
                            signalStrength < 0.6 ? "text-yellow-400" :
                                signalStrength < 0.8 ? "text-cyan-400" :
                                    "text-green-400"
                    )}>
                        {Math.round(signalStrength * 100)}%
                    </div>
                    <div className="text-[8px] text-gray-500 font-mono tracking-wider mt-1">
                        CONFIDENCE
                    </div>
                </div>
            </div>

            {/* Label */}
            <div className={clsx(
                "text-sm font-bold tracking-wider px-3 py-1 rounded border transition-all duration-500",
                signalStrength < 0.3 ? "text-red-400 border-red-500/30 bg-red-950/20" :
                    signalStrength < 0.6 ? "text-yellow-400 border-yellow-500/30 bg-yellow-950/20" :
                        signalStrength < 0.8 ? "text-cyan-400 border-cyan-500/30 bg-cyan-950/20" :
                            "text-green-400 border-green-500/30 bg-green-950/20"
            )}>
                {getStrengthLabel()}
            </div>

            {/* Breakdown */}
            <div className="w-full flex flex-col gap-1 text-[9px] font-mono text-gray-500">
                <div className="flex justify-between">
                    <span>TTI:</span>
                    <span className="text-cyan-400">{tti.toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                    <span>Loop:</span>
                    <span className="text-purple-400">{loopScore.toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                    <span>Whales:</span>
                    <span className={clsx(
                        whaleAlignment === 'aligned' ? "text-green-400" :
                            whaleAlignment === 'divergent' ? "text-red-400" :
                                "text-gray-400"
                    )}>
                        {whaleAlignment.toUpperCase()}
                    </span>
                </div>
            </div>
        </div>
    );
}
