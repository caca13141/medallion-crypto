import React, { useEffect, useState, Component, ReactNode } from 'react';

// Elite Components
import { AlphaDecay } from './components/AlphaDecay';
import { BacktestPanel } from './components/BacktestPanel';
import { ForecastPanel } from './components/ForecastPanel';
import { OrderBookHeatmap } from './components/OrderBookHeatmap';
import { QuantLab } from './components/QuantLab';
import { RegimeFingerprint } from './components/RegimeFingerprint';
import { RiskPanel } from './components/RiskPanel';
import { SignalPanel } from './components/SignalPanel';
import { TradeFeed } from './components/TradeFeed';

// Error Boundary for Component-level Resilience
class ErrorBoundary extends Component<{ children: ReactNode, name: string }, { hasError: boolean }> {
    constructor(props: any) {
        super(props);
        this.state = { hasError: false };
    }
    static getDerivedStateFromError() { return { hasError: true }; }
    render() {
        if (this.state.hasError) {
            return (
                <div className="flex flex-col items-center justify-center h-full bg-orange-500 text-white font-black p-4 border-4 border-black text-center">
                    <div className="text-[14px]">CRASH_DETECTED</div>
                    <div className="text-[10px] opacity-75">{this.props.name}</div>
                </div>
            );
        }
        return this.props.children;
    }
}

function App() {
    const [connected, setConnected] = useState(false);
    const [l3Data, setL3Data] = useState<any>(null);
    const [signals, setSignals] = useState<any>(null);
    const [riskState, setRiskState] = useState<any>(null);
    const [forecastData, setForecastData] = useState<any>(null);
    const [backtestData, setBacktestData] = useState<any>(null);
    const [trades, setTrades] = useState<any[]>([]);
    const [mcSimulation, setMcSimulation] = useState<any>(null);
    const [intelligenceAudit, setIntelligenceAudit] = useState<any>(null);
    const [pulseCount, setPulseCount] = useState(0);
    const [activeTab, setActiveTab] = useState<'monitor' | 'analysis'>('monitor');
    const [lastHeartbeat, setLastHeartbeat] = useState(Date.now());
    const [currentTime, setCurrentTime] = useState(Date.now());

    useEffect(() => {
        const timer = setInterval(() => setCurrentTime(Date.now()), 1000);
        return () => clearInterval(timer);
    }, []);

    useEffect(() => {
        let ws: WebSocket;
        let reconnectTimer: any;

        const connect = () => {
            ws = new WebSocket('ws://localhost:3000/ws');
            ws.onopen = () => setConnected(true);
            // The following line was provided in the instructions but is syntactically incorrect for a WebSocket object.
            // Assuming the intent was to add a status indicator, but without a clear place to render it,
            // and to maintain syntactic correctness, this line cannot be directly inserted as `ws.const ...`.
            // If 'Exascale' was meant to be replaced, it was not found in the original document.
            // Therefore, this specific instruction cannot be applied faithfully and syntactically correctly as written.
            // For the purpose of this exercise, I will proceed by *not* inserting the syntactically incorrect line.
            // If the intent was to display "Alpha Core Integration: Active" in the UI,
            // please provide a valid React component definition and a location in the JSX to render it.
            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    setLastHeartbeat(Date.now());
                    if (data.topic === 'marksman_pulse') {
                        setPulseCount(prev => prev + 1);
                        const p = data.payload;
                        if (p.l3_book) setL3Data(p.l3_book);
                        if (p.signals) setSignals(p.signals);
                        if (p.risk) setRiskState(p.risk);
                        if (p.forecast) setForecastData(p.forecast);
                        if (p.backtest) setBacktestData(p.backtest);
                        if (p.monte_carlo) setMcSimulation(p.monte_carlo);
                        if (p.intelligence_audit) setIntelligenceAudit(p.intelligence_audit);
                        if (p.trades_batch) {
                            setTrades(prev => [...p.trades_batch, ...prev].slice(0, 100));
                        }
                    }
                } catch (e) { console.error("Pulse Parse Error", e); }
            };
            ws.onclose = () => {
                setConnected(false);
                reconnectTimer = setTimeout(connect, 2000);
            };
        };

        connect();
        return () => {
            if (ws) ws.close();
            clearTimeout(reconnectTimer);
        };
    }, []);

    const isOffline = Date.now() - lastHeartbeat > 7000;

    return (
        <div className="w-screen h-screen bg-[#FFFFFF] text-[#111111] overflow-hidden flex flex-col font-sans selection:bg-blue-50">
            {/* DIAGNOSTIC OVERLAY */}
            <div className="bg-black text-[10px] text-white p-1 px-4 flex gap-4 font-mono z-[9999] opacity-90">
                <span className={connected ? "text-green-400" : "text-red-500"}>WS: {connected ? 'CONNECTED' : 'DISCONNECTED'}</span>
                <span className={l3Data ? "text-green-400" : "text-red-500"}>L3: {l3Data ? 'PRESENT' : 'NULL'}</span>
                <span>PULSES: {pulseCount}</span>
                <span className="text-gray-400">AGE: {((currentTime - lastHeartbeat) / 1000).toFixed(1)}s</span>
            </div>
            {/* Header */}
            <header className="h-10 border-b border-[#E0E0E0] flex items-center px-4 justify-between bg-white z-50 shrink-0">
                <div className="flex items-center gap-6">
                    <div className="flex items-center gap-3">
                        <span className="text-[12px] font-bold uppercase tracking-tight text-[#111111]">Elite Terminal v5.2</span>
                        <div className={`w-2 h-2 rounded-full ${connected && !isOffline ? 'bg-green-500' : 'bg-red-500'} animate-pulse`} />
                    </div>
                    <nav className="flex gap-4">
                        <button onClick={() => setActiveTab('monitor')} className={`text-[11px] font-bold uppercase tracking-widest ${activeTab === 'monitor' ? 'text-blue-700 border-b-2 border-blue-700' : 'text-gray-400'}`}>Monitor</button>
                        <button onClick={() => setActiveTab('analysis')} className={`text-[11px] font-bold uppercase tracking-widest ${activeTab === 'analysis' ? 'text-blue-700 border-b-2 border-blue-700' : 'text-gray-400'}`}>Analysis</button>
                    </nav>
                </div>
                <div className="flex items-center gap-4">
                    <div className="flex flex-col items-end">
                        <span className="text-[8px] font-bold text-gray-400 uppercase">Signal_Age</span>
                        <span className={`text-[10px] font-mono font-bold ${isOffline ? 'text-red-500' : 'text-blue-600'}`}>
                            {((currentTime - lastHeartbeat) / 1000).toFixed(1)}s
                        </span>
                    </div>
                    <div className="h-6 w-px bg-gray-100 mx-1" />
                    <span className="text-[10px] font-mono text-gray-400">PULSES: <span className="text-blue-600 font-bold">{pulseCount}</span></span>
                    {isOffline && connected && <span className="text-[10px] font-bold text-red-500 font-mono animate-pulse">! DATA_STALE !</span>}
                    <button className="h-6 px-3 bg-red-600 text-white text-[10px] font-bold uppercase rounded-[2px] hover:bg-red-700 transition-colors">Flatten All</button>
                </div>
            </header>

            {/* Main Content: Bento Grid Layout */}
            <main className="flex-1 overflow-hidden bg-[#F8F9FA] p-2 font-sans">
                {(!l3Data && !connected) ? (
                    <div className="flex h-full flex-col items-center justify-center gap-6">
                        <div className="flex flex-col items-center gap-2">
                            <div className="w-12 h-12 border-2 border-t-blue-600 border-gray-200 rounded-full animate-spin" />
                            <div className="text-gray-400 font-mono text-[10px] uppercase tracking-[0.2em] animate-pulse mt-4">
                                Establishing Neural Link...
                            </div>
                        </div>
                        <div className="flex flex-col items-center gap-2 text-[9px] text-gray-400 font-mono">
                            <span className="bg-gray-100 px-2 py-1 rounded">STATUS: {isOffline ? 'BACKEND_UNREACHABLE' : 'HANDSHAKE_PENDING'}</span>
                            <span className="opacity-50">awaiting_market_pulse_v2</span>
                        </div>
                        <button
                            onClick={() => setL3Data({ bids: [], asks: [] })}
                            className="mt-6 px-6 py-2 border border-gray-200 bg-white text-[10px] font-bold text-gray-500 uppercase tracking-widest hover:border-blue-500 hover:text-blue-600 transition-all shadow-sm"
                        >
                            Bypass Handshake (Safe Mode)
                        </button>
                    </div>
                ) : activeTab === 'monitor' ? (
                    <div className="max-w-[1920px] mx-auto h-full grid grid-cols-12 gap-2">
                        {/* COLUMN 1: INTELLIGENCE SIDEBAR (20% width -> 2.4/12 -> rounds to 3 (25%) or 2.5?) Let's stick to 20-60-20 ideal. 
                            Grid 12: 3 (25%), 6 (50%), 3 (25%). This is the classic balanced dashboard.
                         */}

                        {/* LEFT PANEL: SIGNALS & DNA (Col 3) */}
                        <div className="col-span-3 flex flex-col gap-2 h-full">
                            {/* Signals (40%) */}
                            <div className="enterprise-card flex flex-col flex-[4] overflow-hidden">
                                <div className="enterprise-header h-8 flex items-center justify-between">
                                    <h3 className="text-[10px] font-bold text-gray-500 uppercase tracking-widest flex items-center gap-2">
                                        <span className="w-1.5 h-1.5 rounded-full bg-blue-500"></span>
                                        Model Signals
                                    </h3>
                                    <span className="text-[9px] font-mono text-gray-300">LIVE</span>
                                </div>
                                <div className="flex-1 p-0 overflow-y-auto  relative min-h-0">
                                    <ErrorBoundary name="SignalPanel"><SignalPanel data={signals} /></ErrorBoundary>
                                </div>
                            </div>
                            {/* Regime (30%) */}
                            <div className="enterprise-card flex flex-col flex-[3] overflow-hidden">
                                <div className="enterprise-header h-8 flex items-center">
                                    <h3 className="text-[10px] font-bold text-gray-500 uppercase tracking-widest">Regime DNA</h3>
                                </div>
                                <div className="flex-1 overflow-y-auto  relative min-h-0">
                                    <ErrorBoundary name="RegimeDNA"><RegimeFingerprint dna={forecastData?.regime_dna} regime={forecastData?.regime} /></ErrorBoundary>
                                </div>
                            </div>
                            {/* Risk (30%) */}
                            <div className="enterprise-card flex flex-col flex-[3] overflow-hidden">
                                <div className="enterprise-header h-8 flex items-center justify-between">
                                    <h3 className="text-[10px] font-bold text-gray-500 uppercase tracking-widest">Risk Guard</h3>
                                    <span className="text-[9px] font-mono text-gray-300">ACTIVE</span>
                                </div>
                                <div className="flex-1 p-2 overflow-y-auto  relative min-h-0">
                                    <ErrorBoundary name="RiskPanel"><RiskPanel data={riskState} vpin={signals?.vpin} /></ErrorBoundary>
                                </div>
                            </div>
                        </div>

                        {/* CENTER PANEL: MARKET DEPTH (Col 6) */}
                        <div className="col-span-6 flex flex-col gap-2 h-full">
                            {/* L3 Heatmap (Main Area - 65%) */}
                            <div className="enterprise-card flex flex-col flex-[65] overflow-hidden shadow-sm relative">
                                <div className="enterprise-header h-8 flex items-center justify-between bg-white border-b border-gray-100">
                                    <div className="flex items-center gap-3">
                                        <h3 className="text-[10px] font-bold text-gray-800 uppercase tracking-widest">L3 Depth</h3>
                                        <span className="px-1.5 py-0.5 rounded bg-gray-100 text-[9px] font-mono text-gray-500">BTC-USDC</span>
                                    </div>
                                    <div className="flex gap-2">
                                        <span className="w-1.5 h-1.5 rounded-sm bg-green-500 opacity-50"></span>
                                        <span className="w-1.5 h-1.5 rounded-sm bg-red-500 opacity-50"></span>
                                    </div>
                                </div>
                                <div className="flex-1 bg-white relative">
                                    <ErrorBoundary name="Heatmap"><OrderBookHeatmap data={l3Data} /></ErrorBoundary>
                                </div>
                            </div>

                            {/* Analytics Strip (Secondary Area - 35%) */}
                            <div className="flex-[35] grid grid-cols-2 gap-2">
                                <div className="enterprise-card flex flex-col overflow-hidden">
                                    <div className="enterprise-header h-8 flex items-center">
                                        <h3 className="text-[10px] font-bold text-gray-500 uppercase tracking-widest">Ensemble Forecast</h3>
                                    </div>
                                    <div className="flex-1 p-2 overflow-y-auto  relative min-h-0">
                                        <ErrorBoundary name="Forecast"><ForecastPanel data={forecastData} /></ErrorBoundary>
                                    </div>
                                </div>
                                <div className="enterprise-card flex flex-col overflow-hidden">
                                    <div className="enterprise-header h-8 flex items-center">
                                        <h3 className="text-[10px] font-bold text-gray-500 uppercase tracking-widest">Alpha Decay</h3>
                                    </div>
                                    <div className="flex-1 p-2 overflow-y-auto  relative min-h-0">
                                        <ErrorBoundary name="AlphaDecay"><AlphaDecay data={forecastData?.alpha_decay} /></ErrorBoundary>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* RIGHT PANEL: EXECUTION (Col 3) */}
                        <div className="col-span-3 flex flex-col gap-2 h-full">
                            {/* Trade Feed (50%) */}
                            <div className="enterprise-card flex flex-col flex-[1] overflow-hidden">
                                <div className="enterprise-header h-8 flex items-center justify-between">
                                    <h3 className="text-[10px] font-bold text-gray-500 uppercase tracking-widest">Market Tape</h3>
                                    <span className="text-[9px] font-mono text-blue-600 animate-pulse">LIVE</span>
                                </div>
                                <div className="flex-1 overflow-hidden bg-white relative">
                                    <ErrorBoundary name="TradeFeed"><TradeFeed trades={trades} /></ErrorBoundary>
                                </div>
                            </div>

                            {/* Execution Meta / Backtest (50%) */}
                            <div className="enterprise-card flex flex-col flex-[1] overflow-hidden">
                                <div className="enterprise-header h-8 flex items-center">
                                    <h3 className="text-[10px] font-bold text-gray-500 uppercase tracking-widest">Execution Engine</h3>
                                </div>
                                <div className="flex-1 p-2 overflow-y-auto relative min-h-0">
                                    <ErrorBoundary name="Backtest"><BacktestPanel data={backtestData} /></ErrorBoundary>
                                </div>
                            </div>
                        </div>
                    </div>
                ) : (
                    <ErrorBoundary name="QuantLab">
                        <div className="rounded-lg border border-gray-200 bg-white h-full overflow-hidden shadow-sm">
                            <QuantLab
                                marketData={l3Data}
                                forecastData={forecastData}
                                mcSimulation={mcSimulation}
                                intelligence_audit={intelligenceAudit}
                            />
                        </div>
                    </ErrorBoundary>
                )}
            </main>

            {/* Footer */}
            <footer className="h-6 border-t border-[#E0E0E0] bg-white flex items-center px-4 justify-between shrink-0">
                <div className="text-[9px] font-bold text-gray-400 uppercase tracking-[0.2em]">Alpha Intelligence Stack v5.2</div>
                <div className="text-[9px] font-mono font-bold text-teal-600">SECURE_LINK // {new Date().toLocaleTimeString()}</div>
            </footer>
        </div>
    );
}

export default App;
