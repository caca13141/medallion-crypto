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

            {/* Main Content */}
            <main className="flex-1 overflow-auto bg-[#F9FAFB] p-3">
                {(!l3Data && !connected) ? (
                    <div className="flex h-full flex-col items-center justify-center gap-4">
                        <div className="text-gray-300 font-mono text-xs uppercase tracking-[0.3em] animate-pulse">
                            Synchronizing Institutional Pipeline...
                        </div>
                        <div className="flex flex-col items-center gap-2 text-[9px] text-gray-400 font-mono">
                            <span>RETRYING_SOCKET_CONNECTION...</span>
                            <span>{isOffline ? 'BACKEND_STALL_DETECTED' : 'WAITING_FOR_DATA_PULSE'}</span>
                        </div>
                        <button
                            onClick={() => setL3Data({ bids: [], asks: [] })}
                            className="mt-4 px-4 py-1 border border-gray-200 text-[10px] text-gray-400 uppercase tracking-widest hover:bg-gray-50 transition-colors"
                        >
                            Force Enter (Safe Mode)
                        </button>
                    </div>
                ) : activeTab === 'monitor' ? (
                    <div className="max-w-[1700px] mx-auto grid grid-cols-12 gap-3 h-full">
                        <div className="col-span-3 flex flex-col gap-3">
                            <div className="enterprise-card h-[40%] flex flex-col">
                                <div className="enterprise-header"><h3 className="text-[10px] font-bold text-gray-500 uppercase tracking-widest">Model Signals</h3></div>
                                <div className="flex-1 p-3 overflow-hidden"><ErrorBoundary name="SignalPanel"><SignalPanel data={signals} /></ErrorBoundary></div>
                            </div>
                            <div className="enterprise-card h-[30%] flex flex-col">
                                <div className="enterprise-header"><h3 className="text-[10px] font-bold text-gray-500 uppercase tracking-widest">Regime DNA</h3></div>
                                <div className="flex-1 overflow-hidden"><ErrorBoundary name="RegimeDNA"><RegimeFingerprint dna={forecastData?.regime_dna} regime={forecastData?.regime} /></ErrorBoundary></div>
                            </div>
                            <div className="enterprise-card h-[30%] flex flex-col">
                                <div className="enterprise-header"><h3 className="text-[10px] font-bold text-gray-500 uppercase tracking-widest">Risk Guard</h3></div>
                                <div className="flex-1 p-3 overflow-hidden"><ErrorBoundary name="RiskPanel"><RiskPanel data={riskState} vpin={signals?.vpin} /></ErrorBoundary></div>
                            </div>
                        </div>

                        <div className="col-span-9 flex flex-col gap-3">
                            <div className="grid grid-cols-12 gap-3 h-[60%]">
                                <div className="enterprise-card col-span-8 flex flex-col overflow-hidden">
                                    <div className="enterprise-header flex justify-between"><h3 className="text-[10px] font-bold text-gray-500 uppercase tracking-widest">L3 Depth (BTC/USDC)</h3></div>
                                    <div className="flex-1 bg-white"><ErrorBoundary name="Heatmap"><OrderBookHeatmap data={l3Data} /></ErrorBoundary></div>
                                </div>
                                <div className="enterprise-card col-span-4 flex flex-col overflow-hidden">
                                    <ErrorBoundary name="TradeFeed"><TradeFeed trades={trades} /></ErrorBoundary>
                                </div>
                            </div>
                            <div className="grid grid-cols-3 gap-3 h-[40%]">
                                <div className="enterprise-card flex flex-col"><div className="enterprise-header"><h3 className="text-[10px] font-bold text-gray-500 uppercase tracking-widest">Ensemble Forecast</h3></div><div className="flex-1 p-2"><ErrorBoundary name="Forecast"><ForecastPanel data={forecastData} /></ErrorBoundary></div></div>
                                <div className="enterprise-card flex flex-col"><div className="enterprise-header"><h3 className="text-[10px] font-bold text-gray-500 uppercase tracking-widest">Alpha Decay</h3></div><div className="flex-1 p-2"><ErrorBoundary name="AlphaDecay"><AlphaDecay data={forecastData?.alpha_decay} /></ErrorBoundary></div></div>
                                <div className="enterprise-card flex flex-col"><div className="enterprise-header"><h3 className="text-[10px] font-bold text-gray-500 uppercase tracking-widest">Execution Meta</h3></div><div className="flex-1 p-2"><ErrorBoundary name="Backtest"><BacktestPanel data={backtestData} /></ErrorBoundary></div></div>
                            </div>
                        </div>
                    </div>
                ) : (
                    <ErrorBoundary name="QuantLab">
                        <QuantLab
                            marketData={l3Data}
                            forecastData={forecastData}
                            mcSimulation={mcSimulation}
                            intelligence_audit={intelligenceAudit}
                        />
                    </ErrorBoundary>
                )}
            </main>

            {/* Footer */}
            <footer className="h-6 border-t border-[#E0E0E0] bg-white flex items-center px-4 justify-between shrink-0">
                <div className="text-[9px] font-bold text-gray-400 uppercase tracking-[0.2em]">Exascale Intelligence Stack Core v5.2</div>
                <div className="text-[9px] font-mono font-bold text-teal-600">SECURE_LINK // {new Date().toLocaleTimeString()}</div>
            </footer>
        </div>
    );
}

export default App;
